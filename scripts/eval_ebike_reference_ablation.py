#!/usr/bin/env python3
"""Control reference modality and position in original/adapter DA3 inference.

Canonical 24-image input order is unchanged. A temporary selector override fixes
the reference index inside the native backbone, before its normal reordering.
No source/target cameras enter reconstruction. Four held-out cameras are mapped
using only the 12 RGB source camera centers, as in the existing evaluation.
"""

import argparse
import csv
from dataclasses import fields
import gc
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]

import lpips
import numpy as np
from PIL import Image, ImageDraw
import pycolmap
import torch

import depth_anything_3.model.dinov2.vision_transformer as vit
from depth_anything_3.specs import Gaussians
from depth_anything_3.thermal.adapter import load_adapter, load_giant
from depth_anything_3.thermal.detail_refinement import refine_rgb_depth
from depth_anything_3.thermal.preservation import frozen_base_forward
from depth_anything_3.utils.io.input_processor import InputProcessor
from eval_ebike_gsdpt import (
    TEST_NAMES, align_poses_umeyama, colmap_cameras, homogeneous,
    render, ssim_per_image, transform_cameras,
)
from infer_sear_comparison import file_hash
from render_sear_gsdpt import record, split_gaussians

CASES = [('rgb_001', 0), ('thermal_001', 12), ('rgb_026', 6), ('thermal_026', 18)]
MODALITIES = ('rgb', 'thermal')


def rotation_angles(a, b):
    relative = a @ b.transpose(0, 2, 1)
    return np.degrees(np.arccos(np.clip((np.trace(relative, axis1=1, axis2=2) - 1) / 2, -1, 1)))


def camera_diagnostics(predicted, input_ext, rotation, translation, scale):
    n = len(input_ext)
    aligned = transform_cameras(input_ext, rotation, translation, scale)
    poses = np.linalg.inv(homogeneous(predicted))
    aligned_poses = np.linalg.inv(aligned)
    rgb_errors = rotation_angles(poses[:n, :3, :3], aligned_poses[:, :3, :3])
    thermal_errors = rotation_angles(poses[n:, :3, :3], aligned_poses[:, :3, :3])
    pair_errors = rotation_angles(poses[:n, :3, :3], poses[n:, :3, :3])
    rgb_centers = poses[:n, :3, 3]
    center_rmse = np.sqrt(np.mean(np.sum((rgb_centers - aligned_poses[:, :3, 3]) ** 2, axis=-1)))
    return {'mean_rgb_source_rotation_error_degrees': float(rgb_errors.mean()),
            'mean_thermal_source_rotation_error_degrees': float(thermal_errors.mean()),
            'mean_rgb_thermal_pair_rotation_difference_degrees': float(pair_errors.mean()),
            'rgb_source_rotation_errors_degrees': rgb_errors.tolist(),
            'thermal_source_rotation_errors_degrees': thermal_errors.tolist(),
            'rgb_thermal_pair_rotation_differences_degrees': pair_errors.tolist(),
            'rgb_source_center_rmse_model_units': float(center_rmse)}


def panels(output, metrics, height, width):
    comparisons = output / 'comparisons'
    comparisons.mkdir()
    for target in TEST_NAMES:
        for modality in MODALITIES:
            sheet = Image.new('RGB', (width * 3, (height + 48) * 2), 'black')
            draw = ImageDraw.Draw(sheet)
            for row, position in enumerate(('001', '026')):
                for col, ref_modality in enumerate(('reference', 'rgb', 'thermal')):
                    case = None if col == 0 else ref_modality + '_' + position
                    folder = output / 'reference' / modality if case is None else output / case / modality
                    with Image.open(folder / f'{target}.png') as picture:
                        sheet.paste(picture, (col * width, row * (height + 48) + 48))
                    label = f'GT {target}' if case is None else f'Ref {ref_modality.upper()} img_{position} | target {target}'
                    draw.text((col * width + 5, row * (height + 48) + 5), label, fill='white')
                    if case is not None:
                        m = next(m for m in metrics if m['case'] == case and m['modality'] == modality and m['target'] == target)
                        draw.text((col * width + 5, row * (height + 48) + 25),
                                  f'PSNR {m["psnr"]:.2f} SSIM {m["ssim"]:.3f} LPIPS {m["lpips"]:.3f}', fill='white')
            sheet.save(comparisons / f'{target}_{modality}.jpg', quality=95)
            if target == 'img_017':
                sheet.save(output / f'comparison_{modality}.jpg', quality=95)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, default=ROOT / 'outputs/novel_views/ThermalGaussian/Ebike_original_da3_joint_gsdpt_20261013')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(8)
    source = json.loads((args.reference / 'run_config.json').read_text())
    views = source['scene']['views']
    n = len(views) // 2
    assert n == 12
    adapter = source.get('adapter')
    sigma = source.get('rgb_detail_refinement', {}).get('sigma_pixels', 0.) if source.get('rgb_detail_refinement') else 0.
    model_name = ('RGB depth detail refine' if sigma else 'Thermal adapter') if adapter else source['model']
    if adapter:
        assert file_hash(Path(adapter)) == source['adapter_sha256']
    assert [v['modality'] for v in views] == ['rgb'] * n + ['thermal'] * n
    assert set(TEST_NAMES).isdisjoint(Path(v['image']).stem for v in views)
    normalized, _, _ = InputProcessor()([v['image'] for v in views], process_res=518, sequential=True)
    if normalized.ndim == 5:
        normalized = normalized[0]
    digest = hashlib.sha256(normalized.numpy().tobytes()).hexdigest()
    assert digest == source['input_sha256']
    height, width = source['input_hw']
    args.output.mkdir(parents=True, exist_ok=False)
    dataset = Path(views[0]['image']).parents[2]
    reconstruction = pycolmap.Reconstruction(str(dataset / 'colmap/sparse/0'))
    source_names = [Path(v['image']).stem for v in views[:n]]
    input_ext, _ = colmap_cameras(reconstruction, source_names, height, width)
    target_ext, target_intr = colmap_cameras(reconstruction, TEST_NAMES, height, width)
    np.savez_compressed(args.output / 'target_camera_colmap.npz', targets=np.array(TEST_NAMES),
                        world_to_camera=target_ext, intrinsics=target_intr)
    references, reference_hashes = {}, {}
    for modality, directory, suffix in [('rgb', 'images', '.jpg'), ('thermal', 'thermal', '.png')]:
        folder = args.output / 'reference' / modality
        folder.mkdir(parents=True)
        pictures = []
        for target in TEST_NAMES:
            path = dataset / 'VGGT' / directory / (target + suffix)
            with Image.open(path) as image:
                image = image.convert('RGB')
                assert image.size == (width, height)
                image.save(folder / f'{target}.png')
                pictures.append(np.asarray(image).astype(np.float32) / 255)
            reference_hashes[modality + '/' + target] = {'path': str(path), 'sha256': file_hash(path)}
        references[modality] = np.stack(pictures)
    info = {'status': 'running', 'model': model_name, 'base_checkpoint': source['base_checkpoint'],
            'adapter': adapter, 'adapter_sha256': source.get('adapter_sha256'), 'rgb_detail_sigma': sigma,
            'scene': source['scene'], 'input_sha256': digest, 'input_hw': [height, width],
            'seed': source['seed'], 'input_order_unchanged': True, 'source_cameras_provided_to_inference': False,
            'target_cameras_provided_to_inference': False, 'targets_excluded_from_inference': TEST_NAMES,
            'reference_selection': 'Fixed original input index, native reorder and restore; no external input permutation',
            'alignment': 'Same source-only Umeyama Sim(3) on 12 RGB source centers in every case; no target fitting',
            'gaussian_filter': source['filter'], 'metric_protocol': 'Same float32 full-image PSNR/3DGS SSIM/LPIPS AlexNet v0.1 as prior evaluation',
            'script_sha256': file_hash(Path(__file__)), 'reference_images': reference_hashes, 'cases': []}
    record(args.output / 'run_config.json', info)
    shutil.copy2(Path(__file__), args.output / 'script_snapshot.py')
    print(f'Verified canonical input tensor {digest}; all four targets excluded', flush=True)
    model = load_giant(source['base_checkpoint'], include_gaussians=True).cuda().eval()
    if adapter:
        load_adapter(model, adapter)
        model.eval()
    else:
        assert not hasattr(model.backbone.pretrained, 'thermal_camera_token')
        assert not any('lora_' in name for name, _ in model.named_parameters())
    mask = torch.tensor([[v['modality'] == 'thermal' for v in views]], dtype=torch.bool, device='cuda') if adapter else None
    perceptual = lpips.LPIPS(net='alex', version='0.1').cuda().eval()
    all_metrics, summaries, replay_checks = [], {}, {}
    for case, index in CASES:
        folder = args.output / case
        folder.mkdir()
        expected = views[index]
        assert expected['modality'] == case.split('_')[0]
        assert Path(expected['image']).stem == 'img_' + case.split('_')[1]
        print(f'Inferring {case}: forced canonical index {index}, {expected["image"]}', flush=True)
        original_selector = vit.select_reference_view
        captured = []

        def fixed_selector(x, strategy):
            assert x.shape[1] == 24
            captured.append(index)
            return torch.full((x.shape[0],), index, dtype=torch.long, device=x.device)

        vit.select_reference_view = fixed_selector
        started = time.monotonic()
        teacher_arrays, refined_depth, refinement, depth_hook = None, None, None, None
        try:
            if sigma:
                torch.manual_seed(source['seed'])
                with frozen_base_forward(model) as teacher, torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                    official = teacher(normalized[None].cuda(), thermal_mask=None, infer_gs=False,
                                       ref_view_strategy='first', use_ray_pose=False)
                teacher_arrays = {k: official[k][0].float().cpu().numpy() for k in ('depth', 'depth_conf')}
                del official

                def refined_gaussian_depth(module, arguments, kwargs):
                    nonlocal refined_depth, refinement
                    refined_depth, refinement = refine_rgb_depth(
                        teacher_arrays['depth'], kwargs['depths'][0].float().cpu().numpy(),
                        mask[0].cpu().numpy(), sigma)
                    kwargs['depths'] = torch.from_numpy(refined_depth)[None].to(kwargs['depths'])
                    return arguments, kwargs

                depth_hook = model.gs_adapter.register_forward_pre_hook(refined_gaussian_depth, with_kwargs=True)
            torch.manual_seed(source['seed'])
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                raw = model(normalized[None].cuda(), thermal_mask=mask, infer_gs=True,
                            ref_view_strategy='first', use_ray_pose=False)
        finally:
            vit.select_reference_view = original_selector
            if depth_hook is not None:
                depth_hook.remove()
        torch.cuda.synchronize()
        assert captured == [index] * (2 if sigma else 1)
        elapsed = time.monotonic() - started
        arrays = {k: raw[k][0].float().cpu().numpy() for k in ('depth', 'depth_conf', 'extrinsics', 'intrinsics')}
        if sigma:
            expected_depth, _ = refine_rgb_depth(teacher_arrays['depth'], arrays['depth'], mask[0].cpu().numpy(), sigma)
            np.testing.assert_array_equal(expected_depth, refined_depth)
            arrays['depth'] = refined_depth
            rgb_mask = ~mask[0].cpu().numpy()
            arrays['depth_conf'][rgb_mask] = np.minimum(arrays['depth_conf'][rgb_mask], teacher_arrays['depth_conf'][rgb_mask])
            raw.depth = torch.from_numpy(refined_depth)[None].to(raw.depth)
        assert all(np.isfinite(a).all() for a in arrays.values())
        np.savez_compressed(folder / 'predictions.npz', **arrays)
        # Validate the two known endpoints against their previous complete inference.
        old_folder = {'thermal_026': args.reference}.get(case)
        if adapter is None and case == 'rgb_001':
            old_folder = args.reference.parent / 'Ebike_original_da3_first_rgb_20261013'
        if old_folder is not None:
            with np.load(old_folder / 'predictions.npz') as old:
                differences = {k: float(np.max(np.abs(arrays[k] - old[k]))) for k in arrays}
            assert all(value == 0 for value in differences.values()), differences
            replay_checks[case] = {'source': str(old_folder.resolve()), 'prediction_max_abs_differences': differences}
        rotation, translation, scale = align_poses_umeyama(arrays['extrinsics'][:n], input_ext, ransac=False)
        render_ext = transform_cameras(target_ext, rotation, translation, scale)
        diagnostics = camera_diagnostics(arrays['extrinsics'], input_ext, rotation, translation, scale)
        np.savez_compressed(folder / 'cameras.npz', targets=np.array(TEST_NAMES), world_to_camera=render_ext,
                            intrinsics=target_intr, target_colmap_world_to_camera=target_ext,
                            input_colmap_world_to_camera=input_ext)
        cfg = {'status': 'running', 'model': model_name, 'adapter': adapter,
               'adapter_sha256': source.get('adapter_sha256'), 'rgb_detail_refinement': refinement,
               'scene': source['scene'], 'input_sha256': digest, 'seed': source['seed'],
               'reference_view': {'index_zero_based': index, **expected}, 'ref_view_strategy': 'fixed_input_index',
               'source_camera_conditioning': False, 'forward_seconds': elapsed, 'gaussian_counts': {},
               'mapping': {'rotation': rotation.tolist(), 'translation': translation.tolist(), 'scale': float(scale),
                           'target_used_in_fit': False}, 'camera_diagnostics': diagnostics}
        rendered = {}
        for modality, start, stop in [('rgb', 0, n), ('thermal', n, 2 * n)]:
            gs = split_gaussians(raw.gaussians, raw.depth, start, stop)
            torch.save({f.name: getattr(gs, f.name).cpu() for f in fields(Gaussians)}, folder / f'gaussians_{modality}.pt')
            cfg['gaussian_counts'][modality] = gs.means.shape[1]
            with torch.inference_mode():
                images = render(gs, render_ext, target_intr, height, width)
                x = torch.from_numpy(images).permute(0, 3, 1, 2).cuda()
                y = torch.from_numpy(references[modality]).permute(0, 3, 1, 2).cuda()
                psnr = -10 * torch.log10((x - y).square().mean(dim=(1, 2, 3)))
                ssim = ssim_per_image(x, y)
                distances = perceptual(x, y, normalize=True).flatten()
            assert np.isfinite(images).all()
            rendered[modality] = images
            image_dir = folder / modality
            image_dir.mkdir()
            for i, target in enumerate(TEST_NAMES):
                Image.fromarray(np.rint(images[i] * 255).clip(0, 255).astype(np.uint8)).save(image_dir / f'{target}.png')
                row = {'case': case, 'reference_modality': expected['modality'], 'reference_position': Path(expected['image']).stem,
                       'target': target, 'modality': modality, 'psnr': float(psnr[i]), 'ssim': float(ssim[i]), 'lpips': float(distances[i])}
                assert all(np.isfinite(row[k]) for k in ('psnr', 'ssim', 'lpips'))
                all_metrics.append(row)
            # Native RGB source renders help distinguish geometry failure from target mapping.
            if modality == 'rgb':
                native = render(gs, homogeneous(arrays['extrinsics'][[0, 6]]), arrays['intrinsics'][[0, 6]], height, width)
                native_dir = folder / 'native_rgb_sources'
                native_dir.mkdir()
                for name, image in zip(('img_001', 'img_026'), native):
                    Image.fromarray(np.rint(image * 255).clip(0, 255).astype(np.uint8)).save(native_dir / f'{name}.png')
            del gs, x, y
        np.savez_compressed(folder / 'rendered_float.npz', targets=np.array(TEST_NAMES), **rendered)
        cfg['status'] = 'complete'
        record(folder / 'run_config.json', cfg)
        summary = {'reference_view': cfg['reference_view'], 'camera_diagnostics': diagnostics}
        for modality in MODALITIES:
            rows = [m for m in all_metrics if m['case'] == case and m['modality'] == modality]
            summary[modality] = {k: float(np.mean([m[k] for m in rows])) for k in ('psnr', 'ssim', 'lpips')}
            summary[modality]['img_017'] = next(m for m in rows if m['target'] == 'img_017')
        summaries[case] = summary
        info['cases'].append({'case': case, **cfg})
        record(args.output / 'run_config.json', info)
        print(json.dumps({'case': case, 'rgb_psnr_mean': summary['rgb']['psnr'],
                          'thermal_psnr_mean': summary['thermal']['psnr'],
                          'rgb_source_rotation_error_degrees': diagnostics['mean_rgb_source_rotation_error_degrees']}, indent=2), flush=True)
        del raw, arrays, rendered, teacher_arrays, refined_depth
        gc.collect()
        torch.cuda.empty_cache()
    factors = {}
    for modality in MODALITIES:
        a, b, c, d = [summaries[key][modality]['psnr'] for key in ('rgb_001', 'thermal_001', 'rgb_026', 'thermal_026')]
        factors[modality] = {'rgb_minus_thermal_reference_psnr_at_001': a-b,
                             'rgb_minus_thermal_reference_psnr_at_026': c-d,
                             'position_026_minus_001_psnr_with_rgb_reference': c-a,
                             'position_026_minus_001_psnr_with_thermal_reference': d-b,
                             'average_rgb_minus_thermal_reference_psnr': ((a-b)+(c-d))/2,
                             'average_position_026_minus_001_psnr': ((c-a)+(d-b))/2,
                             'interaction_difference_of_modality_effects': (c-d)-(a-b)}
    record(args.output / 'metrics.json', {'targets': TEST_NAMES, 'per_case_target': all_metrics})
    with (args.output / 'metrics.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_metrics[0]))
        writer.writeheader()
        writer.writerows(all_metrics)
    record(args.output / 'summary.json', {'cases': summaries, 'factor_comparisons': factors,
                                         'scope': model_name + ' joint RGB/thermal, Ebike, positions 001 and 026, four held-out targets; one fixed seed'})
    record(args.output / 'previous_inference_replay_check.json', replay_checks)
    panels(args.output, all_metrics, height, width)
    checks = []
    for case, index in CASES:
        with np.load(args.output / case / 'rendered_float.npz') as saved:
            for modality in MODALITIES:
                for i, target in enumerate(TEST_NAMES):
                    expected_pixels = np.rint(saved[modality][i] * 255).clip(0, 255).astype(np.uint8)
                    np.testing.assert_array_equal(expected_pixels, np.asarray(Image.open(args.output / case / modality / f'{target}.png')))
        cfg = json.loads((args.output / case / 'run_config.json').read_text())
        assert cfg['input_sha256'] == digest and cfg['reference_view']['index_zero_based'] == index
        checks.append({'case': case, 'fixed_reference_and_rendered_pixels': 'passed'})
    record(args.output / 'verification.json', {'status': 'passed', 'same_24_image_tensor_and_order': True,
                                              'only_reference_index_changed': True, 'targets_excluded': TEST_NAMES,
                                              'checks': checks, 'previous_inferences_reproduced_exactly': list(replay_checks)})
    info['status'] = 'complete'
    record(args.output / 'run_config.json', info)
    print('Reference ablation complete: ' + str(args.output.resolve()), flush=True)


if __name__ == '__main__':
    main()
