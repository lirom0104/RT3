#!/usr/bin/env python3
"""Replay original/adapter variants with COLMAP source camera conditioning.

COLMAP cameras condition the backbone and construct Gaussian geometry. Native
GSDPT parameters are retained; one RGB-baseline scale converts depths and depth
offsets to the normalized COLMAP frame. No target image or scene optimization.
"""

import argparse
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

import numpy as np
from PIL import Image, ImageDraw
import pycolmap
import torch

from depth_anything_3.thermal.adapter import load_adapter, load_giant
from depth_anything_3.thermal.detail_refinement import refine_rgb_depth
from depth_anything_3.utils.geometry import affine_inverse
from depth_anything_3.utils.gsply_helpers import save_gaussian_ply
from depth_anything_3.utils.io.input_processor import InputProcessor
from eval_ebike_gsdpt import colmap_cameras
from infer_sear_comparison import export_clouds, file_hash
from render_sear_gsdpt import record, render_outputs, split_gaussians, trajectory


def source_inputs(reference):
    config = json.loads((reference / 'run_config.json').read_text())
    views = config['scene']['views']
    native_ext, image_intr, colmap_paths = [], [], set()
    reconstructions = {}
    for view in views:
        image_path = Path(view['image'])
        with Image.open(image_path) as image:
            width, height = image.size
        dataset = image_path.parents[2]
        folder = dataset / 'colmap/sparse/0'
        if folder not in reconstructions:
            reconstructions[folder] = pycolmap.Reconstruction(str(folder))
            colmap_paths.add(folder)
        ext, intr = colmap_cameras(
            reconstructions[folder], [image_path.stem], height, width)
        native_ext.append(ext[0])
        image_intr.append(intr[0])
    normalized, ext, intr = InputProcessor()(
        [v['image'] for v in views], np.stack(native_ext), np.stack(image_intr),
        process_res=518, sequential=True)
    if normalized.ndim == 5:
        normalized = normalized[0]
    digest = hashlib.sha256(normalized.numpy().tobytes()).hexdigest()
    if digest != config['input_sha256']:
        raise ValueError(f'Input tensor differs from {reference}')
    n = len(config['scene']['selected_pair_indices'])
    if [v['modality'] for v in views] != ['rgb'] * n + ['thermal'] * n:
        raise ValueError('Expected RGB inputs followed by thermal inputs')
    torch.testing.assert_close(ext[:n], ext[n:], rtol=0, atol=0)
    torch.testing.assert_close(intr[:n], intr[n:], rtol=0, atol=0)
    return config, normalized, ext, intr, sorted(colmap_paths)


def normalize_cameras(ext):
    # Same normalization as the official DepthAnything3 API: first source frame,
    # then median distance to that camera. This is shared by every model variant.
    relative = ext @ affine_inverse(ext[:1])
    distances = affine_inverse(relative)[:, :3, 3].norm(dim=-1)
    scale = distances.median().clamp_min(.1)
    relative[:, :3, 3] /= scale
    return relative, float(scale)


def depth_scale(predicted, supplied, n):
    pred_centers = affine_inverse(predicted[:, :n])[..., :3, 3]
    known_centers = affine_inverse(supplied[:, :n])[..., :3, 3]
    pairs = torch.triu_indices(n, n, 1, device=predicted.device)
    pred_dist = (pred_centers[:, pairs[0]] - pred_centers[:, pairs[1]]).norm(dim=-1)
    known_dist = (known_centers[:, pairs[0]] - known_centers[:, pairs[1]]).norm(dim=-1)
    valid = (pred_dist[0] > 1e-6) & (known_dist[0] > 1e-6)
    if int(valid.sum()) < 3:
        raise ValueError('Insufficient nonzero input camera baselines for depth scale')
    scale = (known_dist[0, valid] / pred_dist[0, valid]).median()
    if not bool(torch.isfinite(scale)) or float(scale) <= 0:
        raise ValueError('Invalid predicted-to-COLMAP depth scale')
    return scale


def compare_sheets(output, folders, height, width):
    for modality in ('rgb', 'thermal'):
        sheet = Image.new('RGB', (width * len(folders), (height + 30) * 5), 'black')
        draw = ImageDraw.Draw(sheet)
        for row, segment in enumerate((0, 2, 5, 7, 10)):
            for col, folder in enumerate(folders):
                with Image.open(folder / f'{modality}_novel_{segment:02d}.png') as picture:
                    sheet.paste(picture, (col * width, row * (height + 30) + 30))
                draw.text((col * width + 5, row * (height + 30) + 7),
                          f'{folder.name} | {modality} | midpoint {segment:02d}', fill='white')
        sheet.save(output / f'comparison_{modality}.jpg', quality=94)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, action='append', required=True,
                        help='First reference must be the freshly inferred original DA3 run')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(8)
    prepared = [source_inputs(path) for path in args.reference]
    first, normalized, raw_ext, intr, colmap_paths = prepared[0]
    if first.get('adapter') is not None:
        raise ValueError('Original DA3 must be first, for the frozen detail teacher')
    for config, images, ext2, intr2, _ in prepared[1:]:
        if not torch.equal(images, normalized):
            raise ValueError('Variant input tensors differ')
        torch.testing.assert_close(raw_ext, ext2, rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(intr, intr2, rtol=1e-6, atol=1e-6)
        if config['seed'] != first['seed'] or config['base_checkpoint'] != first['base_checkpoint']:
            raise ValueError('Variant foundation or seed differs')
    args.output.mkdir(parents=True, exist_ok=False)
    source_ext, normalization_scale = normalize_cameras(raw_ext)
    n = len(first['scene']['selected_pair_indices'])
    height, width = normalized.shape[-2:]
    np.savez_compressed(args.output / 'source_cameras.npz',
                        colmap_world_to_camera=raw_ext.numpy(),
                        world_to_camera=source_ext.numpy(), intrinsics=intr.numpy())
    summary = {
        'status': 'running', 'input_sha256': first['input_sha256'],
        'input_hw': [height, width], 'source_views': first['scene']['views'],
        'camera_protocol': 'COLMAP source cameras supplied to camera encoder AND used for Gaussian construction',
        'thermal_registration': 'Paired RGB/thermal reuse the corresponding registered RGB COLMAP camera',
        'normalization': {'first_source_camera': 'img_001',
                          'colmap_distance_scale': normalization_scale,
                          'method': 'Official DA3 input extrinsic normalization'},
        'colmap_files': {str(p): file_hash(p) for folder in colmap_paths
                         for p in folder.glob('*.bin')},
        'script_sha256': file_hash(Path(__file__)),
        'camera_token_compatibility': 'On thermal views add learned thermal-minus-base token offset to provided camera token; retain calibrated conditioning',
        'variants': [],
    }
    record(args.output / 'run_config.json', summary)
    shutil.copy2(Path(__file__), args.output / 'render_script_snapshot.py')
    source_ext = source_ext[None].cuda()
    source_intr = intr[None].cuda()
    camera_ext, camera_intr, locations = trajectory(source_ext[:, :n], source_intr[:, :n], 8)
    np.savez_compressed(args.output / 'render_cameras.npz',
                        world_to_camera=camera_ext.cpu().numpy(),
                        intrinsics=camera_intr.cpu().numpy())
    print(f'Verified identical 24-image inputs; COLMAP-conditioned resolution {height}x{width}', flush=True)
    model = load_giant(first['base_checkpoint'], include_gaussians=True).cuda().eval()
    teacher_depth, teacher_confidence = None, None
    folders = []
    for index, (reference, entry) in enumerate(zip(args.reference, prepared)):
        config = entry[0]
        adapter = config.get('adapter')
        sigma = config.get('rgb_detail_refinement', {}).get('sigma_pixels', 0.) if config.get('rgb_detail_refinement') else 0.
        name = 'original_da3' if adapter is None else ('rgb_detail_refine' if sigma else 'thermal_adapter')
        folder = args.output / name
        folder.mkdir(exist_ok=False)
        folders.append(folder)
        if adapter:
            if file_hash(Path(adapter)) != config['adapter_sha256']:
                raise ValueError('Adapter checkpoint checksum differs from the reference')
            load_adapter(model, adapter)
            model.eval()
        mask = torch.tensor([[v['modality'] == 'thermal' for v in first['scene']['views']]],
                            dtype=torch.bool, device='cuda') if adapter else None
        memo = {}

        def calibrated_gaussian_inputs(module, arguments, kwargs):
            native_depth = kwargs['depths'][0].float().cpu().numpy()
            used_depth = native_depth
            if sigma:
                used_depth, refinement = refine_rgb_depth(
                    teacher_depth, native_depth, mask[0].cpu().numpy(), sigma)
                memo['rgb_detail_refinement'] = refinement
            scale = depth_scale(kwargs['extrinsics'].float(), source_ext, n)
            memo['depth_to_colmap_scale'] = float(scale)
            memo['unscaled_depth'] = native_depth
            memo['used_depth'] = torch.from_numpy(used_depth)[None].to(kwargs['depths']) * scale
            raw_gs = kwargs['raw_gaussians']
            if module.pred_offset_depth:
                raw_gs = torch.cat([raw_gs[..., :-1], raw_gs[..., -1:] * scale], dim=-1)
            indices = torch.linspace(0, height * width - 1, 96, device='cuda').long()
            memo['sample_indices'] = indices
            memo['sample_offsets'] = raw_gs[0, ..., :2].reshape(2 * n, -1, 2)[:, indices]
            kwargs.update(extrinsics=source_ext, intrinsics=source_intr,
                          depths=memo['used_depth'], raw_gaussians=raw_gs, gt_extrinsics=None)
            return arguments, kwargs

        hook = model.gs_adapter.register_forward_pre_hook(calibrated_gaussian_inputs, with_kwargs=True)
        torch.manual_seed(first['seed'])
        torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        print(f'Running {name}: provided COLMAP E/K, thermal adapter={bool(adapter)}, sigma={sigma}', flush=True)
        try:
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                raw = model(normalized[None].cuda(), extrinsics=source_ext,
                            intrinsics=source_intr, thermal_mask=mask, infer_gs=True,
                            ref_view_strategy='first', use_ray_pose=False)
        finally:
            hook.remove()
        torch.cuda.synchronize()
        seconds = time.monotonic() - started
        indices = memo['sample_indices']
        means = raw.gaussians.means[0].reshape(2 * n, -1, 3)[:, indices]
        camera_points = torch.einsum('vij,vnj->vni', source_ext[0, :, :3, :3], means)
        camera_points += source_ext[0, :, None, :3, 3]
        pixels = torch.einsum('vij,vnj->vni', source_intr[0], camera_points)
        pixels = pixels[..., :2] / pixels[..., 2:]
        expected = torch.stack((indices % width + .5, indices // width + .5), dim=-1)
        expected = expected[None] + memo['sample_offsets']
        error = (pixels - expected).abs().max().item()
        if not np.isfinite(error) or error > .05:
            raise ValueError(f'Gaussian/source-camera reprojection error: {error} pixels')
        arrays = {key: raw[key][0].float().cpu().numpy()
                  for key in ('depth_conf', 'extrinsics', 'intrinsics')}
        predicted_ext, predicted_intr = arrays['extrinsics'], arrays['intrinsics']
        arrays.update(depth=memo['used_depth'][0].float().cpu().numpy(),
                      extrinsics=source_ext[0, :, :3].cpu().numpy(),
                      intrinsics=source_intr[0].cpu().numpy(),
                      predicted_extrinsics=predicted_ext, predicted_intrinsics=predicted_intr,
                      unscaled_predicted_depth=memo['unscaled_depth'])
        if sigma:
            arrays['depth_conf'][:n] = np.minimum(arrays['depth_conf'][:n], teacher_confidence[:n])
        if index == 0:
            teacher_depth = memo['unscaled_depth'].copy()
            teacher_confidence = arrays['depth_conf'].copy()
        np.savez_compressed(folder / 'predictions.npz', **arrays)
        gaussians = {modality: split_gaussians(raw.gaussians, memo['used_depth'], a, b)
                     for modality, a, b in [('rgb', 0, n), ('thermal', n, 2 * n)]}
        counts = {modality: gs.means.shape[1] for modality, gs in gaussians.items()}
        info = {
            'status': 'rendering', 'variant': name, 'reference': str(reference.resolve()),
            'reference_config_sha256': file_hash(reference / 'run_config.json'),
            'base_checkpoint': first['base_checkpoint'], 'adapter': adapter,
            'adapter_sha256': config.get('adapter_sha256'),
            'scene': first['scene'], 'input_sha256': first['input_sha256'],
            'input_tensor_exactly_matches_reference': True,
            'input_hw': [height, width], 'seed': first['seed'],
            'source_camera_input': str((args.output / 'source_cameras.npz').resolve()),
            'source_camera_conditioning': True, 'gaussian_source_cameras': 'Provided normalized COLMAP E/K',
            'camera_token_mode': summary['camera_token_compatibility'] if adapter else 'Original provided camera encoder',
            'depth_scale_method': 'Median corresponding pairwise RGB source baseline ratio, no depth labels',
            'depth_to_colmap_scale': memo['depth_to_colmap_scale'],
            'gaussian_source_reprojection_max_error_pixels': error,
            'rgb_detail_refinement': memo.get('rgb_detail_refinement'),
            'gaussian_counts': counts, 'filter': config['filter'],
            'forward_seconds': seconds, 'peak_memory_gib': torch.cuda.max_memory_allocated() / 1024 ** 3,
            'trajectory': 'Native focus-point interpolation of provided COLMAP RGB source cameras',
            'render_frames': locations, 'scene_optimization_steps': 0,
        }
        record(folder / 'run_config.json', info)
        for modality, gs in gaussians.items():
            torch.save({f.name: getattr(gs, f.name).cpu() for f in fields(gs)},
                       folder / f'gaussians_{modality}.pt')
            save_gaussian_ply(gs, str(folder / f'gaussians_{modality}_dc.ply'),
                              torch.ones((1, 1, gs.means.shape[1], 1), device='cuda'),
                              save_sh_dc_only=True, prune_border_gs=False, prune_by_depth_percent=1.)
        cloud_dir = folder / 'pointcloud'
        cloud_dir.mkdir()
        mean = normalized.new_tensor([.485, .456, .406])[None, :, None, None]
        std = normalized.new_tensor([.229, .224, .225])[None, :, None, None]
        images = (normalized * std + mean).clamp(0, 1).permute(0, 2, 3, 1).numpy()
        export_clouds(cloud_dir, arrays, images, first['scene']['views'], first['seed'], 15000)
        shutil.copy2(args.output / 'source_cameras.npz', folder / 'source_cameras.npz')
        shutil.copy2(args.output / 'render_cameras.npz', folder / 'render_cameras.npz')
        del raw, means, camera_points, pixels, memo, arrays, images
        gc.collect()
        torch.cuda.empty_cache()
        print(f'{name}: Gaussian counts {counts}; calibrated reprojection error {error:.5g} pixels', flush=True)
        render_outputs(folder, gaussians, camera_ext, camera_intr, height, width, locations)
        info['status'] = 'complete'
        record(folder / 'run_config.json', info)
        summary['variants'].append(info)
        record(args.output / 'run_config.json', summary)
        del gaussians, gs
        gc.collect()
        torch.cuda.empty_cache()
    compare_sheets(args.output, folders, height, width)
    summary['status'] = 'complete'
    record(args.output / 'run_config.json', summary)
    print('Complete: ' + str(args.output.resolve()), flush=True)


if __name__ == '__main__':
    main()
