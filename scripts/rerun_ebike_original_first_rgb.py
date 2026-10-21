#!/usr/bin/env python3
"""Infer original DA3 with the first RGB reference and update a target comparison.

Only original_da3/without_source_cameras is replaced. The other five renders are
retained byte for byte. RGB and thermal use the same fresh joint reconstruction.
The previous complete comparison is archived before updating any result files.
"""

import argparse
import csv
from dataclasses import fields
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]

import lpips
import numpy as np
from PIL import Image
import pycolmap
import torch

import depth_anything_3.model.dinov2.vision_transformer as vit
from depth_anything_3.specs import Gaussians
from depth_anything_3.thermal.adapter import load_giant
from depth_anything_3.utils.io.input_processor import InputProcessor
from eval_ebike_gsdpt import (
    align_poses_umeyama, colmap_cameras, homogeneous, render,
    ssim_per_image, transform_cameras,
)
from infer_sear_comparison import file_hash
from render_ebike_target_six_modes import LABELS, MODELS, MODES, save_panels
from render_sear_gsdpt import record, split_gaussians


def infer(reference, output, target):
    source = json.loads((reference / 'run_config.json').read_text())
    assert source['adapter'] is None and source['thermal_mask'] is None
    views = source['scene']['views']
    n = len(views) // 2
    assert [v['modality'] for v in views] == ['rgb'] * n + ['thermal'] * n
    assert Path(views[0]['image']).name == 'img_001.jpg'
    assert target not in {Path(v['image']).stem for v in views}
    normalized, _, _ = InputProcessor()(
        [v['image'] for v in views], process_res=source['process_res'], sequential=True)
    if normalized.ndim == 5:
        normalized = normalized[0]
    digest = hashlib.sha256(normalized.numpy().tobytes()).hexdigest()
    assert digest == source['input_sha256']
    output.mkdir(parents=True, exist_ok=False)
    info = {key: source[key] for key in (
        'model', 'base_checkpoint', 'adapter', 'thermal_mask', 'rgb_detail_refinement',
        'protocol', 'training', 'scene', 'seed', 'process_res', 'infer_gs',
        'use_ray_pose', 'filter', 'input_sha256', 'input_hw')}
    info.update(status='running', ref_view_strategy='first', source_camera_conditioning=False,
                inference_source=str(reference.resolve()),
                inference_source_config_sha256=file_hash(reference / 'run_config.json'),
                input_tensor_exactly_matches_previous=True,
                target_used_in_reconstruction=False, target_camera_used_in_reconstruction=False,
                script_sha256=file_hash(Path(__file__)))
    record(output / 'run_config.json', info)
    print(f'Exact input hash verified: {digest}; target {target} excluded', flush=True)
    model = load_giant(source['base_checkpoint'], include_gaussians=True).cuda().eval()
    assert not hasattr(model.backbone.pretrained, 'thermal_camera_token')
    assert not any('lora_' in name for name, _ in model.named_parameters())
    original_selector = vit.select_reference_view

    def capture_reference(x, strategy):
        selected = original_selector(x, strategy=strategy)
        index = int(selected[0].item())
        assert strategy == 'first' and index == 0
        info['reference_view'] = {'index_zero_based': index, **views[index]}
        print('Actual reference: ' + json.dumps(info['reference_view']), flush=True)
        return selected

    vit.select_reference_view = capture_reference
    torch.manual_seed(source['seed'])
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    try:
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            raw = model(normalized[None].cuda(), thermal_mask=None, infer_gs=True,
                        ref_view_strategy='first', use_ray_pose=False)
    finally:
        vit.select_reference_view = original_selector
    torch.cuda.synchronize()
    assert info['reference_view']['index_zero_based'] == 0
    info['forward_seconds'] = time.monotonic() - started
    info['peak_memory_gib'] = torch.cuda.max_memory_allocated() / 1024 ** 3
    arrays = {key: raw[key][0].float().cpu().numpy()
              for key in ('depth', 'depth_conf', 'extrinsics', 'intrinsics')}
    assert all(np.isfinite(value).all() for value in arrays.values())
    np.savez_compressed(output / 'predictions.npz', **arrays)
    info['gaussian_counts'], info['gaussian_files_sha256'] = {}, {}
    for modality, start, stop in [('rgb', 0, n), ('thermal', n, 2 * n)]:
        gs = split_gaussians(raw.gaussians, raw.depth, start, stop)
        path = output / f'gaussians_{modality}.pt'
        torch.save({f.name: getattr(gs, f.name).cpu() for f in fields(Gaussians)}, path)
        info['gaussian_counts'][modality] = gs.means.shape[1]
        info['gaussian_files_sha256'][modality] = file_hash(path)
        del gs
    info['status'] = 'complete'
    record(output / 'run_config.json', info)
    shutil.copy2(Path(__file__), output / 'inference_script_snapshot.py')
    print(f'Fresh first-RGB inference complete: {info["forward_seconds"]:.1f}s; '
          f'Gaussians {info["gaussian_counts"]}', flush=True)
    del raw, model, normalized
    gc.collect()
    torch.cuda.empty_cache()
    return info, arrays['extrinsics'][:n]


def verify(output, info, unchanged):
    height, width = info['input_hw']
    checks = []
    known_targets = []
    with np.load(output / 'target_camera_colmap.npz') as saved:
        target_ext, target_intr = saved['world_to_camera'], saved['intrinsics']
    for case in info['cases']:
        folder = output / case['model'] / case['mode']
        cfg = json.loads((Path(case['source']) / 'run_config.json').read_text())
        assert cfg['input_sha256'] == info['input_sha256']
        assert info['target'] not in {Path(v['image']).stem for v in cfg['scene']['views']}
        assert file_hash(Path(case['source']) / 'run_config.json') == case['source_config_sha256']
        with np.load(folder / 'cameras.npz') as saved:
            np.testing.assert_array_equal(saved['target_colmap_world_to_camera'], target_ext)
            np.testing.assert_array_equal(saved['intrinsics'], target_intr)
            assert np.isfinite(saved['world_to_camera']).all()
            if case['mode'] == MODES[1]:
                known_targets.append(saved['world_to_camera'].copy())
        with np.load(folder / 'rendered_float.npz') as saved:
            for modality in ('rgb', 'thermal'):
                pixels = saved[modality]
                assert pixels.shape == (height, width, 3) and np.isfinite(pixels).all()
                np.testing.assert_array_equal(
                    np.asarray(Image.open(folder / f'{modality}.png')),
                    np.rint(pixels * 255).clip(0, 255).astype(np.uint8))
        checks.append({'model': case['model'], 'mode': case['mode'],
                       'images_and_target_camera': 'passed'})
    for value in known_targets[1:]:
        np.testing.assert_array_equal(value, known_targets[0])
    for relative, digest in unchanged.items():
        assert file_hash(output / relative) == digest, relative
    return {'status': 'passed', 'cases': 6, 'rendered_images': 12,
            'target_excluded_from_all_source_inputs': True,
            'physical_target_camera': info['target'] + ' COLMAP camera', 'checks': checks,
            'with_camera_target_matrices_identical': True,
            'other_five_case_files_unchanged': True,
            'original_no_camera_reference': 'RGB img_001.jpg, first, index 0'}


def update_comparison(output, reconstruction, config, predicted):
    info = json.loads((output / 'run_config.json').read_text())
    target = info['target']
    height, width = info['input_hw']
    assert config['input_sha256'] == info['input_sha256']
    unchanged = {str(path.relative_to(output)): file_hash(path)
                 for model in MODELS for mode in MODES
                 if (model, mode) != ('original_da3', MODES[0])
                 for path in (output / model / mode).rglob('*') if path.is_file()}
    dataset = Path(config['scene']['views'][0]['image']).parents[2]
    colmap = pycolmap.Reconstruction(str(dataset / 'colmap/sparse/0'))
    input_names = [Path(v['image']).stem for v in config['scene']['views'][:len(predicted)]]
    input_ext, _ = colmap_cameras(colmap, input_names, height, width)
    with np.load(output / 'target_camera_colmap.npz') as saved:
        target_ext, target_intr = saved['world_to_camera'], saved['intrinsics']
    actual_ext, actual_intr = colmap_cameras(colmap, [target], height, width)
    np.testing.assert_array_equal(target_ext, actual_ext)
    np.testing.assert_array_equal(target_intr, actual_intr)
    r, t, s = align_poses_umeyama(predicted, input_ext, ransac=False)
    render_ext = transform_cameras(target_ext, r, t, s)
    mapping = {'method': 'SOURCE-only Umeyama Sim(3)', 'rotation': r.tolist(),
               'translation': t.tolist(), 'scale': float(s), 'target_used_in_fit': False}
    aligned = transform_cameras(input_ext, r, t, s)
    rotations = np.linalg.inv(homogeneous(predicted))[:, :3, :3]
    aligned_rotations = np.linalg.inv(aligned)[:, :3, :3]
    relative = rotations @ aligned_rotations.transpose(0, 2, 1)
    angles = np.degrees(np.arccos(np.clip((np.trace(relative, axis1=1, axis2=2) - 1) / 2, -1, 1)))
    case = {'model': 'original_da3', 'mode': MODES[0], 'source': str(reconstruction.resolve()),
            'source_config_sha256': file_hash(reconstruction / 'run_config.json'),
            'inference_source_cameras_provided': False, 'target_camera_provided_to_renderer': True,
            'target_used_in_reconstruction': False, 'mapping': mapping,
            'ref_view_strategy': 'first', 'reference_view': config['reference_view'],
            'gaussian_files_sha256': config['gaussian_files_sha256'],
            'mean_source_rotation_error_degrees': float(angles.mean())}
    stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    with tempfile.TemporaryDirectory(prefix='.update_original_first_', dir=output) as temporary:
        stage = Path(temporary) / 'comparison'
        shutil.copytree(output, stage, ignore=shutil.ignore_patterns('_history', '.update*'))
        folder = stage / 'original_da3' / MODES[0]
        np.savez_compressed(folder / 'cameras.npz', world_to_camera=render_ext,
                            intrinsics=target_intr, target_colmap_world_to_camera=target_ext,
                            input_extrinsics=predicted)
        perceptual = lpips.LPIPS(net='alex', version='0.1').cuda().eval()
        predictions, new_metrics = {}, []
        for modality in ('rgb', 'thermal'):
            print(f'Rendering fresh original DA3/{modality} at {target}', flush=True)
            data = torch.load(reconstruction / f'gaussians_{modality}.pt', map_location='cuda',
                              weights_only=True, mmap=True)
            gs = Gaussians(**data)
            assert gs.means.shape[1] == config['gaussian_counts'][modality]
            with torch.inference_mode():
                image = render(gs, render_ext, target_intr, height, width)[0]
                x = torch.from_numpy(image).permute(2, 0, 1)[None].cuda()
                reference = np.asarray(Image.open(output / 'reference' / f'{modality}.png')).astype(np.float32) / 255
                y = torch.from_numpy(reference).permute(2, 0, 1)[None].cuda()
                metric = {'model': 'original_da3', 'mode': MODES[0], 'modality': modality,
                          'psnr': float(-10 * torch.log10((x - y).square().mean())),
                          'ssim': float(ssim_per_image(x, y)[0]),
                          'lpips': float(perceptual(x, y, normalize=True).flatten()[0])}
            assert np.isfinite(image).all()
            assert all(np.isfinite(metric[k]) for k in ('psnr', 'ssim', 'lpips'))
            predictions[modality] = image
            new_metrics.append(metric)
            Image.fromarray(np.rint(image * 255).clip(0, 255).astype(np.uint8)).save(folder / f'{modality}.png')
            del data, gs, x, y
            gc.collect()
            torch.cuda.empty_cache()
        np.savez_compressed(folder / 'rendered_float.npz', **predictions)
        record(folder / 'run_config.json', case)
        metrics_info = json.loads((stage / 'metrics.json').read_text())
        metrics_info['per_case'] = [next(m for m in new_metrics if m['modality'] == row['modality'])
                                   if (row['model'], row['mode']) == ('original_da3', MODES[0]) else row
                                   for row in metrics_info['per_case']]
        metrics = metrics_info['per_case']
        record(stage / 'metrics.json', metrics_info)
        with (stage / 'metrics.csv').open('w') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(metrics[0]))
            writer.writeheader()
            writer.writerows(metrics)
        info['cases'] = [case if (c['model'], c['mode']) == ('original_da3', MODES[0]) else c
                         for c in info['cases']]
        backup = output / '_history' / ('before_original_first_rgb_' + stamp)
        info['source_inference'] = 'Original DA3 without source cameras freshly rerun with first RGB reference; other five saved reconstructions retained'
        info['last_update'] = {'updated_at_utc': stamp, 'case': 'original_da3/without_source_cameras',
                               'ref_view_strategy': 'first', 'reference': 'RGB img_001.jpg',
                               'backup': str(backup.resolve()), 'script_sha256': file_hash(Path(__file__))}
        record(stage / 'run_config.json', info)
        save_panels(stage, target, metrics, height, width,
                    labels=('Original DA3 (ref: RGB img_001)', *LABELS[1:]))
        record(stage / 'verification.json', verify(stage, info, unchanged))
        replay = json.loads((stage / 'previous_no_camera_replay_check.json').read_text())
        replay['original_da3'] = {'status': 'superseded',
                                  'reason': 'Reference changed from saddle_balanced to first RGB; previous replay check archived'}
        record(stage / 'previous_no_camera_replay_check.json', replay)
        readme = (stage / 'README.md').read_text()
        readme = readme.replace('复用已保存并已校验的六份 Gaussian 重建参数，不重新训练或推理。',
                                '原始 DA3 无源相机组已重新推理，固定第一张 RGB img_001.jpg 为参考（ref_view_strategy=first）。另外五组复用已保存的 Gaussian 重建参数。')
        readme = readme.replace('--target img_017 --output outputs/novel_views/ThermalGaussian/Ebike_target_img017_reproduce',
                                '--target img_017 --original-no-camera-source ' + str(reconstruction.resolve()) +
                                ' --output outputs/novel_views/ThermalGaussian/Ebike_target_img017_reproduce')
        readme += '\n本次更新：仅替换 original_da3/without_source_cameras 的 RGB/红外、目标坐标转换、指标及对应对照图。原始 DA3 有源相机组原本已使用第一张 RGB 作为参考。更新前完整结果保存在 ' + str(backup.resolve()) + '。\n'
        (stage / 'README.md').write_text(readme)
        report = {'status': 'passed', 'previous_case': json.loads((output / 'original_da3' / MODES[0] / 'run_config.json').read_text()),
                  'new_case': case, 'new_metrics': new_metrics,
                  'other_five_case_files_sha256': unchanged, 'backup': str(backup.resolve())}
        record(stage / 'first_rgb_reference_update.json', report)
        shutil.copy2(Path(__file__), stage / 'first_rgb_update_script_snapshot.py')
        shutil.copytree(output, backup, ignore=shutil.ignore_patterns('_history', '.update*'))
        # Publish only the changed case and its accompanying metadata/panels.
        changed = [path.relative_to(stage) for path in folder.iterdir() if path.is_file()]
        changed += [Path(name) for name in ('comparison_rgb.jpg', 'comparison_thermal.jpg', 'preview.jpg',
                                            'metrics.json', 'metrics.csv', 'run_config.json', 'verification.json',
                                            'previous_no_camera_replay_check.json', 'README.md',
                                            'first_rgb_reference_update.json', 'first_rgb_update_script_snapshot.py')]
        changed += [Path('original_da3') / name for name in ('comparison_rgb.jpg', 'comparison_thermal.jpg')]
        for relative in changed:
            shutil.copy2(stage / relative, output / relative)
        verify(output, info, unchanged)
        print(json.dumps({'metrics': new_metrics, 'mean_source_rotation_error_degrees': float(angles.mean()),
                          'backup': str(backup.resolve()), 'comparison': str((output / 'comparison_rgb.jpg').resolve())}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--reconstruction-output', type=Path, required=True)
    parser.add_argument('--comparison-output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(8)
    current = json.loads((args.comparison_output / 'run_config.json').read_text())
    assert current['status'] == 'complete'
    config, predicted = infer(args.reference, args.reconstruction_output, current['target'])
    update_comparison(args.comparison_output, args.reconstruction_output, config, predicted)


if __name__ == '__main__':
    main()
