#!/usr/bin/env python3
"""Render a registered held-out Ebike target with three models and two protocols.

Reuses verified saved Gaussian reconstructions. Every target comes from the same
COLMAP camera. No-camera reconstructions require a source-only Sim(3) conversion
at rendering time; their inference did not receive source or target cameras.
"""

import argparse
import csv
import gc
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]

import lpips
import numpy as np
from PIL import Image, ImageDraw
import pycolmap
import torch

from depth_anything_3.specs import Gaussians
from depth_anything_3.utils.geometry import affine_inverse
from depth_anything_3.utils.io.input_processor import InputProcessor
from eval_ebike_gsdpt import (
    align_poses_umeyama, colmap_cameras, homogeneous, render,
    ssim_per_image, transform_cameras,
)
from infer_sear_comparison import file_hash
from render_sear_gsdpt import record

MODELS = ('original_da3', 'thermal_adapter', 'rgb_detail_refine')
MODES = ('without_source_cameras', 'with_source_cameras')
LABELS = ('Original DA3', 'Thermal adapter', 'RGB depth detail refine')


def normalize_colmap_target(extrinsics, saved_source, distance_scale):
    """Apply exactly the normalization used to build the known-camera Gaussians."""
    ext = torch.from_numpy(extrinsics.copy()).float()
    first = torch.from_numpy(saved_source['colmap_world_to_camera'][:1].copy()).float()
    transformed = ext @ affine_inverse(first)
    transformed[:, :3, 3] /= distance_scale
    return transformed.numpy()


def save_panels(output, target, metrics, height, width, labels=LABELS):
    for modality in ('rgb', 'thermal'):
        sheet = Image.new('RGB', (width * 3, (height + 46) * 3), 'black')
        draw = ImageDraw.Draw(sheet)
        for row, (model, label) in enumerate(zip(MODELS, labels)):
            for col, mode in enumerate(('reference', *MODES)):
                folder = output / 'reference' if mode == 'reference' else output / model / mode
                with Image.open(folder / f'{modality}.png') as picture:
                    sheet.paste(picture, (col * width, row * (height + 46) + 46))
                title = 'GT ' + target if mode == 'reference' else (
                    label + (' | no source cameras' if mode == MODES[0] else ' | COLMAP source cameras'))
                draw.text((col * width + 5, row * (height + 46) + 5), title, fill='white')
                if mode != 'reference':
                    m = next(r for r in metrics if r['model'] == model and
                             r['mode'] == mode and r['modality'] == modality)
                    draw.text((col * width + 5, row * (height + 46) + 23),
                              f'PSNR {m["psnr"]:.2f}  SSIM {m["ssim"]:.3f}  LPIPS {m["lpips"]:.3f}',
                              fill='white')
        sheet.save(output / f'comparison_{modality}.jpg', quality=95)
        for row, model in enumerate(MODELS):
            sheet.crop((0, row * (height + 46), width * 3, (row + 1) * (height + 46))).save(
                output / model / f'comparison_{modality}.jpg', quality=95)
    preview = Image.new('RGB', (width * 4, (height + 40) * 3), 'black')
    draw = ImageDraw.Draw(preview)
    for row, (model, label) in enumerate(zip(MODELS, labels)):
        for col, (mode, modality) in enumerate((
                (MODES[0], 'rgb'), (MODES[0], 'thermal'),
                (MODES[1], 'rgb'), (MODES[1], 'thermal'))):
            with Image.open(output / model / mode / f'{modality}.png') as picture:
                preview.paste(picture, (col * width, row * (height + 40) + 40))
            text = label + ' | ' + ('No source cameras' if mode == MODES[0] else 'COLMAP source cameras')
            draw.text((col * width + 5, row * (height + 40) + 4), text, fill='white')
            draw.text((col * width + 5, row * (height + 40) + 21),
                      f'{modality.upper()} | {target}', fill='white')
    preview.save(output / 'preview.jpg', quality=95)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--target', default='img_017')
    parser.add_argument('--source-root', type=Path, default=ROOT / 'outputs/novel_views/ThermalGaussian')
    parser.add_argument('--original-no-camera-source', type=Path,
                        help='Override the saved original DA3 reconstruction without source cameras')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if Path(args.target).name != args.target:
        parser.error('--target must be a registered image basename')
    target = Path(args.target).stem
    torch.set_num_threads(4)
    originals = {
        'original_da3': args.source_root / 'Ebike_original_da3_joint_gsdpt_20261013',
        'thermal_adapter': args.source_root / 'Ebike_gsdpt_20261012',
        'rgb_detail_refine': args.source_root / 'Ebike_rgb_detail_refine_gsdpt_20261013',
    }
    if args.original_no_camera_source is not None:
        originals['original_da3'] = args.original_no_camera_source
    calibrated_root = args.source_root / 'Ebike_colmap_sources_20261013'
    calibrated_info = json.loads((calibrated_root / 'run_config.json').read_text())
    configs, sources = {}, {}
    for model in MODELS:
        for mode in MODES:
            source = originals[model] if mode == MODES[0] else calibrated_root / model
            config = json.loads((source / 'run_config.json').read_text())
            if config['status'] != 'complete':
                raise ValueError('A source reconstruction is incomplete')
            stems = [Path(v['image']).stem for v in config['scene']['views']]
            if target in stems:
                raise ValueError('Target RGB/thermal is present in reconstruction inputs; rerun without it')
            if config['input_sha256'] != calibrated_info['input_sha256']:
                raise ValueError('Reconstruction input tensors differ')
            if mode == MODES[1] and not config['source_camera_conditioning']:
                raise ValueError('Expected source-camera-conditioned reconstruction')
            if mode == MODES[0] and config.get('source_camera_conditioning', False):
                raise ValueError('Expected reconstruction without input source cameras')
            configs[model, mode], sources[model, mode] = config, source
    first = configs[MODELS[0], MODES[0]]
    height, width = first['input_hw']
    views = first['scene']['views']
    n = len(views) // 2
    input_names = [Path(v['image']).stem for v in views[:n]]
    if any([Path(v['image']).stem for v in c['scene']['views'][:n]] != input_names
           for c in configs.values()):
        raise ValueError('Reconstruction RGB view order differs')
    # Verify current source pixels once per distinct source path set.
    checked_inputs = set()
    for c in configs.values():
        paths = tuple(v['image'] for v in c['scene']['views'])
        if paths not in checked_inputs:
            normalized, _, _ = InputProcessor()(list(paths), process_res=518, sequential=True)
            if normalized.ndim == 5:
                normalized = normalized[0]
            digest = hashlib.sha256(normalized.numpy().tobytes()).hexdigest()
            if digest != c['input_sha256']:
                raise ValueError('Current source image tensors changed since inference')
            checked_inputs.add(paths)
    dataset = Path(views[0]['image']).parents[2]
    reconstruction = pycolmap.Reconstruction(str(dataset / 'colmap/sparse/0'))
    input_ext, _ = colmap_cameras(reconstruction, input_names, height, width)
    target_ext, target_intr = colmap_cameras(reconstruction, [target], height, width)
    args.output.mkdir(parents=True, exist_ok=False)
    reference_dir = args.output / 'reference'
    reference_dir.mkdir()
    references, reference_files = {}, {}
    for modality, directory, suffix in [('rgb', 'images', '.jpg'), ('thermal', 'thermal', '.png')]:
        path = dataset / 'VGGT' / directory / (target + suffix)
        with Image.open(path) as image:
            image = image.convert('RGB')
            if image.size != (width, height):
                raise ValueError('Target reference grid differs')
            image.save(reference_dir / f'{modality}.png')
            references[modality] = np.asarray(image).astype(np.float32) / 255
        reference_files[modality] = {'path': str(path), 'sha256': file_hash(path)}
    info = {
        'status': 'rendering', 'target': target, 'target_in_reconstruction_inputs': False,
        'source_view_names': input_names, 'source_rgb_thermal_pairs': n,
        'input_sha256': first['input_sha256'], 'input_hw': [height, width],
        'source_inference': 'Reuse the six saved original/adapter Gaussian reconstructions; no new optimization',
        'target_camera': 'Same registered COLMAP target camera in all six cases, expressed in each Gaussian coordinate frame',
        'target_camera_used_in_reconstruction': False,
        'reference_images_used_in_reconstruction': False,
        'reference_files': reference_files,
        'no_camera_alignment': 'Render-time Umeyama Sim(3) using only corresponding 12 RGB SOURCE camera centers, no target fitting',
        'metric_protocol': 'Float32 [0,1] renders; full image; no crop/mask/exposure/color fitting; standard 3DGS SSIM; LPIPS AlexNet v0.1',
        'script_sha256': file_hash(Path(__file__)), 'cases': [],
    }
    record(args.output / 'run_config.json', info)
    np.savez_compressed(args.output / 'target_camera_colmap.npz',
                        world_to_camera=target_ext, intrinsics=target_intr)
    with np.load(calibrated_root / 'source_cameras.npz') as src:
        saved_sources = {k: src[k] for k in src.files}
    distance_scale = calibrated_info['normalization']['colmap_distance_scale']
    normalized_target = normalize_colmap_target(target_ext, saved_sources, distance_scale)
    normalized_inputs = normalize_colmap_target(
        saved_sources['colmap_world_to_camera'], saved_sources, distance_scale)
    np.testing.assert_array_equal(normalized_inputs, saved_sources['world_to_camera'])
    print(f'Verified {n} RGB + {n} thermal source inputs; {target} excluded in all six cases', flush=True)
    perceptual = lpips.LPIPS(net='alex', version='0.1').cuda().eval()
    metrics = []
    for model in MODELS:
        for mode in MODES:
            source, config = sources[model, mode], configs[model, mode]
            folder = args.output / model / mode
            folder.mkdir(parents=True)
            with np.load(source / 'predictions.npz') as saved:
                predicted = saved['extrinsics'][:n]
            if mode == MODES[0]:
                r, t, s = align_poses_umeyama(predicted, input_ext, ransac=False)
                render_ext = transform_cameras(target_ext, r, t, s)
                mapping = {'method': 'SOURCE-only Umeyama Sim(3)', 'rotation': r.tolist(),
                           'translation': t.tolist(), 'scale': float(s), 'target_used_in_fit': False}
            else:
                render_ext = normalized_target
                np.testing.assert_allclose(predicted, saved_sources['world_to_camera'][:n, :3],
                                           rtol=1e-6, atol=1e-6)
                mapping = {'method': 'Exact saved COLMAP first-camera normalization',
                           'colmap_distance_scale': distance_scale, 'target_used_in_fit': False}
            np.savez_compressed(folder / 'cameras.npz', world_to_camera=render_ext,
                                intrinsics=target_intr, target_colmap_world_to_camera=target_ext,
                                input_extrinsics=predicted)
            case = {'model': model, 'mode': mode, 'source': str(source.resolve()),
                    'source_config_sha256': file_hash(source / 'run_config.json'),
                    'inference_source_cameras_provided': mode == MODES[1],
                    'target_camera_provided_to_renderer': True,
                    'target_used_in_reconstruction': False, 'mapping': mapping,
                    'gaussian_files_sha256': {}}
            if config.get('reference_view') is not None:
                case['reference_view'] = config['reference_view']
            case['ref_view_strategy'] = config.get('ref_view_strategy', 'first' if mode == MODES[1] else None)
            predictions = {}
            for modality in ('rgb', 'thermal'):
                path = source / f'gaussians_{modality}.pt'
                case['gaussian_files_sha256'][modality] = file_hash(path)
                print(f'Rendering {model}/{mode}/{modality} at {target}', flush=True)
                data = torch.load(path, map_location='cuda', weights_only=True, mmap=True)
                gs = Gaussians(**data)
                if gs.means.shape[1] != config['gaussian_counts'][modality]:
                    raise ValueError('Gaussian counts differ from recorded inference')
                with torch.inference_mode():
                    image = render(gs, render_ext, target_intr, height, width)[0]
                    x = torch.from_numpy(image).permute(2, 0, 1)[None].cuda()
                    y = torch.from_numpy(references[modality]).permute(2, 0, 1)[None].cuda()
                    m = {'model': model, 'mode': mode, 'modality': modality,
                         'psnr': float(-10 * torch.log10((x - y).square().mean())),
                         'ssim': float(ssim_per_image(x, y)[0]),
                         'lpips': float(perceptual(x, y, normalize=True).flatten()[0])}
                if not np.isfinite(image).all() or not all(np.isfinite(m[k]) for k in ('psnr', 'ssim', 'lpips')):
                    raise ValueError('Nonfinite render or metric')
                predictions[modality] = image
                metrics.append(m)
                Image.fromarray(np.rint(image * 255).clip(0, 255).astype(np.uint8)).save(folder / f'{modality}.png')
                del gs, data, x, y
                gc.collect()
                torch.cuda.empty_cache()
            np.savez_compressed(folder / 'rendered_float.npz', **predictions)
            record(folder / 'run_config.json', case)
            info['cases'].append(case)
            record(args.output / 'run_config.json', info)
    record(args.output / 'metrics.json', {'target': target, 'per_case': metrics})
    with (args.output / 'metrics.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics[0]))
        writer.writeheader()
        writer.writerows(metrics)
    labels = list(LABELS)
    if first.get('ref_view_strategy') == 'first':
        labels[0] += ' (ref: RGB img_001)'
    save_panels(args.output, target, metrics, height, width, labels=labels)
    info['status'] = 'complete'
    record(args.output / 'run_config.json', info)
    print(json.dumps(metrics, indent=2), flush=True)
    print('Complete: ' + str(args.output.resolve()), flush=True)


if __name__ == '__main__':
    main()
