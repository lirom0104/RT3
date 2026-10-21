#!/usr/bin/env python3
"""Infer original DA3 on recorded joint inputs and render existing camera paths.

Reference paths are carried into the original prediction's coordinates by one
Sim(3) fit to corresponding RGB input camera centers. Intrinsics, frame order,
Gaussian filtering, and the renderer are inherited from the reference runs.
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
import torch

from depth_anything_3.specs import Gaussians
from depth_anything_3.thermal.adapter import load_giant
from depth_anything_3.utils.gsply_helpers import save_gaussian_ply
from depth_anything_3.utils.io.input_processor import InputProcessor
from depth_anything_3.utils.pose_align import (
    align_poses_umeyama, apply_umeyama_alignment_to_ext,
)
from infer_sear_comparison import export_clouds, file_hash
from render_sear_gsdpt import record, render_outputs, split_gaussians


def homogeneous(extrinsics):
    if extrinsics.shape[-2:] == (4, 4):
        return extrinsics
    result = np.broadcast_to(np.eye(4), (*extrinsics.shape[:-2], 4, 4)).copy()
    result[..., :3, :] = extrinsics
    return result


def matched_cameras(predicted, reference, trajectory):
    """Map a saved reference trajectory into the new reconstruction frame."""
    rotation, translation, scale = align_poses_umeyama(
        predicted.astype(np.float64), reference.astype(np.float64), ransac=False)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('Camera alignment returned an invalid scale')
    mapped = apply_umeyama_alignment_to_ext(
        rotation, translation, scale, trajectory.astype(np.float64))
    restored = apply_umeyama_alignment_to_ext(
        rotation.T, -rotation.T @ translation / scale, 1 / scale, mapped)
    np.testing.assert_allclose(restored, trajectory, rtol=2e-5, atol=2e-5)
    aligned_inputs = apply_umeyama_alignment_to_ext(
        rotation, translation, scale, homogeneous(reference).astype(np.float64))
    predicted_poses = np.linalg.inv(homogeneous(predicted))
    aligned_poses = np.linalg.inv(aligned_inputs)
    center_errors = np.linalg.norm(
        predicted_poses[:, :3, 3] - aligned_poses[:, :3, 3], axis=-1)
    relative = predicted_poses[:, :3, :3] @ aligned_poses[:, :3, :3].transpose(0, 2, 1)
    angles = np.degrees(np.arccos(np.clip(
        (np.trace(relative, axis1=1, axis2=2) - 1) / 2, -1, 1)))
    info = {
        'direction': 'reference reconstruction -> original DA3 reconstruction',
        'method': 'Umeyama Sim(3), all 12 corresponding RGB INPUT camera centers; no per-view correction',
        'rotation': rotation.tolist(), 'translation': translation.tolist(),
        'scale': float(scale),
        'input_center_rmse_original_units': float(np.sqrt(np.mean(center_errors ** 2))),
        'mean_input_rotation_error_degrees': float(angles.mean()),
        'input_rotation_error_degrees': angles.tolist(),
        'camera_roundtrip_max_abs_error': float(np.max(np.abs(restored - trajectory))),
        'intrinsics': 'Copied exactly from the reference trajectory',
    }
    return mapped.astype(np.float32), info


def comparisons(output, reference, locations, height, width):
    segments = [p['segment'] for p in locations if p['fraction'] == .5]
    selected = np.linspace(0, len(segments) - 1, min(5, len(segments)), dtype=int)
    for modality in ('rgb', 'thermal'):
        sheet = Image.new('RGB', (2 * width, len(selected) * (height + 38)), 'black')
        draw = ImageDraw.Draw(sheet)
        for row, index in enumerate(selected):
            segment = segments[index]
            filename = f'{modality}_novel_{segment:02d}.png'
            for col, folder in enumerate((reference, output)):
                with Image.open(folder / filename) as picture:
                    if picture.size != (width, height):
                        raise ValueError('Reference midpoint image resolution differs')
                    sheet.paste(picture, (col * width, row * (height + 38) + 38))
                label = 'Reference: ' + reference.name if col == 0 else 'Original DA3: joint inputs'
                draw.text((col * width + 5, row * (height + 38) + 4), label, fill='white')
                draw.text((col * width + 5, row * (height + 38) + 20),
                          f'{modality.upper()} | midpoint {segment:02d}', fill='white')
        sheet.save(output / f'comparison_{modality}.jpg', quality=94)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--process-res', type=int, default=518)
    parser.add_argument('--render-only', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(8)
    references = [path.resolve() for path in args.reference]
    configs = [json.loads((path / 'run_config.json').read_text()) for path in references]
    first = configs[0]
    if len(set(path.name for path in references)) != len(references):
        raise ValueError('Reference names must be distinct')
    if not args.render_only:
        args.output.mkdir(parents=True, exist_ok=False)
        info = {
            'status': 'running', 'model': 'Original DA3-GIANT-1.1 with original pretrained GSDPT',
            'base_checkpoint': first['base_checkpoint'], 'adapter': None,
            'thermal_mask': None, 'rgb_detail_refinement': None,
            'protocol': 'One fresh joint inference on 12 RGB + 12 thermal images, original input order',
            'training': 'No adapter, no depth refinement, no scene optimization',
            'scene': first['scene'], 'seed': first['seed'],
            'ref_view_strategy': first['ref_view_strategy'],
            'process_res': args.process_res, 'infer_gs': True, 'use_ray_pose': False,
            'filter': first['filter'], 'reference_runs': [],
            'script_sha256': file_hash(Path(__file__)),
            'checkpoint_config_sha256': file_hash(Path(first['base_checkpoint']) / 'config.json'),
        }
        record(args.output / 'run_config.json', info)
        normalized = None
        for path, config in zip(references, configs):
            if any(config[key] != first[key] for key in
                   ('base_checkpoint', 'seed', 'ref_view_strategy', 'input_hw', 'input_sha256')):
                raise ValueError('Reference inference settings or inputs differ')
            if config['scene']['selected_pair_indices'] != first['scene']['selected_pair_indices']:
                raise ValueError('Reference pair selections differ')
            views = config['scene']['views']
            candidate, _, _ = InputProcessor()(
                [v['image'] for v in views], process_res=args.process_res, sequential=True)
            if candidate.ndim == 5:
                candidate = candidate[0]
            digest = hashlib.sha256(candidate.numpy().tobytes()).hexdigest()
            if digest != config['input_sha256']:
                raise ValueError(f'Input tensor differs from {path.name}')
            if normalized is not None and not torch.equal(candidate, normalized):
                raise ValueError('Reference input tensors differ')
            normalized = candidate
            info['reference_runs'].append({
                'path': str(path), 'run_config_sha256': file_hash(path / 'run_config.json'),
                'predictions_sha256': file_hash(path / 'predictions.npz'),
                'render_cameras_sha256': file_hash(path / 'render_cameras.npz'),
                'input_sha256': digest, 'input_tensor_exactly_matches': True,
                'input_images': [
                    {'path': v['image'], 'sha256': file_hash(Path(v['image']))}
                    for v in views],
            })
        info['input_sha256'] = first['input_sha256']
        n = len(first['scene']['selected_pair_indices'])
        if [v['modality'] for v in first['scene']['views']] != ['rgb'] * n + ['thermal'] * n:
            raise ValueError('Expected RGB inputs followed by thermal inputs')
        if list(normalized.shape) != [2 * n, 3, *first['input_hw']]:
            raise ValueError('Reference input shape differs')
        print(f'Inputs verified against {len(references)} references: {list(normalized.shape)}', flush=True)
        record(args.output / 'run_config.json', info)
        print('Loading ORIGINAL Giant with pretrained GSDPT, no adapter', flush=True)
        model = load_giant(first['base_checkpoint'], include_gaussians=True).cuda().eval()
        if hasattr(model.backbone.pretrained, 'thermal_camera_token') or any(
                'lora_' in name for name, _ in model.named_parameters()):
            raise ValueError('Original model unexpectedly contains an adapter')
        torch.manual_seed(first['seed'])
        torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            raw = model(normalized[None].cuda(), thermal_mask=None, infer_gs=True,
                        ref_view_strategy=first['ref_view_strategy'], use_ray_pose=False)
        torch.cuda.synchronize()
        info['forward_seconds'] = time.monotonic() - started
        info['peak_memory_gib'] = torch.cuda.max_memory_allocated() / 1024 ** 3
        arrays = {key: raw[key][0].float().cpu().numpy()
                  for key in ('depth', 'depth_conf', 'extrinsics', 'intrinsics')}
        np.savez_compressed(args.output / 'predictions.npz', **arrays)
        height, width = normalized.shape[-2:]
        info['input_hw'] = [height, width]
        gaussians = {name: split_gaussians(raw.gaussians, raw.depth, a, b)
                     for name, a, b in [('rgb', 0, n), ('thermal', n, 2 * n)]}
        info['gaussian_counts'] = {name: gs.means.shape[1] for name, gs in gaussians.items()}
        for name, gs in gaussians.items():
            torch.save({f.name: getattr(gs, f.name).cpu() for f in fields(Gaussians)},
                       args.output / f'gaussians_{name}.pt')
            save_gaussian_ply(
                gs, str(args.output / f'gaussians_{name}_dc.ply'),
                torch.ones((1, 1, gs.means.shape[1], 1), device='cuda'),
                save_sh_dc_only=True, prune_border_gs=False, prune_by_depth_percent=1.)
        mean = normalized.new_tensor([.485, .456, .406])[None, :, None, None]
        std = normalized.new_tensor([.229, .224, .225])[None, :, None, None]
        images = (normalized * std + mean).clamp(0, 1).permute(0, 2, 3, 1).numpy()
        cloud_dir = args.output / 'pointcloud'
        cloud_dir.mkdir()
        export_clouds(cloud_dir, arrays, images, first['scene']['views'], first['seed'], 15000)
        info['status'] = 'inference_complete'
        record(args.output / 'run_config.json', info)
        print('Fresh original Gaussian counts: ' + str(info['gaussian_counts']), flush=True)
        del raw, model, normalized, images, candidate
        gc.collect()
        torch.cuda.empty_cache()
    else:
        info = json.loads((args.output / 'run_config.json').read_text())
        if [r['path'] for r in info['reference_runs']] != [str(p) for p in references]:
            raise ValueError('Resume references differ from the recorded runs')
        for path, saved in zip(references, info['reference_runs']):
            for filename, key in [('run_config.json', 'run_config_sha256'),
                                  ('predictions.npz', 'predictions_sha256'),
                                  ('render_cameras.npz', 'render_cameras_sha256')]:
                if file_hash(path / filename) != saved[key]:
                    raise ValueError('Reference file changed since inference')
        gaussians = {name: Gaussians(**torch.load(
            args.output / f'gaussians_{name}.pt', map_location='cuda', weights_only=True))
            for name in ('rgb', 'thermal')}
        with np.load(args.output / 'predictions.npz') as saved:
            arrays = {key: saved[key] for key in saved.files}
        height, width = info['input_hw']
        n = len(info['scene']['selected_pair_indices'])
    info['status'] = 'rendering'
    info['matched_paths'] = []
    record(args.output / 'run_config.json', info)
    for index, (path, config) in enumerate(zip(references, configs)):
        folder = args.output if index == 0 else args.output / ('matched_' + path.name)
        folder.mkdir(exist_ok=True)
        locations = config['render_frames']
        with np.load(path / 'predictions.npz') as saved:
            reference_inputs = saved['extrinsics'][:n]
        with np.load(path / 'render_cameras.npz') as saved:
            source_ext, intr = saved['world_to_camera'], saved['intrinsics']
        if len(source_ext) != len(locations):
            raise ValueError('Reference trajectory frame metadata differs')
        ext, alignment = matched_cameras(arrays['extrinsics'][:n], reference_inputs, source_ext)
        np.savez_compressed(folder / 'render_cameras.npz', world_to_camera=ext, intrinsics=intr)
        shutil.copy2(path / 'render_cameras.npz', folder / 'source_render_cameras.npz')
        shutil.copy2(path / 'run_config.json', folder / 'source_run_config.json')
        entry = {'reference': str(path), 'output': str(folder.resolve()),
                 'frames': len(locations), 'render_frames': locations,
                 'alignment': alignment, 'intrinsics_exactly_match_reference': True}
        record(folder / 'camera_alignment.json', entry)
        print(f'Rendering reference path {path.name}: {len(locations)} frames; '
              f'input rotation discrepancy {alignment["mean_input_rotation_error_degrees"]:.3f} deg',
              flush=True)
        render_outputs(folder, gaussians, torch.from_numpy(ext).cuda(),
                       torch.from_numpy(intr).cuda(), height, width, locations)
        comparisons(folder, path, locations, height, width)
        info['matched_paths'].append(entry)
        record(args.output / 'run_config.json', info)
    info['status'] = 'complete'
    record(args.output / 'run_config.json', info)
    print('Complete: ' + str(args.output.resolve()), flush=True)


if __name__ == '__main__':
    main()
