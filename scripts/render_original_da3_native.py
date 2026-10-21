#!/usr/bin/env python3
"""Render saved original DA3 Gaussians along its own predicted RGB cameras."""

import argparse
from dataclasses import fields
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]

import numpy as np
import torch

from depth_anything_3.specs import Gaussians
from infer_sear_comparison import file_hash
from render_sear_gsdpt import record, render_outputs, trajectory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=8)
    parser.add_argument('--render-only', action='store_true')
    args = parser.parse_args()
    if args.steps < 2 or args.steps % 2:
        parser.error('--steps must be even and at least 2')
    torch.set_num_threads(8)
    source = json.loads((args.source / 'run_config.json').read_text())
    if source['adapter'] is not None or source['thermal_mask'] is not None:
        raise ValueError('Expected an original DA3 reconstruction without an adapter')
    if not args.render_only:
        args.output.mkdir(parents=True, exist_ok=False)
        for filename in ('predictions.npz', 'gaussians_rgb.pt', 'gaussians_thermal.pt',
                         'gaussians_rgb_dc.ply', 'gaussians_thermal_dc.ply'):
            shutil.copy2(args.source / filename, args.output / filename)
        info = {key: source[key] for key in (
            'model', 'base_checkpoint', 'adapter', 'thermal_mask', 'rgb_detail_refinement',
            'protocol', 'training', 'scene', 'seed', 'ref_view_strategy', 'process_res',
            'infer_gs', 'use_ray_pose', 'filter', 'input_sha256', 'input_hw', 'gaussian_counts')}
        info.update({
            'status': 'rendering', 'inference_source': str(args.source.resolve()),
            'inference_source_config_sha256': file_hash(args.source / 'run_config.json'),
            'prediction_sha256': file_hash(args.output / 'predictions.npz'),
            'trajectory': 'Native DA3 focus-point interpolation of ORIGINAL DA3 predicted RGB cameras only',
            'camera_alignment': None, 'reference_camera_parameters': None,
            'steps_per_input_interval': args.steps,
            'script_sha256': file_hash(Path(__file__)),
        })
        with np.load(args.output / 'predictions.npz') as saved:
            n = len(info['scene']['selected_pair_indices'])
            inputs_ext = torch.from_numpy(saved['extrinsics'][:n].copy())[None].cuda()
            inputs_int = torch.from_numpy(saved['intrinsics'][:n].copy())[None].cuda()
        ext, intr, locations = trajectory(inputs_ext, inputs_int, args.steps)
        if not bool(torch.isfinite(ext).all() and torch.isfinite(intr).all()):
            raise ValueError('Own-camera interpolation returned nonfinite parameters')
        ext_array, intr_array = ext.cpu().numpy(), intr.cpu().numpy()
        input_frames = list(range(0, len(locations), args.steps))
        np.testing.assert_allclose(ext_array[input_frames, :3], inputs_ext[0].cpu().numpy(),
                                   rtol=1e-4, atol=1e-4)
        np.testing.assert_array_equal(intr_array[input_frames], inputs_int[0].cpu().numpy())
        info['input_camera_endpoint_max_abs_error'] = float(np.max(np.abs(
            ext_array[input_frames, :3] - inputs_ext[0].cpu().numpy())))
        info['input_intrinsic_endpoints_exact'] = True
        info['render_frames'] = locations
        info['render_frame_count'] = len(locations)
        np.savez_compressed(args.output / 'render_cameras.npz',
                            world_to_camera=ext_array, intrinsics=intr_array)
        record(args.output / 'run_config.json', info)
        shutil.copy2(Path(__file__), args.output / 'render_script_snapshot.py')
    else:
        info = json.loads((args.output / 'run_config.json').read_text())
        if info['camera_alignment'] is not None or info['reference_camera_parameters'] is not None:
            raise ValueError('Expected an own-camera trajectory')
        with np.load(args.output / 'render_cameras.npz') as saved:
            ext = torch.from_numpy(saved['world_to_camera']).cuda()
            intr = torch.from_numpy(saved['intrinsics']).cuda()
    gaussians = {name: Gaussians(**torch.load(
        args.output / f'gaussians_{name}.pt', map_location='cuda', weights_only=True))
        for name in ('rgb', 'thermal')}
    for name, gs in gaussians.items():
        if gs.means.shape[1] != info['gaussian_counts'][name]:
            raise ValueError('Saved Gaussian counts differ from the source')
    print(f'Rendering {info["render_frame_count"]} frames using ONLY original DA3 predicted cameras',
          flush=True)
    print(f'Input camera endpoint max error: {info["input_camera_endpoint_max_abs_error"]:.3g}',
          flush=True)
    height, width = info['input_hw']
    render_outputs(args.output, gaussians, ext, intr, height, width, info['render_frames'])
    info['status'] = 'complete'
    record(args.output / 'run_config.json', info)
    print('Complete: ' + str(args.output.resolve()), flush=True)


if __name__ == '__main__':
    main()
