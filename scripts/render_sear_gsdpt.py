#!/usr/bin/env python3
"""Replay a paired comparison scene with pretrained GSDPT and render new views.

Retains the original comparison inputs, adapter, seed, and reference strategy.
RGB and thermal Gaussians are split after joint inference and rendered separately
at the same interpolated RGB camera poses. No per-scene training is performed.
"""

import argparse
from dataclasses import fields
import gc
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw
import torch

from depth_anything_3.model.utils.gs_renderer import render_3dgs
from depth_anything_3.specs import Gaussians
from depth_anything_3.thermal.adapter import load_adapter, load_giant
from depth_anything_3.thermal.detail_refinement import refine_rgb_depth
from depth_anything_3.thermal.preservation import frozen_base_forward
from depth_anything_3.utils.camera_trj_helpers import interpolate_extrinsics, interpolate_intrinsics
from depth_anything_3.utils.geometry import as_homogeneous, affine_inverse
from depth_anything_3.utils.gsply_helpers import save_gaussian_ply
from depth_anything_3.utils.io.input_processor import InputProcessor
from infer_sear_comparison import export_clouds, file_hash
from output_paths import resolve_recorded_output


def record(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def split_gaussians(raw, depths, start, stop):
    _, _, height, width = depths.shape
    # Apply the same geometric selection to rendering and exported parameters.
    # Border trimming follows the native GS exporter; keep the nearest 99% depth.
    depth = depths[0, start:stop]
    mask = torch.zeros_like(depth, dtype=torch.bool)
    bh, bw = max(1, int(8/256*height)), max(1, int(8/256*width))
    mask[:, bh:-bh, bw:-bw] = True
    cutoff = torch.quantile(depth.flatten(1), .99, dim=1)[:, None, None]
    mask &= torch.isfinite(depth) & (depth > 0) & (depth <= cutoff)
    mask = mask.flatten()
    begin, end = start*height*width, stop*height*width
    data = {f.name: getattr(raw, f.name)[:, begin:end][:, mask].float().contiguous()
            for f in fields(Gaussians)}
    valid = torch.ones(data['means'].shape[1], device=depth.device, dtype=torch.bool)
    for value in data.values():
        valid &= torch.isfinite(value[0].reshape(len(valid), -1)).all(dim=-1)
    valid &= (data['scales'][0] > 0).all(dim=-1)
    if not bool(valid.all()):
        raise ValueError('GSDPT returned invalid Gaussian parameters')
    return Gaussians(**data)


def trajectory(extrinsics, intrinsics, steps):
    c2w = affine_inverse(as_homogeneous(extrinsics[0]))
    cameras, lenses, indices = [], [], []
    for i in range(len(c2w)-1):
        t = torch.arange(steps, device=c2w.device, dtype=torch.float32) / steps
        cameras.append(interpolate_extrinsics(c2w[i], c2w[i+1], t))
        lenses.append(interpolate_intrinsics(intrinsics[0, i], intrinsics[0, i+1], t))
        indices.extend([{'segment': i, 'fraction': j/steps} for j in range(steps)])
    cameras.append(c2w[-1:])
    lenses.append(intrinsics[0, -1:])
    indices.append({'segment': len(c2w)-2, 'fraction': 1.0})
    return affine_inverse(torch.cat(cameras)), torch.cat(lenses), indices


def render_outputs(output, gaussians, camera_ext, camera_int, height, width, locations):
    out_h, out_w = height, width
    normalized = camera_int.clone()
    normalized[:, 0] /= width
    normalized[:, 1] /= height
    frames, stills = [], []
    with imageio.get_writer(str(output / 'novel_views.mp4'), fps=24, codec='libx264',
                            quality=8, macro_block_size=2) as writer:
        for index in range(len(camera_ext)):
            pair = Image.new('RGB', (out_w*2, out_h+24), 'black')
            for col, (modality, gs) in enumerate(gaussians.items()):
                with torch.inference_mode():
                    color, _ = render_3dgs(camera_ext[index:index+1], normalized[index:index+1],
                                          (out_h, out_w), gs, num_view=1, color_mode='RGB+ED')
                rgb = (color[0].clamp(0, 1)*255).byte().permute(1, 2, 0).cpu().numpy()
                picture = Image.fromarray(rgb)
                pair.paste(picture, (col*out_w, 24))
                label = f'{modality.upper()} | GSDPT | frame {index:03d}'
                ImageDraw.Draw(pair).text((col*out_w+8, 6), label, fill='white')
                if locations[index]['fraction'] == .5:
                    picture.save(output / f'{modality}_novel_{locations[index]["segment"]:02d}.png')
            writer.append_data(np.array(pair))
            frames.append(pair.resize((out_w*2, out_h+24)))
            if locations[index]['fraction'] == .5:
                stills.append(pair)
            if index % 10 == 0:
                print(f'Rendered {index+1}/{len(camera_ext)} paired frames', flush=True)
    frames[0].save(output/'novel_views.gif', save_all=True, append_images=frames[1:],
                   duration=1000//24, loop=0)
    selected = np.linspace(0, len(stills)-1, min(5, len(stills)), dtype=int)
    sheet = Image.new('RGB', (out_w*2, (out_h+24)*len(selected)), 'black')
    for row, idx in enumerate(selected):
        sheet.paste(stills[idx], (0, row*(out_h+24)))
    sheet.save(output/'preview.jpg', quality=94)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--record-dir', type=Path, default=ROOT/'artifacts/comparison_all_scenes_20261010')
    parser.add_argument('--scene', default='ThermalGaussian/Ebike')
    parser.add_argument('--comparison-mode', default='new',
                        help='Model key in comparison records, e.g. new_data_deal_shared')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=8)
    parser.add_argument('--render-only', action='store_true', help='Render saved Gaussian parameters without model inference')
    args = parser.parse_args()
    if args.steps < 2 or args.steps % 2:
        parser.error('--steps must be positive and even (to save the midpoint views)')
    args.output.mkdir(parents=True, exist_ok=True)
    if args.render_only:
        info = json.loads((args.output/'run_config.json').read_text())
        gaussians = {name: Gaussians(**torch.load(args.output/f'gaussians_{name}.pt',
                                                map_location='cuda', weights_only=True))
                     for name in ('rgb', 'thermal')}
        with np.load(args.output/'render_cameras.npz') as saved:
            ext = torch.from_numpy(saved['world_to_camera']).cuda()
            intr = torch.from_numpy(saved['intrinsics']).cuda()
        height, width = info['input_hw']
        render_outputs(args.output, gaussians, ext, intr, height, width, info['render_frames'])
        info['status'] = 'complete'
        info['trajectory'] = 'Native DA3 focus-point camera interpolation between predicted RGB input cameras'
        record(args.output/'run_config.json', info)
        print('Complete: '+str(args.output.resolve()), flush=True)
        return
    if (args.output/'gaussians_rgb.pt').exists():
        raise FileExistsError('Refusing to replace an existing Gaussian reconstruction')
    config = json.loads((args.record_dir/'run_config.json').read_text())
    if 'models' in config:
        if args.comparison_mode not in config['models']:
            raise ValueError('Unknown comparison model: ' + args.comparison_mode)
        selected_model = config['models'][args.comparison_mode]
        config = {**config, 'adapter': selected_model['adapter'],
                  'adapter_sha256': selected_model['sha256']}
    scene = next(s for s in config['scenes'] if s['scene'] == args.scene)
    if file_hash(config['adapter']) != config['adapter_sha256']:
        raise ValueError('Comparison adapter checksum no longer matches')
    torch.set_num_threads(8)
    info = {'scene': scene, 'base_checkpoint': config['base_checkpoint'], 'adapter': config['adapter'],
            'comparison_mode': args.comparison_mode,
            'comparison_record': str(args.record_dir.resolve()),
            'adapter_sha256': config['adapter_sha256'], 'seed': config['seed'],
            'infer_gs': True, 'gs_weights': 'Original Giant pretrained GSDPT, strict loading',
            'ref_view_strategy': config.get('ref_view_strategy', 'saddle_balanced'),
            'filter': 'Native GS border trimming and per-view nearest 99% depth; no confidence filter',
            'color': 'Joint RGB/thermal inference; Gaussian parameters separated by input modality',
            'trajectory': 'Native DA3 focus-point camera interpolation between predicted RGB input cameras',
            'training': 'No scene optimization; thermal adapter was not trained for Gaussian rendering'}
    record(args.output/'run_config.json', info)
    print('Loading Giant with its pretrained GSDPT branch', flush=True)
    model = load_giant(config['base_checkpoint'], include_gaussians=True).cuda().eval()
    assert model.gs_head is not None and model.gs_adapter is not None
    load_adapter(model, config['adapter'])
    adapter_record = torch.load(config['adapter'], map_location='cpu', weights_only=True, mmap=True)
    info.update({'adapter_epoch': adapter_record['epoch'], 'adapter_step': adapter_record['step'],
                 'adapter_config': adapter_record['adapter_config'],
                 'training_data_root': adapter_record.get('settings', {}).get('data_root')})
    del adapter_record
    model.eval()
    normalized, _, _ = InputProcessor()([v['image'] for v in scene['views']],
                                        process_res=config['process_res'], sequential=True)
    if normalized.ndim == 5:
        normalized = normalized[0]
    input_hash = hashlib.sha256(normalized.numpy().tobytes()).hexdigest()
    baseline = json.loads((args.record_dir/'progress.json').read_text())['completed'][args.comparison_mode+'/'+args.scene]
    assert input_hash == baseline['input_sha256'], 'Replay inputs differ from original comparison'
    mask = torch.tensor([[v['modality']=='thermal' for v in scene['views']]], dtype=torch.bool, device='cuda')
    detail_sigma = config.get('rgb_detail_sigma', 0.)
    teacher_arrays, refined_depth, refinement = None, None, None
    if detail_sigma > 0:
        print(f'Predicting frozen official RGB depth detail (sigma={detail_sigma})', flush=True)
        torch.manual_seed(config['seed'])
        with frozen_base_forward(model) as teacher, torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            official = teacher(normalized[None].cuda(), thermal_mask=None,
                               ref_view_strategy=info['ref_view_strategy'], infer_gs=False,
                               use_ray_pose=False)
        teacher_arrays = {key: official[key][0].float().cpu().numpy()
                          for key in ('depth', 'depth_conf')}
        del official
        # Replace depth before GaussianAdapter computes centers and scales.
        # A post-forward modification would leave rendered geometry unrefined.
        def refined_gaussian_depth(module, arguments, kwargs):
            nonlocal refined_depth, refinement
            refined_depth, refinement = refine_rgb_depth(
                teacher_arrays['depth'], kwargs['depths'][0].float().cpu().numpy(),
                mask[0].cpu().numpy(), detail_sigma)
            kwargs['depths'] = torch.from_numpy(refined_depth)[None].to(kwargs['depths'])
            return arguments, kwargs
        depth_hook = model.gs_adapter.register_forward_pre_hook(refined_gaussian_depth, with_kwargs=True)
    torch.manual_seed(config['seed'])
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    print(f'Running infer_gs=True on {list(normalized.shape)}', flush=True)
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        raw = model(normalized[None].cuda(), thermal_mask=mask,
                    ref_view_strategy=info['ref_view_strategy'], infer_gs=True, use_ray_pose=False)
    if detail_sigma > 0:
        depth_hook.remove()
    torch.cuda.synchronize()
    info.update({'input_sha256': input_hash, 'input_matches_original': True,
                 'forward_seconds': time.monotonic()-started,
                 'peak_memory_gib': torch.cuda.max_memory_allocated()/1024**3})
    assert 'gaussians' in raw, 'No Gaussian output despite infer_gs=True'
    arrays = {key: raw[key][0].float().cpu().numpy() for key in ('depth', 'depth_conf', 'extrinsics', 'intrinsics')}
    if detail_sigma > 0:
        # Match the original detail-refined pointcloud export, including its
        # conservative RGB confidence. Thermal depth remains unchanged.
        expected_depth, expected_refinement = refine_rgb_depth(
            teacher_arrays['depth'], arrays['depth'], mask[0].cpu().numpy(), detail_sigma)
        assert np.array_equal(expected_depth, refined_depth), 'GS and exported depth refinements differ'
        arrays['depth'] = refined_depth
        rgb_mask = ~mask[0].cpu().numpy()
        arrays['depth_conf'][rgb_mask] = np.minimum(arrays['depth_conf'][rgb_mask], teacher_arrays['depth_conf'][rgb_mask])
        raw.depth = torch.from_numpy(refined_depth)[None].to(raw.depth)
        info['rgb_detail_refinement'] = refinement
        info['depth_method'] = config['depth_method']
        info['gs_depth_source'] = 'Detail-refined depth passed to original GaussianAdapter before adding pretrained GSDPT offsets and computing centers/scales'
    np.savez_compressed(args.output/'predictions.npz', **arrays)
    mean = normalized.new_tensor([.485, .456, .406])[None, :, None, None]
    std = normalized.new_tensor([.229, .224, .225])[None, :, None, None]
    images = (normalized*std+mean).clamp(0, 1).permute(0, 2, 3, 1).numpy()
    replay_folder = args.output/'pointcloud_replay'
    replay_folder.mkdir(exist_ok=True)
    export_clouds(replay_folder, arrays, images, scene['views'], config['seed'], config['max_points_per_view'])
    source = resolve_recorded_output(Path(config['output_root'])/args.comparison_mode/args.scene)
    info['pointcloud_replay_matches_original'] = {
        name: file_hash(replay_folder/name)==file_hash(source/name)
        for name in ('rgb.ply', 'thermal.ply', 'combined.ply')}
    assert all(info['pointcloud_replay_matches_original'].values()), 'Pointcloud replay differs from requested comparison version'
    n = len(scene['selected_pair_indices'])
    _, _, height, width = raw.depth.shape
    gaussians = {name: split_gaussians(raw.gaussians, raw.depth, a, b)
                 for name, a, b in [('rgb', 0, n), ('thermal', n, 2*n)]}
    info['gaussian_counts'] = {name: gs.means.shape[1] for name, gs in gaussians.items()}
    info['input_hw'] = [height, width]
    ext, intr, locations = trajectory(raw.extrinsics[:, :n].float(), raw.intrinsics[:, :n].float(), args.steps)
    np.savez_compressed(args.output/'render_cameras.npz', world_to_camera=ext.cpu().numpy(),
                        intrinsics=intr.cpu().numpy())
    info['render_frames'] = locations
    for name, gs in gaussians.items():
        torch.save({f.name: getattr(gs, f.name).cpu() for f in fields(Gaussians)}, args.output/f'gaussians_{name}.pt')
        # Selected parameters already match our filtering; disable further pruning.
        count = gs.means.shape[1]
        save_gaussian_ply(gs, str(args.output/f'gaussians_{name}_dc.ply'),
                          torch.ones((1, 1, count, 1), device='cuda'),
                          save_sh_dc_only=True, prune_border_gs=False, prune_by_depth_percent=1.)
    info['ply_export'] = 'DC-only SH for standard viewers; PT files retain all degree-2 SH coefficients used in rendering'
    record(args.output/'run_config.json', info)
    del raw, model, normalized
    gc.collect()
    torch.cuda.empty_cache()
    print('Gaussian counts: '+str(info['gaussian_counts']), flush=True)
    print('Original pointcloud replay: '+str(info['pointcloud_replay_matches_original']), flush=True)
    render_outputs(args.output, gaussians, ext, intr, height, width, locations)
    info['status'] = 'complete'
    record(args.output/'run_config.json', info)
    print('Complete: '+str(args.output.resolve()), flush=True)


if __name__ == '__main__':
    main()
