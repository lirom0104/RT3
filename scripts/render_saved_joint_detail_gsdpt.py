#!/usr/bin/env python3
"""Render saved multiview-refined depths on an existing GSDPT camera path.

Replays the recorded base model/adapter, replaces both modalities' depths before
GaussianAdapter computes centers and scales, and copies reference cameras exactly.
The Gaussian renderer and native geometric filtering match render_sear_gsdpt.py.
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
from plyfile import PlyData
import torch

from depth_anything_3.thermal.adapter import load_giant, load_adapter
from depth_anything_3.utils.gsply_helpers import save_gaussian_ply
from depth_anything_3.utils.io.input_processor import InputProcessor
from infer_sear_comparison import depth_to_world, write_ply
from render_sear_gsdpt import record, render_outputs, split_gaussians


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024**2), b''):
            result.update(chunk)
    return result.hexdigest()


def verify_and_replay_clouds(output, source, arrays, images, selections, run):
    """Verify that the depth used for GS corresponds exactly to requested PLYs."""
    clouds, checks = [], {}
    for modality in ('rgb', 'thermal'):
        pixels = selections[f'{modality}_pixels']
        views = selections[f'{modality}_views']
        points, colors = [], []
        for i in np.unique(views):
            selected = pixels[views == i]
            world = depth_to_world(arrays['depth'][i], arrays['extrinsics'][i], arrays['intrinsics'][i])
            points.append(world.reshape(-1, 3)[selected])
            colors.append(images[i].reshape(-1, 3)[selected])
        points, colors = np.concatenate(points).astype(np.float32), np.concatenate(colors)
        saved = PlyData.read(source / f'{modality}.ply')['vertex'].data
        expected_xyz = np.column_stack([saved[k] for k in ('x', 'y', 'z')])
        expected_rgb = np.column_stack([saved[k] for k in ('red', 'green', 'blue')])
        np.testing.assert_array_equal(points, expected_xyz)
        np.testing.assert_array_equal(np.rint(colors*255).astype(np.uint8), expected_rgb)
        write_ply(output / f'{modality}.ply', points, colors)
        assert digest(output / f'{modality}.ply') == digest(source / f'{modality}.ply')
        checks[modality] = {'points': len(points), 'xyz_rgb_exact': True, 'file_sha256_exact': True}
        clouds.append((points, colors))
    write_ply(output / 'combined.ply', np.concatenate([v[0] for v in clouds]),
              np.concatenate([v[1] for v in clouds]))
    assert digest(output / 'combined.ply') == digest(source / 'combined.ply')
    checks['combined'] = {'points': sum(len(v[0]) for v in clouds), 'file_sha256_exact': True}
    return checks


def comparisons(output, reference, height, width):
    for modality in ('rgb', 'thermal'):
        canvas = Image.new('RGB', (2*width, 11*(height+32)), 'black')
        draw = ImageDraw.Draw(canvas)
        for i in range(11):
            filename = f'{modality}_novel_{i:02d}.png'
            for j, directory in enumerate((reference, output)):
                with Image.open(directory / filename) as picture:
                    assert picture.size == (width, height)
                    canvas.paste(picture, (j*width, i*(height+32)+32))
                label = 'Previous depth' if j == 0 else 'Joint multiview detail depth'
                draw.text((j*width+8, i*(height+32)+8),
                          f'{modality.upper()} {i:02d} | {label}', fill='white')
        canvas.save(output / f'comparison_{modality}.png')
    canvas = Image.new('RGB', (2*width, 2*(height+32)), 'black')
    draw = ImageDraw.Draw(canvas)
    for row, modality in enumerate(('rgb', 'thermal')):
        for col, directory in enumerate((reference, output)):
            canvas.paste(Image.open(directory / f'{modality}_novel_05.png'),
                         (col*width, row*(height+32)+32))
            label = 'Previous depth' if col == 0 else 'Joint multiview detail depth'
            draw.text((col*width+8, row*(height+32)+8), f'{modality.upper()} novel_05 | {label}', fill='white')
    canvas.save(output / 'comparison_novel_05.png')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--record-dir', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(8)
    config = json.loads((args.record_dir / 'run_config.json').read_text())
    reference_config = json.loads((args.reference / 'run_config.json').read_text())
    scene_name = reference_config['scene']['scene']
    scene_record = args.record_dir / 'scenes/new' / scene_name
    run = json.loads((scene_record / 'run.json').read_text())
    scene = run['scene']
    arrays = dict(np.load(scene_record / 'predictions.npz'))
    base = dict(np.load(args.reference / 'predictions.npz'))
    assert scene == reference_config['scene']
    assert run['input_sha256'] == reference_config['input_sha256']
    assert config['base_checkpoint'] == reference_config['base_checkpoint']
    assert config['adapter_sha256'] == reference_config['adapter_sha256'] == digest(config['adapter'])
    assert run['base_sha256'] == digest(run['base'])
    for item in run['files']:
        path = Path(config['output_root']) / 'new' / scene_name / item['path']
        assert path.stat().st_size == item['bytes'] and digest(path) == item['sha256']
    for key in ('extrinsics', 'intrinsics', 'depth_conf'):
        np.testing.assert_array_equal(arrays[key], base[key])
    assert arrays['depth'].shape == base['depth'].shape
    assert np.isfinite(arrays['depth']).all() and (arrays['depth']>0).all()
    args.output.mkdir(parents=True, exist_ok=False)
    info = {
        'status': 'running', 'scene': scene, 'input_hw': reference_config['input_hw'],
        'base_checkpoint': config['base_checkpoint'], 'adapter': config['adapter'],
        'adapter_sha256': config['adapter_sha256'], 'seed': config['seed'],
        'ref_view_strategy': reference_config['ref_view_strategy'], 'infer_gs': True,
        'source_output': str((Path(config['output_root']) / 'new' / scene_name).resolve()),
        'source_record': str(scene_record.resolve()), 'reference': str(args.reference.resolve()),
        'source_depth_sha256': digest(scene_record / 'predictions.npz'),
        'source_config_sha256': digest(args.record_dir / 'run_config.json'),
        'script_sha256': digest(Path(__file__)),
        'gs_weights': 'Same original Giant pretrained GSDPT and GaussianAdapter as reference',
        'gs_depth_source': 'Exact saved joint-detail RGB and thermal depth replaces base depth '
                           'before GaussianAdapter adds GSDPT offsets and computes means/scales',
        'filter': reference_config['filter'],
        'sampling': 'Same native GS border/depth filter as reference. Source PLY confidence, '
                    'support masks and 15000-point sampling are preserved in replay only.',
        'trajectory': 'Exact copy of reference render_cameras.npz; no re-interpolation or alignment',
        'training': 'No additional training or scene optimization',
        'render_frames': reference_config['render_frames'],
        'depth_refinement_parameters': run['parameters'],
    }
    record(args.output / 'run_config.json', info)
    replay = args.output / 'pointcloud_replay'; replay.mkdir()
    images = np.load(scene_record / 'processed_images.npy')
    selections = dict(np.load(scene_record / 'export_pixels.npz'))
    info['pointcloud_replay'] = verify_and_replay_clouds(
        replay, Path(info['source_output']), arrays, images, selections, run)
    print('Saved depth exactly reproduces requested XYZ/RGB point clouds', flush=True)
    normalized, _, _ = InputProcessor()([v['image'] for v in scene['views']],
                                        process_res=config['process_res'], sequential=True)
    if normalized.ndim == 5:
        normalized = normalized[0]
    input_hash = hashlib.sha256(normalized.numpy().tobytes()).hexdigest()
    assert input_hash == run['input_sha256']
    info['input_sha256'] = input_hash
    print('Loading Giant + recorded adapter, GPU input:', list(normalized.shape), flush=True)
    model = load_giant(config['base_checkpoint'], include_gaussians=True).cuda().eval()
    load_adapter(model, config['adapter']); model.eval()
    thermal_mask = torch.tensor([[v['modality']=='thermal' for v in scene['views']]], device='cuda')
    hook_seen = []
    def replace_depth(module, arguments, kwargs):
        np.testing.assert_array_equal(kwargs['depths'][0].float().cpu().numpy(), base['depth'])
        kwargs['depths'] = torch.from_numpy(arrays['depth'])[None].to(kwargs['depths'])
        np.testing.assert_array_equal(kwargs['depths'][0].float().cpu().numpy(), arrays['depth'])
        hook_seen.append(True)
        return arguments, kwargs
    hook = model.gs_adapter.register_forward_pre_hook(replace_depth, with_kwargs=True)
    torch.manual_seed(config['seed']); torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        raw = model(normalized[None].cuda(), thermal_mask=thermal_mask, infer_gs=True,
                    ref_view_strategy=info['ref_view_strategy'], use_ray_pose=False)
    hook.remove(); torch.cuda.synchronize()
    assert hook_seen == [True]
    for key in ('depth', 'depth_conf', 'extrinsics', 'intrinsics'):
        np.testing.assert_array_equal(raw[key][0].float().cpu().numpy(), base[key])
    info['base_replay_exact'] = True
    info['saved_depth_exactly_used_by_gaussian_adapter'] = True
    info['forward_seconds'] = time.monotonic()-started
    info['peak_memory_gib'] = torch.cuda.max_memory_allocated()/1024**3
    raw.depth = torch.from_numpy(arrays['depth'])[None].to(raw.depth)
    np.savez_compressed(args.output / 'predictions.npz', **arrays)
    height, width = info['input_hw']; n = len(scene['selected_pair_indices'])
    gaussians = {name: split_gaussians(raw.gaussians, raw.depth, a, b)
                 for name, a, b in [('rgb', 0, n), ('thermal', n, 2*n)]}
    info['gaussian_counts'] = {name: gs.means.shape[1] for name, gs in gaussians.items()}
    for name, gs in gaussians.items():
        torch.save({f.name: getattr(gs, f.name).cpu() for f in fields(gs)}, args.output / f'gaussians_{name}.pt')
        save_gaussian_ply(gs, str(args.output / f'gaussians_{name}_dc.ply'),
                          torch.ones((1, 1, gs.means.shape[1], 1), device='cuda'),
                          save_sh_dc_only=True, prune_border_gs=False, prune_by_depth_percent=1.)
    shutil.copy2(args.reference / 'render_cameras.npz', args.output / 'render_cameras.npz')
    assert digest(args.output / 'render_cameras.npz') == digest(args.reference / 'render_cameras.npz')
    info['render_cameras_sha256'] = digest(args.output / 'render_cameras.npz')
    info['render_cameras_byte_identical_to_reference'] = True
    shutil.copy2(Path(__file__), args.output / 'render_script_snapshot.py')
    record(args.output / 'run_config.json', info)
    print('Gaussian counts:', info['gaussian_counts'], flush=True)
    del raw, model, normalized, images, selections
    gc.collect(); torch.cuda.empty_cache()
    with np.load(args.output / 'render_cameras.npz') as cameras:
        ext = torch.from_numpy(cameras['world_to_camera']).cuda()
        intrinsic = torch.from_numpy(cameras['intrinsics']).cuda()
    render_outputs(args.output, gaussians, ext, intrinsic, height, width, info['render_frames'])
    comparisons(args.output, args.reference, height, width)
    info['status'] = 'complete'
    record(args.output / 'run_config.json', info)
    print('Complete:', args.output.resolve(), flush=True)


if __name__ == '__main__':
    main()
