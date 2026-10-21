#!/usr/bin/env python3
"""Replay a saved Ebike result with only the GSDPT depth offset disabled.

Preserves the native GaussianAdapter ray convention and XY offsets. Recomputes
Gaussian centers and world scales from the unchanged base depth, and uses an
exact copy of the source camera trajectory.
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

from depth_anything_3.thermal.adapter import load_adapter, load_giant
from depth_anything_3.utils.gsply_helpers import save_gaussian_ply
from depth_anything_3.utils.io.input_processor import InputProcessor
from render_sear_gsdpt import record, render_outputs, split_gaussians


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b''):
            result.update(chunk)
    return result.hexdigest()


def comparisons(output, source, height, width):
    for modality in ('rgb', 'thermal'):
        canvas = Image.new('RGB', (2 * width, 11 * (height + 32)), 'black')
        draw = ImageDraw.Draw(canvas)
        for row in range(11):
            for column, folder in enumerate((source, output)):
                with Image.open(folder / f'{modality}_novel_{row:02d}.png') as picture:
                    canvas.paste(picture, (column * width, row * (height + 32) + 32))
                label = 'Original depth + GSDPT offset' if column == 0 else 'Base depth; depth offset = 0'
                draw.text((column * width + 6, row * (height + 32) + 9),
                          f'{modality.upper()} {row:02d} | {label}', fill='white')
        canvas.save(output / f'comparison_{modality}.png')
    canvas = Image.new('RGB', (2 * width, 2 * (height + 32)), 'black')
    draw = ImageDraw.Draw(canvas)
    for row, modality in enumerate(('rgb', 'thermal')):
        for column, folder in enumerate((source, output)):
            with Image.open(folder / f'{modality}_novel_05.png') as picture:
                canvas.paste(picture, (column * width, row * (height + 32) + 32))
            label = 'With depth offset' if column == 0 else 'Depth offset = 0; XY retained'
            draw.text((column * width + 6, row * (height + 32) + 9),
                      f'{modality.upper()} | {label}', fill='white')
    canvas.save(output / 'comparison_novel_05.png')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.source = args.source.resolve()
    args.output = args.output.resolve()
    torch.set_num_threads(8)
    source_info = json.loads((args.source / 'run_config.json').read_text())
    if source_info.get('status') != 'complete':
        raise ValueError('Source result is incomplete')
    if digest(source_info['adapter']) != source_info['adapter_sha256']:
        raise ValueError('Adapter checksum differs from source')
    with np.load(args.source / 'predictions.npz') as saved:
        arrays = {key: saved[key] for key in saved.files}
    height, width = source_info['input_hw']
    scene = source_info['scene']
    count = len(scene['selected_pair_indices'])
    normalized, _, _ = InputProcessor()(
        [view['image'] for view in scene['views']],
        process_res=max(height, width), sequential=True)
    if normalized.ndim == 5:
        normalized = normalized[0]
    input_hash = hashlib.sha256(normalized.numpy().tobytes()).hexdigest()
    if input_hash != source_info['input_sha256']:
        raise ValueError('Processed inputs differ from source')
    args.output.mkdir(parents=True, exist_ok=False)
    info = {key: source_info[key] for key in (
        'scene', 'base_checkpoint', 'adapter', 'adapter_sha256', 'seed',
        'ref_view_strategy', 'input_hw', 'filter', 'render_frames')}
    info.update({
        'status': 'running', 'source': str(args.source), 'infer_gs': True,
        'input_sha256': input_hash,
        'script_sha256': digest(Path(__file__)),
        'source_predictions_sha256': digest(args.source / 'predictions.npz'),
        'depth_offset': 'Only raw_gaussians[..., -1] is set to zero before GaussianAdapter',
        'xy_offset': 'Original pretrained GSDPT XY offsets retained exactly',
        'gs_depth_source': 'Exact source base depth; no additive GSDPT depth offset',
        'centers': 'Unchanged native GaussianAdapter unit-ray convention with original XY offsets',
        'scales': 'Native world scales recomputed with base depth; learned scale coefficients unchanged',
        'appearance': 'Full degree-2 SH, opacities and rotations retained exactly',
        'trajectory': 'Byte-identical copy of source render_cameras.npz',
        'training': 'No additional training or scene optimization',
    })
    record(args.output / 'run_config.json', info)
    print('Loading source Giant + thermal adapter; disabling ONLY depth offset', flush=True)
    model = load_giant(info['base_checkpoint'], include_gaussians=True).cuda().eval()
    load_adapter(model, info['adapter'])
    model.eval()
    assert model.gs_adapter.pred_offset_depth and model.gs_adapter.pred_offset_xy
    captured = {}

    def zero_depth_offset(module, arguments, kwargs):
        np.testing.assert_array_equal(kwargs['depths'][0].float().cpu().numpy(), arrays['depth'])
        # Direct forward bypasses registered hooks and verifies the unmodified
        # Gaussian branch against every saved source Gaussian parameter.
        captured['baseline'] = module.forward(*arguments, **kwargs)
        raw = kwargs['raw_gaussians']
        captured['original_depth_offset'] = raw[..., -1].float().cpu()
        captured['old_effective_depth'] = (kwargs['depths'] + raw[..., -1]).float().cpu()
        clean = raw.clone()
        clean[..., -1] = 0
        assert torch.equal(clean[..., :-1], raw[..., :-1])
        assert bool((clean[..., -1] == 0).all())
        assert torch.equal(kwargs['depths'] + clean[..., -1], kwargs['depths'])
        kwargs['raw_gaussians'] = clean
        captured['hook_count'] = captured.get('hook_count', 0) + 1
        return arguments, kwargs

    hook = model.gs_adapter.register_forward_pre_hook(zero_depth_offset, with_kwargs=True)
    thermal_mask = torch.tensor(
        [[view['modality'] == 'thermal' for view in scene['views']]],
        dtype=torch.bool, device='cuda')
    torch.manual_seed(info['seed'])
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    try:
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            raw = model(normalized[None].cuda(), thermal_mask=thermal_mask,
                        infer_gs=True, use_ray_pose=False,
                        ref_view_strategy=info['ref_view_strategy'])
    finally:
        hook.remove()
    torch.cuda.synchronize()
    assert captured['hook_count'] == 1
    for key in ('depth', 'depth_conf', 'extrinsics', 'intrinsics'):
        np.testing.assert_array_equal(raw[key][0].float().cpu().numpy(), arrays[key])
    info['base_predictions_exactly_source'] = True
    info['depth_offset_zero_verified'] = True
    info['other_raw_gs_channels_exactly_retained'] = True
    info['forward_seconds'] = time.monotonic() - started
    info['peak_memory_gib'] = torch.cuda.max_memory_allocated() / 1024**3
    offsets = captured['original_depth_offset'][0].numpy()
    np.savez_compressed(args.output / 'disabled_depth_offsets.npz', depth_offset=offsets)
    info['original_depth_offset_percentiles'] = {
        name: np.percentile(offsets[a:b], [0, 1, 50, 99, 100]).tolist()
        for name, a, b in [('rgb', 0, count), ('thermal', count, 2 * count)]}
    gaussians, checks = {}, {}
    for name, begin, end in [('rgb', 0, count), ('thermal', count, 2 * count)]:
        gs = split_gaussians(raw.gaussians, raw.depth, begin, end)
        baseline = split_gaussians(captured['baseline'], raw.depth, begin, end)
        previous = torch.load(args.source / f'gaussians_{name}.pt',
                              map_location='cpu', weights_only=True, mmap=True)
        for field in fields(gs):
            np.testing.assert_array_equal(getattr(baseline, field.name).cpu().numpy(),
                                          previous[field.name].numpy())
        for key in ('harmonics', 'opacities', 'rotations'):
            np.testing.assert_array_equal(getattr(gs, key).cpu().numpy(), previous[key].numpy())
        assert gs.means.shape[1] == source_info['gaussian_counts'][name]
        # Verify depth-dependent scale change without changing learned scale logits.
        depth = raw.depth[0, begin:end].float().cpu()
        mask = torch.zeros_like(depth, dtype=torch.bool)
        bh, bw = max(1, int(8 / 256 * height)), max(1, int(8 / 256 * width))
        mask[:, bh:-bh, bw:-bw] = True
        cutoff = torch.quantile(depth.flatten(1), .99, dim=1)[:, None, None]
        mask &= torch.isfinite(depth) & (depth > 0) & (depth <= cutoff)
        ratio = (depth / captured['old_effective_depth'][0, begin:end])[mask]
        torch.testing.assert_close(gs.scales.cpu(), previous['scales'] * ratio[None, :, None],
                                   rtol=2e-5, atol=1e-8)
        checks[name] = {'original_gaussian_replay_exact_all_fields': True,
                        'unchanged_harmonics_opacities_rotations': True,
                        'same_gaussian_count': True,
                        'scales_match_base_over_original_gs_depth': True}
        gaussians[name] = gs
        torch.save({field.name: getattr(gs, field.name).cpu() for field in fields(gs)},
                   args.output / f'gaussians_{name}.pt')
        save_gaussian_ply(gs, str(args.output / f'gaussians_{name}_dc.ply'),
                          torch.ones((1, 1, gs.means.shape[1], 1), device='cuda'),
                          save_sh_dc_only=True, prune_border_gs=False, prune_by_depth_percent=1.)
        del baseline, previous
    info['parameter_checks'] = checks
    info['gaussian_counts'] = {name: value.means.shape[1] for name, value in gaussians.items()}
    for name in ('predictions.npz', 'render_cameras.npz'):
        shutil.copy2(args.source / name, args.output / name)
        assert digest(args.source / name) == digest(args.output / name)
    info['predictions_byte_identical_to_source'] = True
    info['render_cameras_byte_identical_to_source'] = True
    shutil.copy2(Path(__file__), args.output / 'render_script_snapshot.py')
    record(args.output / 'run_config.json', info)
    print('Verified unchanged source depth, cameras, appearance and XY channels.', flush=True)
    print('Gaussian counts:', info['gaussian_counts'], flush=True)
    del raw, captured, model, normalized
    gc.collect()
    torch.cuda.empty_cache()
    with np.load(args.output / 'render_cameras.npz') as cameras:
        ext = torch.from_numpy(cameras['world_to_camera']).cuda()
        intr = torch.from_numpy(cameras['intrinsics']).cuda()
    print('Rendering identical source 89-frame camera path', flush=True)
    render_outputs(args.output, gaussians, ext, intr, height, width, info['render_frames'])
    comparisons(args.output, args.source, height, width)
    info['status'] = 'complete'
    record(args.output / 'run_config.json', info)
    print('Complete:', args.output, flush=True)


if __name__ == '__main__':
    main()
