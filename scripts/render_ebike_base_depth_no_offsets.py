#!/usr/bin/env python3
"""Use saved base-depth XYZ directly as GS centers, without GSDPT offsets.

Keeps learned SH, opacity, rotation and scale coefficients. World scales are
computed from base depth after zeroing depth/XY offset channels, rather than
retaining oversized scales associated with displaced Gaussian centers.
"""
import argparse
from dataclasses import fields, replace
import gc
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]

import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F

from depth_anything_3.thermal.adapter import load_giant, load_adapter
from depth_anything_3.utils.geometry import as_homogeneous
from depth_anything_3.utils.gsply_helpers import save_gaussian_ply
from depth_anything_3.utils.io.input_processor import InputProcessor
from depth_anything_3.model.utils.gs_renderer import render_3dgs
from depth_anything_3.specs import Gaussians
from infer_sear_comparison import depth_to_world
from render_sear_gsdpt import record, render_outputs, split_gaussians
from render_saved_joint_detail_gsdpt import digest


def native_mask(depth):
    _, _, h, w = depth.shape
    mask = torch.zeros_like(depth, dtype=torch.bool)
    bh, bw = max(1, int(8/256*h)), max(1, int(8/256*w))
    mask[:, :, bh:-bh, bw:-bw] = True
    cutoff = torch.quantile(depth.flatten(2), .99, dim=2)[:, :, None, None]
    return mask & torch.isfinite(depth) & (depth>0) & (depth<=cutoff)


def picture(gs, ext, intrinsic, h, w):
    lens = intrinsic.clone()
    lens[:, 0] /= w; lens[:, 1] /= h
    with torch.inference_mode():
        image, _ = render_3dgs(ext, lens, (h, w), gs, num_view=1, color_mode='RGB+ED')
    return image[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy()


def save(path, values):
    Image.fromarray((np.clip(values, 0, 1)*255).astype(np.uint8)).save(path)


def score(image, reference):
    x = torch.from_numpy(image[20:-20, 20:-20]).permute(2, 0, 1)[None].float()
    y = torch.from_numpy(reference[20:-20, 20:-20]).permute(2, 0, 1)[None].float()
    coordinate = torch.arange(11).float()-5
    window = torch.exp(-coordinate.square()/(2*1.5**2)); window /= window.sum()
    kernel = (window[:, None]*window[None])[None, None].repeat(3, 1, 1, 1)
    blur = lambda value: F.conv2d(value, kernel, padding=5, groups=3)
    mx, my = blur(x), blur(y)
    vx, vy, covariance = blur(x*x)-mx*mx, blur(y*y)-my*my, blur(x*y)-mx*my
    ssim = ((2*mx*my+.01**2)*(2*covariance+.03**2)/
            ((mx*mx+my*my+.01**2)*(vx+vy+.03**2))).mean()
    return {'psnr_db': float(-10*torch.log10((x-y).square().mean())), 'ssim': float(ssim)}


def sheet(output, name, rows, h, w):
    canvas = Image.new('RGB', (w*len(rows[0]), (h+32)*len(rows)), 'black')
    draw = ImageDraw.Draw(canvas)
    for row, panels in enumerate(rows):
        for column, (label, image) in enumerate(panels):
            x, y = column*w, row*(h+32)
            if isinstance(image, Path):
                picture_image = Image.open(image)
            else:
                picture_image = Image.fromarray((np.clip(image, 0, 1)*255).astype(np.uint8))
            canvas.paste(picture_image, (x, y+32))
            draw.text((x+6, y+9), label, fill='white')
    canvas.save(output / name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(8)
    source_info = json.loads((args.source / 'run_config.json').read_text())
    assert source_info['status'] == 'complete'
    original = Path(source_info['reference'])
    original_arrays = dict(np.load(original / 'predictions.npz'))
    arrays = dict(np.load(args.source / 'predictions.npz'))
    scene = source_info['scene']; h, w = source_info['input_hw']
    n = len(scene['selected_pair_indices'])
    for k in ('extrinsics', 'intrinsics', 'depth_conf'):
        np.testing.assert_array_equal(arrays[k], original_arrays[k])
    assert digest(source_info['adapter']) == source_info['adapter_sha256']
    args.output.mkdir(parents=True, exist_ok=False)
    info = {k: source_info[k] for k in ('scene', 'input_hw', 'base_checkpoint', 'adapter',
            'adapter_sha256', 'seed', 'ref_view_strategy', 'input_sha256', 'filter', 'render_frames')}
    info.update({
        'status': 'running', 'source': str(args.source.resolve()), 'infer_gs': True,
        'script_sha256': digest(Path(__file__)),
        'source_predictions_sha256': digest(args.source / 'predictions.npz'),
        'centers': 'Exact standard camera-Z depth backprojection using integer source pixels '
                   'and saved intrinsics/extrinsics, identical to source depth_to_world point clouds',
        'offsets': 'Both GSDPT XY and depth offset channels zeroed before GaussianAdapter',
        'scale': 'Learned scale coefficients retained; native world-scale computation uses '
                 'saved base depth with zero offset, not displaced GS depth',
        'appearance': 'Full degree-2 SH, opacity and rotations exactly equal to source run',
        'sampling': 'Same native GS border and 99% base-depth mask as source run; '
                    'no new confidence/support masking or sampling',
        'trajectory': 'Exact source render_cameras.npz, copied without interpolation',
        'training': 'No retraining, color replacement or scene optimization',
    })
    record(args.output / 'run_config.json', info)
    world = np.stack([depth_to_world(d, e, k) for d, e, k in
                      zip(arrays['depth'], arrays['extrinsics'], arrays['intrinsics'])]).astype(np.float32)
    base_centers = torch.from_numpy(world.reshape(1, -1, 3)).cuda()
    normalized, _, _ = InputProcessor()([v['image'] for v in scene['views']],
                                        process_res=518, sequential=True)
    if normalized.ndim == 5: normalized = normalized[0]
    assert hashlib.sha256(normalized.numpy().tobytes()).hexdigest() == info['input_sha256']
    print('Loading original Giant + saved adapter; removing depth and XY offsets', flush=True)
    model = load_giant(info['base_checkpoint'], include_gaussians=True).cuda().eval()
    load_adapter(model, info['adapter']); model.eval()
    assert model.gs_adapter.pred_offset_depth and model.gs_adapter.pred_offset_xy
    assert model.gs_adapter.d_sh == 9
    capture = {}
    def zero_offsets(module, arguments, kwargs):
        np.testing.assert_array_equal(kwargs['depths'][0].float().cpu().numpy(), original_arrays['depth'])
        kwargs['depths'] = torch.from_numpy(arrays['depth'])[None].to(kwargs['depths'])
        raw = kwargs['raw_gaussians']
        capture['original_gs_depth'] = kwargs['depths'] + raw[..., -1]
        clean = raw.clone(); clean[..., :2] = 0; clean[..., -1] = 0
        assert bool((clean[..., :2]==0).all()) and bool((clean[..., -1]==0).all())
        kwargs['raw_gaussians'] = clean
        capture['zero_offsets_verified'] = True
        return arguments, kwargs
    def base_xyz(module, arguments, kwargs, result):
        # The native adapter uses unit-length rays; keep that result for a small
        # control, then set centers to exactly the source point-cloud convention.
        capture['native_zero_means'] = result.means
        result.means = base_centers
        return result
    pre = model.gs_adapter.register_forward_pre_hook(zero_offsets, with_kwargs=True)
    post = model.gs_adapter.register_forward_hook(base_xyz, with_kwargs=True)
    thermal = torch.tensor([[v['modality']=='thermal' for v in scene['views']]], device='cuda')
    torch.manual_seed(info['seed'])
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        raw = model(normalized[None].cuda(), thermal_mask=thermal, infer_gs=True,
                    ref_view_strategy=info['ref_view_strategy'], use_ray_pose=False)
    pre.remove(); post.remove()
    for k in ('depth', 'depth_conf', 'extrinsics', 'intrinsics'):
        np.testing.assert_array_equal(raw[k][0].float().cpu().numpy(), original_arrays[k])
    raw.depth = torch.from_numpy(arrays['depth'])[None].to(raw.depth)
    masks = native_mask(raw.depth)
    gaussians = {name: split_gaussians(raw.gaussians, raw.depth, a, b)
                 for name, a, b in [('rgb', 0, n), ('thermal', n, 2*n)]}
    checks = {}
    for name, a, b in [('rgb', 0, n), ('thermal', n, 2*n)]:
        gs = gaussians[name]
        selected = masks[:, a:b].flatten(1)
        expected = world[a:b].reshape(-1, 3)[selected[0].cpu().numpy()]
        np.testing.assert_array_equal(gs.means[0].cpu().numpy(), expected)
        previous = torch.load(args.source / f'gaussians_{name}.pt', map_location='cpu', weights_only=True, mmap=True)
        assert gs.means.shape[1] == source_info['gaussian_counts'][name]
        unchanged = {}
        for field in ('harmonics', 'opacities', 'rotations'):
            np.testing.assert_array_equal(getattr(gs, field).cpu().numpy(), previous[field].numpy())
            unchanged[field] = True
        base_depth = raw.depth[:, a:b].flatten(1)[selected]
        old_depth = capture['original_gs_depth'][:, a:b].flatten(1)[selected]
        ratio = (base_depth/old_depth).cpu()
        torch.testing.assert_close(gs.scales.cpu(), previous['scales']*ratio[None, :, None], rtol=2e-5, atol=1e-8)
        radius = np.linalg.norm(expected, axis=1)
        checks[name] = {'points': len(expected), 'xyz_exactly_saved_depth_backprojection': True,
                        'unchanged_fields': unchanged, 'scale_ratio_matches_base_over_offset_depth': True,
                        'origin_radius_p50_p90_p99_max': np.percentile(radius, [50, 90, 99, 100]).tolist(),
                        'fraction_radius_over_100': float((radius>100).mean())}
        torch.save({f.name: getattr(gs, f.name).cpu() for f in fields(gs)}, args.output / f'gaussians_{name}.pt')
        save_gaussian_ply(gs, str(args.output / f'gaussians_{name}_dc.ply'),
                          torch.ones((1, 1, len(expected), 1), device='cuda'),
                          save_sh_dc_only=True, prune_border_gs=False, prune_by_depth_percent=1.)
    info['parameter_checks'] = checks
    info['offsets_zeroed_verified'] = capture['zero_offsets_verified']
    info['gaussian_counts'] = {m: v.means.shape[1] for m, v in gaussians.items()}
    record(args.output / 'run_config.json', info)
    np.savez_compressed(args.output / 'predictions.npz', **arrays)
    shutil.copy2(args.source / 'render_cameras.npz', args.output / 'render_cameras.npz')
    shutil.copy2(Path(__file__), args.output / 'render_script_snapshot.py')
    assert digest(args.output / 'render_cameras.npz') == digest(args.source / 'render_cameras.npz')
    info['render_cameras_byte_identical'] = True
    del model, normalized
    gc.collect(); torch.cuda.empty_cache()
    with np.load(args.output / 'render_cameras.npz') as cameras:
        ext = torch.from_numpy(cameras['world_to_camera']).cuda()
        lens = torch.from_numpy(cameras['intrinsics']).cuda()
    novel_rows, source_rows, metrics = [], [], {}
    for name, a, b in [('rgb', 0, n), ('thermal', n, 2*n)]:
        gs = gaussians[name]
        baseline = Gaussians(**torch.load(args.source / f'gaussians_{name}.pt', map_location='cuda', weights_only=True))
        selected = masks[:, a:b].flatten(1)
        begin, end = a*h*w, b*h*w
        native_means = capture['native_zero_means'][:, begin:end][:, selected[0]].float()
        native = replace(gs, means=native_means)
        strict = replace(gs, scales=baseline.scales)
        rows = {}
        for label, variant in [('native_zero_offsets', native), ('base_xyz_no_offsets', gs),
                               ('position_only_original_scales', strict)]:
            print('Rendering midpoint diagnostic:', name, label, flush=True)
            values = picture(variant, ext[44:45], lens[44:45], h, w)
            save(args.output / f'{name}_novel_05_{label}.png', values)
            rows[label] = values
        target = a+5
        source_ext = as_homogeneous(torch.from_numpy(arrays['extrinsics'][target:target+1])).cuda()
        source_lens = torch.from_numpy(arrays['intrinsics'][target:target+1]).cuda()
        reference = np.asarray(Image.open(scene['views'][target]['image']).convert('RGB'))/255.
        original_picture = picture(baseline, source_ext, source_lens, h, w)
        result_picture = picture(gs, source_ext, source_lens, h, w)
        save(args.output / f'{name}_input_05_with_offsets.png', original_picture)
        save(args.output / f'{name}_input_05_no_offsets.png', result_picture)
        metrics[name] = {'input_view': 'img_022', 'protocol': 'Same saved input camera; '
                         '20px border excluded; input reconstruction, not held-out novel-view quality',
                         'with_offsets': score(original_picture, reference),
                         'base_depth_no_offsets': score(result_picture, reference)}
        print(name, 'input reconstruction metrics:', metrics[name], flush=True)
        novel_rows.append([(f'{name.upper()} | with offsets', args.source / f'{name}_novel_05.png'),
                           ('No offsets: exact base-depth XYZ', rows['base_xyz_no_offsets']),
                           ('No offsets: native unit-ray convention', rows['native_zero_offsets']),
                           ('Only move centers; retain old world scales', rows['position_only_original_scales'])])
        source_rows.append([(f'{name.upper()} input img_022', reference),
                            ('Input camera | with offsets', original_picture),
                            ('Input camera | base depth, no offsets', result_picture)])
        del baseline, native, strict, rows
        gc.collect(); torch.cuda.empty_cache()
    sheet(args.output, 'comparison_novel_05_variants.png', novel_rows, h, w)
    sheet(args.output, 'comparison_input_05.png', source_rows, h, w)
    record(args.output / 'input_reconstruction_metrics.json', metrics)
    del capture, raw, masks, base_centers
    gc.collect(); torch.cuda.empty_cache()
    print('Rendering full identical 89-frame path with exact base-depth XYZ', flush=True)
    render_outputs(args.output, gaussians, ext, lens, h, w, info['render_frames'])
    for name in ('rgb', 'thermal'):
        rows = [[(f'{name.upper()} {i:02d} | with offsets', args.source / f'{name}_novel_{i:02d}.png'),
                 (f'{name.upper()} {i:02d} | base depth, no offsets', args.output / f'{name}_novel_{i:02d}.png')]
                for i in range(11)]
        sheet(args.output, f'comparison_{name}.png', rows, h, w)
    sheet(args.output, 'comparison_novel_05.png',
          [[(f'{name.upper()} | with offsets', args.source / f'{name}_novel_05.png'),
            (f'{name.upper()} | base depth, no offsets', args.output / f'{name}_novel_05.png')]
           for name in ('rgb', 'thermal')], h, w)
    info['status'] = 'complete'; info['input_reconstruction_metrics'] = metrics
    record(args.output / 'run_config.json', info)
    print('Complete:', args.output.resolve(), flush=True)


if __name__ == '__main__':
    main()
