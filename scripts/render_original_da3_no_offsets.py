#!/usr/bin/env python3
"""Replay an official RGB-only GSDPT run with both XY and depth offsets zeroed.

Preserves the native unit-ray convention, SH, opacity and rotations. Recomputes
world scales from unchanged base depth and retains the exact source trajectory.
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
from plyfile import PlyData
import depth_anything_3.model.dinov2.vision_transformer as vit
from depth_anything_3.thermal.adapter import load_giant
from depth_anything_3.specs import Gaussians
from depth_anything_3.utils.gsply_helpers import save_gaussian_ply
from depth_anything_3.utils.io.input_processor import InputProcessor
from render_sear_gsdpt import split_gaussians
from render_original_da3_single_modality import render_single, verify_video


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def record(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')


def mask_for(depth):
    h, w = depth.shape[-2:]
    mask = torch.zeros_like(depth, dtype=torch.bool)
    bh, bw = max(1, int(8/256*h)), max(1, int(8/256*w))
    mask[:, bh:-bh, bw:-bw] = True
    cutoff = torch.quantile(depth.flatten(1), .99, dim=1)[:, None, None]
    return mask & torch.isfinite(depth) & (depth > 0) & (depth <= cutoff)


def comparisons(source, output, height, width):
    sheet = Image.new('RGB', (2*width, 11*(height+28)), 'black')
    draw = ImageDraw.Draw(sheet)
    for row in range(11):
        for col, folder in enumerate((source, output)):
            with Image.open(folder / f'rgb_novel_{row:02d}.png') as frame:
                sheet.paste(frame, (col*width, row*(height+28)+28))
            label = 'Native depth + XY offsets' if col == 0 else 'Depth offset = 0; XY offsets = 0'
            draw.text((col*width+6, row*(height+28)+8), f'{row:02d} | {label}', fill='white')
    sheet.save(output / 'comparison_rgb.png')
    row_height = height+28
    sheet.crop((0, 5*row_height, 2*width, 6*row_height)).save(output / 'comparison_novel_05.png')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    source_info = json.loads((source / 'run_config.json').read_text())
    assert source_info['status'] == 'complete' and source_info['adapter'] is None
    views = source_info['scene']['views']
    assert len(views) == 12 and all(v['modality'] == 'rgb' for v in views)
    source_paths = [source / name for name in ('run_config.json', 'predictions.npz',
                    'render_cameras.npz', 'gaussians_rgb.pt', 'gaussians_rgb_dc.ply')]
    source_hashes = {p.name: digest(p) for p in source_paths}
    arrays = dict(np.load(source / 'predictions.npz'))
    height, width = source_info['input_hw']
    assert arrays['depth'].shape == (12, height, width)
    normalized, _, _ = InputProcessor()([v['image'] for v in views],
                                       process_res=source_info.get('process_res', 518), sequential=True)
    if normalized.ndim == 5: normalized = normalized[0]
    assert hashlib.sha256(normalized.numpy().tobytes()).hexdigest() == source_info['input_sha256']
    output.mkdir(parents=True, exist_ok=False)
    info = {k: v for k, v in source_info.items() if k not in
            ('status', 'script_sha256', 'depth_offset', 'xy_offset', 'appearance', 'training', 'trajectory')}
    info.update({'status': 'running', 'source': str(source), 'source_sha256': source_hashes,
                 'script_sha256': digest(Path(__file__)),
                 'depth_offset': 'raw_gaussians[..., -1] set to zero before native GaussianAdapter',
                 'xy_offset': 'raw_gaussians[..., :2] set to zero before native GaussianAdapter',
                 'centers': 'Native normalized unit rays at source pixel centers, multiplied by unchanged base depth',
                 'scales': 'Native world scales recomputed from base depth; learned scale coefficients unchanged',
                 'appearance': 'Complete degree-2 SH, opacity and rotation exactly retained from source',
                 'trajectory': 'Byte-identical source render_cameras.npz; same 89 cameras',
                 'training': 'No adapter, retraining, color replacement or scene optimization'})
    record(output / 'run_config.json', info)
    torch.set_num_threads(8)
    print('Loading original RGB-only Giant + GSDPT; zeroing BOTH depth and XY offsets', flush=True)
    model = load_giant(source_info['base_checkpoint'], include_gaussians=True).cuda().eval()
    assert not any('lora_' in n for n, _ in model.named_parameters())
    assert model.gs_adapter.pred_offset_depth and model.gs_adapter.pred_offset_xy
    captured = {}
    original_selector = vit.select_reference_view
    def capture_reference(x, strategy):
        selected = original_selector(x, strategy=strategy)
        idx = int(selected[0].item())
        expected = source_info['reference_view']['index_zero_based']
        assert idx == expected
        captured['reference_view_verified'] = True
        print('Verified reference: ' + views[idx]['image'], flush=True)
        return selected
    def zero_offsets(module, arguments, kwargs):
        np.testing.assert_array_equal(kwargs['depths'][0].float().cpu().numpy(), arrays['depth'])
        captured['baseline'] = module.forward(*arguments, **kwargs)
        raw = kwargs['raw_gaussians']
        captured['original_effective_depth'] = (kwargs['depths'] + raw[..., -1]).float().cpu()
        captured['depth_offset'] = raw[0, ..., -1].float().cpu().numpy()
        captured['xy_offset'] = raw[0, ..., :2].float().cpu().numpy()
        clean = raw.clone()
        clean[..., :2] = 0
        clean[..., -1] = 0
        assert torch.equal(clean[..., 2:-1], raw[..., 2:-1])
        assert bool((clean[..., :2] == 0).all() and (clean[..., -1] == 0).all())
        kwargs['raw_gaussians'] = clean
        captured['zero_offsets_verified'] = True
        captured['hook_count'] = captured.get('hook_count', 0) + 1
        return arguments, kwargs
    vit.select_reference_view = capture_reference
    hook = model.gs_adapter.register_forward_pre_hook(zero_offsets, with_kwargs=True)
    torch.manual_seed(source_info['seed'])
    started = time.monotonic()
    try:
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            raw = model(normalized[None].cuda(), thermal_mask=None, infer_gs=True,
                        ref_view_strategy=source_info['ref_view_strategy'], use_ray_pose=False)
    finally:
        hook.remove()
        vit.select_reference_view = original_selector
    torch.cuda.synchronize()
    info['forward_seconds'] = time.monotonic()-started
    assert captured['hook_count'] == 1
    for k, values in arrays.items():
        np.testing.assert_array_equal(raw[k][0].float().cpu().numpy(), values)
    gs = split_gaussians(raw.gaussians, raw.depth, 0, 12)
    baseline = split_gaussians(captured['baseline'], raw.depth, 0, 12)
    previous = torch.load(source / 'gaussians_rgb.pt', map_location='cpu', mmap=True, weights_only=True)
    for field in fields(Gaussians):
        np.testing.assert_array_equal(getattr(baseline, field.name).cpu().numpy(), previous[field.name].numpy())
    for name in ('harmonics', 'opacities', 'rotations'):
        np.testing.assert_array_equal(getattr(gs, name).cpu().numpy(), previous[name].numpy())
    mask = mask_for(raw.depth[0]).cpu()
    ratio = (raw.depth[0].float().cpu() / captured['original_effective_depth'][0])[mask]
    torch.testing.assert_close(gs.scales.cpu(), previous['scales']*ratio[None, :, None], rtol=2e-5, atol=1e-8)
    assert gs.means.shape[1] == source_info['gaussian_counts']['rgb']
    # Independently verify unit-ray means and zero image-space displacement.
    points = gs.means[0].cpu().numpy()
    offsets = np.cumsum([0, *mask.flatten(1).sum(1).tolist()])
    max_center_error = 0.
    max_pixel_error = 0.
    for i in range(12):
        yy, xx = np.nonzero(mask[i].numpy())
        pixels = np.column_stack((xx+.5, yy+.5, np.ones(len(xx))))
        ext, intr = arrays['extrinsics'][i], arrays['intrinsics'][i]
        camera_rays = pixels @ np.linalg.inv(intr).T
        camera_rays /= np.linalg.norm(camera_rays, axis=1, keepdims=True)
        c2w = np.linalg.inv(np.vstack((ext, [0, 0, 0, 1])))
        expected = c2w[:3, 3] + (camera_rays @ c2w[:3, :3].T)*arrays['depth'][i, yy, xx, None]
        actual = points[offsets[i]:offsets[i+1]]
        max_center_error = max(max_center_error, float(np.max(np.abs(expected-actual))))
        projected = (actual @ ext[:3, :3].T + ext[:3, 3]) @ intr.T
        pixel_error = np.linalg.norm(projected[:, :2]/projected[:, 2, None]-pixels[:, :2], axis=1)
        max_pixel_error = max(max_pixel_error, float(pixel_error.max()))
    assert max_center_error < 1e-5, max_center_error
    assert max_pixel_error < .01, max_pixel_error
    torch.save({f.name: getattr(gs, f.name).cpu() for f in fields(Gaussians)}, output / 'gaussians_rgb.pt')
    save_gaussian_ply(gs, str(output / 'gaussians_rgb_dc.ply'),
                      torch.ones((1, 1, len(points), 1), device='cuda'),
                      save_sh_dc_only=True, prune_border_gs=False, prune_by_depth_percent=1.)
    for name in ('predictions.npz', 'render_cameras.npz'):
        shutil.copy2(source / name, output / name)
        assert digest(output / name) == source_hashes[name]
    np.savez_compressed(output / 'disabled_offsets.npz', depth_offset=captured['depth_offset'], xy_offset=captured['xy_offset'])
    shutil.copy2(Path(__file__), output / 'render_script_snapshot.py')
    info['gaussian_counts'] = {'rgb': len(points)}
    info['base_predictions_exactly_source'] = True
    info['both_offsets_zero_verified'] = True
    radius = np.linalg.norm(points, axis=1)
    info['origin_radius_p50_p90_p99_max'] = np.percentile(radius, [50, 90, 99, 100]).tolist()
    verification = {'status': 'parameters_passed', 'reference_view_verified': captured['reference_view_verified'],
                    'unmodified_replay_exact_all_fields': True, 'base_predictions_exactly_source': True,
                    'depth_and_xy_offsets_zero': True, 'other_raw_gaussian_channels_unchanged': True,
                    'sh_opacity_rotation_exactly_source': True, 'same_gaussian_count': True,
                    'scales_match_base_over_original_depth_ratio': True,
                    'native_unit_ray_center_max_abs_error': max_center_error,
                    'source_reprojection_max_error_pixels': max_pixel_error,
                    'predictions_and_camera_trajectory_byte_identical': True,
                    'gaussian_count': len(points), 'origin_radius_p50_p90_p99_max': info['origin_radius_p50_p90_p99_max']}
    record(output / 'verification.json', verification)
    info['status'] = 'rendering'
    record(output / 'run_config.json', info)
    del raw, model, normalized, baseline, captured, previous
    gc.collect(); torch.cuda.empty_cache()
    with np.load(output / 'render_cameras.npz') as cameras:
        ext = torch.from_numpy(cameras['world_to_camera']).cuda()
        intr = torch.from_numpy(cameras['intrinsics']).cuda()
    print('Verified zero offsets and unchanged source parameters; rendering exact source 89-frame path', flush=True)
    render_single(output, 'rgb', gs, ext, intr, height, width, source_info['render_frames'])
    verification.update(verify_video(output, height, width, len(source_info['render_frames'])))
    vertices = PlyData.read(output / 'gaussians_rgb_dc.ply')['vertex'].data
    np.testing.assert_array_equal(np.column_stack([vertices[k] for k in ('x', 'y', 'z')]), gs.means[0].cpu().numpy())
    np.testing.assert_array_equal(np.column_stack([vertices[k] for k in ('f_dc_0', 'f_dc_1', 'f_dc_2')]), gs.harmonics[0, :, :, 0].cpu().numpy())
    for p in source_paths:
        assert digest(p) == source_hashes[p.name], f'Source changed: {p}'
    comparisons(source, output, height, width)
    verification.update(status='passed', source_files_unchanged=True, ply_xyz_dc_exactly_saved_parameters=True)
    record(output / 'verification.json', verification)
    info['status'] = 'complete'
    record(output / 'run_config.json', info)
    (output / 'README.md').write_text(
        '# Ebike：官方 RGB-only GSDPT，关闭深度与 XY offset\n\n'
        f'来源：{source}\n\n'
        '仅输入原来的 12 张 RGB，参考策略 saddle_balanced，参考图 img_018.jpg。'
        '未加载适配器，没有训练或场景优化。\n\n'
        '在原生 GaussianAdapter 前，同时把 raw_gaussians 的前两个 XY offset 通道和最后一个深度 offset 通道清零。'
        '保留原生单位射线投影；中心使用未修改的基础深度与源像素中心射线计算。'
        '世界尺度随基础深度重算，学习到的尺度系数不变。完整二阶球谐、透明度、旋转与来源逐元素相同。'
        '基础深度与相机预测、筛选掩码及 89 帧渲染相机轨迹保持不变。\n\n'
        '这一版本仅关闭两个 offset，不额外改成普通点云的相机 Z 深度反投影，因此中心仍不保证等于 rgb_full.ply。\n\n'
        '- gaussians_rgb.pt：完整二阶球谐高斯参数。\n'
        '- gaussians_rgb_dc.ply：供查看器使用的 DC-only 高斯 PLY。\n'
        '- novel_views.mp4 / novel_views.gif / preview.jpg：新视角渲染。\n'
        '- comparison_novel_05.png：左为原生 offset，右为深度与 XY offset 都关闭。\n'
        '- comparison_rgb.png：全部 11 个中间视角的前后对照。\n'
        '- disabled_offsets.npz：清零前的原始 depth / XY offset。\n'
        '- predictions.npz / render_cameras.npz：与来源逐字节一致。\n'
        '- run_config.json / verification.json / render_script_snapshot.py / run.log：配置、核验与复现记录。\n\n'
        f'高斯数量：{len(points):,}。几何核验、参数重放、PLY 读取以及 89 帧视频解码均通过。'
        '关闭 offset 不会修正保留的颜色预测；渲染质量需结合对照图判断。\n', encoding='utf-8')
    print(json.dumps(verification, indent=2, ensure_ascii=False), flush=True)
    print('Complete: ' + str(output), flush=True)


if __name__ == '__main__':
    main()
