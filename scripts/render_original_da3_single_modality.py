#!/usr/bin/env python3
"""Run original DA3 on one modality and render its own predicted camera path."""

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

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw
import torch

import depth_anything_3.model.dinov2.vision_transformer as vit
from depth_anything_3.model.utils.gs_renderer import render_3dgs
from depth_anything_3.specs import Gaussians
from depth_anything_3.thermal.adapter import load_giant
from depth_anything_3.utils.gsply_helpers import save_gaussian_ply
from depth_anything_3.utils.io.input_processor import InputProcessor
from infer_sear_comparison import file_hash
from render_sear_gsdpt import record, split_gaussians, trajectory


def render_single(output, modality, gs, ext, intr, height, width, locations):
    normalized = intr.clone()
    normalized[:, 0] /= width
    normalized[:, 1] /= height
    frames, stills = [], []
    with imageio.get_writer(str(output / 'novel_views.mp4'), fps=24,
                            codec='libx264', quality=8, macro_block_size=2) as writer:
        for index, location in enumerate(locations):
            with torch.inference_mode():
                color, _ = render_3dgs(ext[index:index + 1], normalized[index:index + 1],
                                      (height, width), gs, num_view=1, color_mode='RGB+ED')
            assert bool(torch.isfinite(color).all()), f'Nonfinite render at frame {index}'
            pixels = (color[0].clamp(0, 1) * 255).byte().permute(1, 2, 0).cpu().numpy()
            picture = Image.fromarray(pixels)
            frame = Image.new('RGB', (width, height + 24), 'black')
            frame.paste(picture, (0, 24))
            ImageDraw.Draw(frame).text((8, 6),
                f'{modality.upper()} ONLY | Original DA3 | frame {index:03d}', fill='white')
            writer.append_data(np.asarray(frame))
            frames.append(frame)
            if location['fraction'] == .5:
                picture.save(output / f'{modality}_novel_{location["segment"]:02d}.png')
                stills.append(frame)
            if index % 10 == 0:
                print(f'Rendered {index + 1}/{len(locations)} {modality}-only frames', flush=True)
    frames[0].save(output / 'novel_views.gif', save_all=True,
                   append_images=frames[1:], duration=1000 // 24, loop=0)
    selected = np.linspace(0, len(stills) - 1, min(5, len(stills)), dtype=int)
    sheet = Image.new('RGB', (width, (height + 24) * len(selected)), 'black')
    for row, index in enumerate(selected):
        sheet.paste(stills[index], (0, row * (height + 24)))
    sheet.save(output / 'preview.jpg', quality=94)


def verify_video(output, height, width, expected_frames):
    with imageio.get_reader(str(output / 'novel_views.mp4')) as reader:
        metadata = reader.get_meta_data()
        assert abs(metadata['fps'] - 24) < 1e-6
        assert tuple(metadata['size']) == (width, height + 24)
        decoded = 0
        for frame in reader:
            assert frame.shape == (height + 24, width, 3)
            assert frame[24:].any(), f'Blank decoded frame {decoded}'
            decoded += 1
    assert decoded == expected_frames
    with Image.open(output / 'novel_views.gif') as animation:
        gif_frames = animation.n_frames
    assert gif_frames == expected_frames
    return {'mp4_decoded_frames': decoded, 'gif_frames': gif_frames,
            'fps': metadata['fps'], 'video_resolution': list(metadata['size']),
            'duration_seconds': decoded / metadata['fps']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True,
                        help='Completed original DA3 first-reference joint reconstruction')
    parser.add_argument('--modality', choices=('rgb', 'thermal'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source_hash = file_hash(args.source / 'run_config.json')
    source = json.loads((args.source / 'run_config.json').read_text())
    assert source['status'] == 'complete'
    assert source['adapter'] is None and source['thermal_mask'] is None
    assert source['ref_view_strategy'] == 'first'
    views = [view for view in source['scene']['views'] if view['modality'] == args.modality]
    assert len(views) == 12
    assert [view['pair_index'] for view in views] == source['scene']['selected_pair_indices']
    assert len({view['image'] for view in views}) == 12
    for view in views:
        assert Path(view['image']).is_file()
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(Path(__file__), args.output / 'inference_render_script_snapshot.py')
    torch.set_num_threads(8)
    normalized, _, _ = InputProcessor()([view['image'] for view in views],
                                       process_res=source['process_res'], sequential=True)
    if normalized.ndim == 5:
        normalized = normalized[0]
    assert normalized.shape[0] == 12 and normalized.shape[1] == 3
    assert list(normalized.shape[-2:]) == source['input_hw']
    scene = {**source['scene'], 'views': views}
    info = {
        'status': 'running', 'model': source['model'],
        'base_checkpoint': source['base_checkpoint'],
        'base_checkpoint_file': str(Path(source['base_checkpoint']) / 'model.safetensors'),
        'adapter': None, 'thermal_mask': None, 'rgb_detail_refinement': None,
        'protocol': f'One fresh single-modality inference on exactly 12 {args.modality} images',
        'modality': args.modality, 'input_view_count': 12,
        'input_modality_counts': {'rgb': 12 if args.modality == 'rgb' else 0,
                                 'thermal': 12 if args.modality == 'thermal' else 0},
        'model_input_shape': [1, *normalized.shape],
        'scene': scene, 'input_hw': source['input_hw'], 'process_res': source['process_res'],
        'seed': source['seed'], 'ref_view_strategy': 'first',
        'infer_gs': True, 'use_ray_pose': False, 'source_camera_conditioning': False,
        'training': 'Original pretrained DA3 and GSDPT; no adapter, depth refinement or scene optimization',
        'filter': source['filter'],
        'input_sha256': hashlib.sha256(normalized.numpy().tobytes()).hexdigest(),
        'input_files_sha256': {view['image']: file_hash(view['image']) for view in views},
        'input_selection_source': str(args.source.resolve()),
        'input_selection_source_config_sha256': source_hash,
        'preprocessing': 'Native InputProcessor RGB conversion and ImageNet normalization',
        'script_sha256': file_hash(Path(__file__)),
        'trajectory': f'Native DA3 focus-point interpolation of own predicted {args.modality} cameras',
        'camera_alignment': None, 'reference_camera_parameters': None,
        'steps_per_input_interval': 8,
    }
    record(args.output / 'run_config.json', info)
    print(f'Loading original DA3 + pretrained GSDPT for exactly 12 {args.modality} images', flush=True)
    print('Model input shape: ' + str(info['model_input_shape']), flush=True)
    model = load_giant(source['base_checkpoint'], include_gaussians=True).cuda().eval()
    assert not hasattr(model.backbone.pretrained, 'thermal_camera_token')
    assert not any('lora_' in name for name, _ in model.named_parameters())
    original_selector = vit.select_reference_view
    forward_inputs = []

    def capture_reference(x, strategy):
        selected = original_selector(x, strategy=strategy)
        index = int(selected[0].item())
        assert strategy == 'first' and index == 0
        info['reference_view'] = {'index_zero_based': index, **views[index]}
        print('Actual reference: ' + json.dumps(info['reference_view']), flush=True)
        return selected

    def capture_input(module, arguments):
        shape = list(arguments[0].shape)
        assert shape == info['model_input_shape']
        forward_inputs.append(shape)

    vit.select_reference_view = capture_reference
    input_hook = model.register_forward_pre_hook(capture_input)
    torch.manual_seed(source['seed'])
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    try:
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            raw = model(normalized[None].cuda(), thermal_mask=None, infer_gs=True,
                        ref_view_strategy='first', use_ray_pose=False)
    finally:
        vit.select_reference_view = original_selector
        input_hook.remove()
    torch.cuda.synchronize()
    assert forward_inputs == [info['model_input_shape']]
    info['forward_seconds'] = time.monotonic() - started
    info['peak_memory_gib'] = torch.cuda.max_memory_allocated() / 1024 ** 3
    arrays = {key: raw[key][0].float().cpu().numpy()
              for key in ('depth', 'depth_conf', 'extrinsics', 'intrinsics')}
    assert all(value.shape[0] == 12 and np.isfinite(value).all() for value in arrays.values())
    np.savez_compressed(args.output / 'predictions.npz', **arrays)
    gs = split_gaussians(raw.gaussians, raw.depth, 0, 12)
    info['gaussian_counts'] = {args.modality: gs.means.shape[1]}
    gaussian_file = args.output / f'gaussians_{args.modality}.pt'
    torch.save({field.name: getattr(gs, field.name).cpu() for field in fields(Gaussians)}, gaussian_file)
    info['gaussian_files_sha256'] = {args.modality: file_hash(gaussian_file)}
    count = gs.means.shape[1]
    save_gaussian_ply(gs, str(args.output / f'gaussians_{args.modality}_dc.ply'),
                      torch.ones((1, 1, count, 1), device='cuda'),
                      save_sh_dc_only=True, prune_border_gs=False, prune_by_depth_percent=1.)
    ext, intr, locations = trajectory(raw.extrinsics.float(), raw.intrinsics.float(), 8)
    assert len(locations) == 89
    assert bool(torch.isfinite(ext).all() and torch.isfinite(intr).all())
    ext_array, intr_array = ext.cpu().numpy(), intr.cpu().numpy()
    endpoints = list(range(0, len(locations), 8))
    np.testing.assert_allclose(ext_array[endpoints, :3], arrays['extrinsics'], rtol=1e-4, atol=1e-4)
    np.testing.assert_array_equal(intr_array[endpoints], arrays['intrinsics'])
    info['input_camera_endpoint_max_abs_error'] = float(np.max(np.abs(ext_array[endpoints, :3] - arrays['extrinsics'])))
    info['input_intrinsic_endpoints_exact'] = True
    info['render_frames'] = locations
    info['render_frame_count'] = len(locations)
    np.savez_compressed(args.output / 'render_cameras.npz', world_to_camera=ext_array,
                        intrinsics=intr_array)
    info['status'] = 'rendering'
    record(args.output / 'run_config.json', info)
    print(f'Fresh {args.modality}-only inference complete: {info["forward_seconds"]:.2f}s; {count} Gaussians', flush=True)
    del raw, model, normalized
    gc.collect()
    torch.cuda.empty_cache()
    height, width = info['input_hw']
    render_single(args.output, args.modality, gs, ext, intr, height, width, locations)
    verification = {
        'status': 'passed', 'single_modality_only': args.modality,
        'model_forward_calls': len(forward_inputs), 'model_input_shape': forward_inputs[0],
        'input_views': 12, 'original_weights_without_adapter': True,
        'reference_view': info['reference_view'], 'prediction_arrays_finite': True,
        'gaussian_count': count, 'gaussian_parameters_finite': True,
        'input_camera_endpoint_max_abs_error': info['input_camera_endpoint_max_abs_error'],
        'input_intrinsic_endpoints_exact': True,
        **verify_video(args.output, height, width, len(locations)),
    }
    stills = sorted(args.output.glob(f'{args.modality}_novel_*.png'))
    assert len(stills) == 11
    for path in stills:
        with Image.open(path) as picture:
            assert picture.size == (width, height)
    assert file_hash(args.source / 'run_config.json') == source_hash
    for path, digest in info['input_files_sha256'].items():
        assert file_hash(path) == digest
    verification['midpoint_pngs'] = len(stills)
    verification['input_files_and_source_config_unchanged'] = True
    record(args.output / 'verification.json', verification)
    info['status'] = 'complete'
    record(args.output / 'run_config.json', info)
    modality_label = 'RGB' if args.modality == 'rgb' else '热红外'
    (args.output / 'README.md').write_text(
        f'Ebike：仅输入 12 张{modality_label}图的新视角渲染\n\n'
        f'使用原始 DA3-GIANT-1.1 和其预训练 GSDPT：{source["base_checkpoint"]}。\n'
        '每次模型输入严格为 12 张同一模态图像；未加载热红外适配器。\n'
        '采样机位和原始 first_rgb 联合推理相同，参考策略为 first，第一张输入图为参考。\n'
        '热红外输入采用原生 InputProcessor 转为三通道后归一化。\n'
        '所有内外参由本次单模态推理自行预测；轨迹由这 12 个预测机位插值生成，'
        '每个区间 8 步，共 89 帧。不同实验使用各自预测的相机轨迹。\n'
        '视频为单栏画面，518×416（含标题），24 fps，约 3.71 秒。\n\n'
        'novel_views.mp4 / novel_views.gif：视频与动画。\n'
        'preview.jpg：5 个中间机位预览。\n'
        f'{args.modality}_novel_00.png～10.png：11 个中间机位图像。\n'
        f'gaussians_{args.modality}.pt：完整 degree-2 SH 高斯参数。\n'
        f'gaussians_{args.modality}_dc.ply：供查看器使用的 DC-only 高斯。\n'
        'predictions.npz / render_cameras.npz：预测结果与相机轨迹。\n'
        'run_config.json / verification.json / inference_render_script_snapshot.py：配置、核验与脚本。\n\n'
        '从 DA3base 根目录复现到新目录：\n\n'
        '```bash\n'
        'env CUDA_VISIBLE_DEVICES=1 CUDA_HOME=/home/lf/miniconda3/envs/da3 '
        'LD_LIBRARY_PATH=/home/lf/miniconda3/envs/da3/lib OMP_NUM_THREADS=8 '
        '/home/lf/miniconda3/envs/da3/bin/python -u scripts/render_original_da3_single_modality.py '
        f'--source {args.source} --modality {args.modality} --output {args.output}_reproduce\n'
        '```\n', encoding='utf-8')
    print(json.dumps(verification, indent=2), flush=True)
    print('Complete: ' + str((args.output / 'novel_views.mp4').resolve()), flush=True)


if __name__ == '__main__':
    main()
