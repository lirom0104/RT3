#!/usr/bin/env python3
"""Evaluate original/trained Giant + the original GSDPT on unseen Ebike views.

Reuses the recorded 12 paired inputs. COLMAP test cameras are transformed into
each prediction's coordinates using a Sim(3) estimated from RGB input cameras
only. Reference test images never enter the network or the alignment fit.
"""

import argparse
import csv
from dataclasses import fields
import gc
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]

import lpips
import numpy as np
from PIL import Image, ImageDraw
import pycolmap
import torch
import torch.nn.functional as F
from gsplat import rasterization

from depth_anything_3.thermal.adapter import load_adapter, load_giant
from depth_anything_3.utils.io.input_processor import InputProcessor
from depth_anything_3.utils.pose_align import align_poses_umeyama
from infer_sear_comparison import file_hash
from render_sear_gsdpt import record, split_gaussians

TEST_NAMES = ['img_017', 'img_025', 'img_033', 'img_041']
MODALITIES = ('rgb', 'thermal')
METHODS = ('original', 'trained')


def homogeneous(ext):
    if ext.shape[-2:] == (4, 4):
        return ext
    result = np.broadcast_to(np.eye(4), (*ext.shape[:-2], 4, 4)).copy()
    result[..., :3, :] = ext
    return result


def transform_cameras(ext, rotation, translation, scale):
    poses = np.linalg.inv(homogeneous(ext))
    poses[:, :3, :3] = rotation @ poses[:, :3, :3]
    poses[:, :3, 3] = (rotation @ (scale * poses[:, :3, 3]).T).T + translation
    return np.linalg.inv(poses)


def colmap_cameras(reconstruction, names, height, width):
    by_stem = {Path(image.name).stem: image for image in reconstruction.images.values()}
    exts, intrs = [], []
    for name in names:
        image = by_stem[name]
        camera = reconstruction.cameras[image.camera_id]
        if camera.model.name != 'PINHOLE':
            raise ValueError('This evaluation expects the Ebike PINHOLE calibration')
        fx, fy, cx, cy = camera.params
        intr = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.]])
        intr[0] *= width / camera.width
        intr[1] *= height / camera.height
        ext = np.eye(4)
        ext[:3] = image.cam_from_world().matrix()
        exts.append(ext)
        intrs.append(intr)
    return np.stack(exts), np.stack(intrs)


def ssim_per_image(x, y):
    """Standard 3DGS SSIM: 11x11 Gaussian, sigma=1.5, zero padding, RGB mean."""
    coordinate = torch.arange(11, device=x.device, dtype=x.dtype) - 5
    window = torch.exp(-coordinate.square() / (2 * 1.5**2))
    window /= window.sum()
    kernel = (window[:, None] * window[None, :])[None, None].repeat(x.shape[1], 1, 1, 1)
    def blur(value):
        return F.conv2d(value, kernel, padding=5, groups=x.shape[1])
    mx, my = blur(x), blur(y)
    vx, vy, covariance = blur(x*x)-mx*mx, blur(y*y)-my*my, blur(x*y)-mx*my
    score = ((2*mx*my+.01**2)*(2*covariance+.03**2) /
             ((mx*mx+my*my+.01**2)*(vx+vy+.03**2)))
    return score.mean(dim=(1, 2, 3))


def render(gs, exts, intrs, height, width):
    """Use native gsplat settings with exact pixel intrinsics, degree-2 SH."""
    images = []
    for ext, intr in zip(exts, intrs):
        colors, _, _ = rasterization(
            means=gs.means[0], quats=gs.rotations[0], scales=gs.scales[0],
            opacities=gs.opacities[0],
            colors=gs.harmonics[0].permute(0, 2, 1).contiguous(),
            viewmats=torch.as_tensor(ext[None], dtype=torch.float32, device='cuda'),
            Ks=torch.as_tensor(intr[None], dtype=torch.float32, device='cuda'),
            backgrounds=torch.zeros((1, 3), device='cuda'),
            render_mode='RGB', width=width, height=height, packed=False,
            sh_degree=int(gs.harmonics.shape[-1]**.5)-1,
        )
        images.append(colors[0].clamp(0, 1).cpu().numpy())
    return np.stack(images)


def gs_weights_hash(model):
    digest = hashlib.sha256()
    for branch in (model.gs_head, model.gs_adapter):
        for name, value in sorted(branch.state_dict().items()):
            digest.update(name.encode())
            digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def save_images(folder, images):
    folder.mkdir(parents=True, exist_ok=True)
    for name, picture in zip(TEST_NAMES, images):
        pixels = np.rint(picture*255).clip(0, 255).astype(np.uint8)
        Image.fromarray(pixels).save(folder / f'{name}.png')


def previews(output, height, width, rows):
    for modality in MODALITIES:
        sheet = Image.new('RGB', (width*3, (height+40)*len(TEST_NAMES)), 'black')
        draw = ImageDraw.Draw(sheet)
        for i, name in enumerate(TEST_NAMES):
            for j, method in enumerate(('reference', *METHODS)):
                picture = Image.open(output/method/modality/f'{name}.png')
                x, y = j*width, i*(height+40)
                sheet.paste(picture, (x, y+40))
                label = f'{modality.upper()} {name} | {method}'
                if method != 'reference':
                    row = next(r for r in rows if r['method']==method and
                               r['modality']==modality and r['view']==name)
                    label += f" | {row['psnr']:.2f}dB SSIM {row['ssim']:.3f} LPIPS {row['lpips']:.3f}"
                draw.text((x+5, y+10), label, fill='white')
        sheet.save(output/f'preview_{modality}.jpg', quality=94)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--record-dir', type=Path,
                        default=ROOT/'artifacts/comparison_all_scenes_20261010')
    parser.add_argument('--metrics-only', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    config = json.loads((args.record_dir/'run_config.json').read_text())
    scene = next(s for s in config['scenes'] if s['scene']=='ThermalGaussian/Ebike')
    dataset = Path(config['data_root'])/scene['scene']
    n = len(scene['selected_pair_indices'])
    assert len(scene['views']) == 2*n
    assert [v['modality'] for v in scene['views']] == ['rgb']*n+['thermal']*n
    input_names = [Path(v['image']).stem for v in scene['views'][:n]]
    assert set(input_names).isdisjoint(TEST_NAMES), 'Test image is among reconstruction inputs'
    for modality in MODALITIES:
        assert set(TEST_NAMES).issubset(p.stem for p in (dataset/modality/'test').glob('*.jpg'))
    assert file_hash(config['adapter']) == config['adapter_sha256']
    normalized, _, _ = InputProcessor()([v['image'] for v in scene['views']],
                                        process_res=config['process_res'], sequential=True)
    if normalized.ndim == 5:
        normalized = normalized[0]
    input_hash = hashlib.sha256(normalized.numpy().tobytes()).hexdigest()
    previous = json.loads((args.record_dir/'progress.json').read_text())
    assert input_hash == previous['completed']['new/'+scene['scene']]['input_sha256']
    height, width = normalized.shape[-2:]
    references, reference_files = {}, {}
    for modality, directory, suffix in [('rgb', 'images', '.jpg'), ('thermal', 'thermal', '.png')]:
        paths = [dataset/'VGGT'/directory/(name+suffix) for name in TEST_NAMES]
        references[modality] = np.stack([np.asarray(Image.open(p).convert('RGB')) for p in paths])/255.
        assert references[modality].shape == (len(TEST_NAMES), height, width, 3)
        save_images(args.output/'reference'/modality, references[modality])
        reference_files[modality] = [{'path': str(p), 'sha256': file_hash(p)} for p in paths]
    info = {
        'status': 'running', 'scene': scene['scene'], 'test_views': TEST_NAMES,
        'input_views': scene['views'], 'input_sha256': input_hash,
        'input_matches_original_comparison': True, 'test_input_overlap': [],
        'base_checkpoint': config['base_checkpoint'], 'adapter': config['adapter'],
        'adapter_sha256': config['adapter_sha256'], 'seed': config['seed'],
        'input_hw': [height, width], 'reference_images': reference_files,
        'protocol': 'Both models jointly infer the same 12 RGB + 12 thermal inputs; split Gaussians by modality. No scene optimization.',
        'gs_head': 'Original pretrained Giant GSDPT and gs_adapter; unchanged in both methods',
        'gaussian_filter': 'Per-view 8/256 border trimming and nearest 99% depth, identical rule for both methods',
        'reference_format': 'Processed VGGT images at 518x392; RGB color, thermal grayscale replicated into 3 channels. Raw thermal/test JPEGs are pseudocolor and are not compared.',
        'test_cameras': 'COLMAP PINHOLE cameras; intrinsics scaled from 640x480 to 518x392. Registered RGB/thermal use the same cameras.',
        'camera_alignment': 'Independent Umeyama Sim(3) per model, fit only to the 12 RGB INPUT camera centers. No test image/camera fitting, refinement, or test-time optimization.',
        'metric_protocol': {
            'range': '[0,1], clamped float32 renders before PNG quantization',
            'psnr': '-10 log10(mean RGB squared error), per-image dB averaged across four views',
            'ssim': '3DGS convention: 11x11 Gaussian sigma 1.5, C1=.01^2, C2=.03^2, population covariance, zero padding, all pixels/channels averaged',
            'lpips': 'AlexNet LPIPS v0.1, normalize=True on RGB [0,1], thermal reference has identical channels',
            'mask_crop_color_alignment': 'None: full image, no foreground mask, no crop, no exposure or color fitting',
        },
        'colmap_files': {p.name: file_hash(p) for p in (dataset/'colmap/sparse/0').glob('*.bin')},
    }
    if not args.metrics_only:
        if (args.output/'original'/'rendered_float.npz').exists():
            raise FileExistsError('Use --metrics-only for an existing evaluation')
        record(args.output/'run_config.json', info)
        reconstruction = pycolmap.Reconstruction(str(dataset/'colmap/sparse/0'))
        input_exts, _ = colmap_cameras(reconstruction, input_names, height, width)
        test_exts, test_intrs = colmap_cameras(reconstruction, TEST_NAMES, height, width)
        np.savez_compressed(args.output/'colmap_cameras.npz', input_extrinsics=input_exts,
                            test_extrinsics=test_exts, test_intrinsics=test_intrs)
        print('Loading original Giant, including pretrained GSDPT', flush=True)
        model = load_giant(config['base_checkpoint'], include_gaussians=True).cuda().eval()
        original_gs_hash = gs_weights_hash(model)
        info['gs_weights_sha256'] = original_gs_hash
        info['methods'] = {}
        for method in METHODS:
            if method == 'trained':
                load_adapter(model, config['adapter'])
                model.eval()
                assert gs_weights_hash(model) == original_gs_hash, 'Adapter changed GS branch weights'
            mask = (torch.tensor([[v['modality']=='thermal' for v in scene['views']]],
                                 device='cuda') if method=='trained' else None)
            torch.manual_seed(config['seed'])
            torch.cuda.reset_peak_memory_stats()
            started = time.monotonic()
            print(f'Running {method}: {list(normalized.shape)}; infer_gs=True', flush=True)
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                raw = model(normalized[None].cuda(), thermal_mask=mask, infer_gs=True,
                            ref_view_strategy='saddle_balanced', use_ray_pose=False)
            torch.cuda.synchronize()
            method_info = {'forward_seconds': time.monotonic()-started,
                           'peak_memory_gib': torch.cuda.max_memory_allocated()/1024**3}
            folder = args.output/method
            folder.mkdir(exist_ok=True)
            predicted = raw.extrinsics[0, :n].float().cpu().numpy()
            r, t, s = align_poses_umeyama(predicted, input_exts, ransac=False)
            aligned_inputs = transform_cameras(input_exts, r, t, s)
            aligned_tests = transform_cameras(test_exts, r, t, s)
            pred_poses, aligned_poses = np.linalg.inv(homogeneous(predicted)), np.linalg.inv(aligned_inputs)
            center_error = np.linalg.norm(pred_poses[:, :3, 3]-aligned_poses[:, :3, 3], axis=-1)
            relative_r = pred_poses[:, :3, :3] @ aligned_poses[:, :3, :3].transpose(0, 2, 1)
            angles = np.degrees(np.arccos(np.clip((np.trace(relative_r, axis1=1, axis2=2)-1)/2, -1, 1)))
            method_info['alignment'] = {
                'rotation': r.tolist(), 'translation': t.tolist(), 'scale': float(s),
                'input_center_rmse_predicted_units': float(np.sqrt(np.mean(center_error**2))),
                'input_rotation_error_degrees': angles.tolist(),
                'mean_input_rotation_error_degrees': float(angles.mean()),
            }
            np.savez_compressed(folder/'cameras.npz', predicted_input_extrinsics=predicted,
                                aligned_test_extrinsics=aligned_tests, test_intrinsics=test_intrs)
            predictions = {}
            method_info['gaussian_counts'] = {}
            for modality, start, stop in [('rgb', 0, n), ('thermal', n, 2*n)]:
                gs = split_gaussians(raw.gaussians, raw.depth, start, stop)
                method_info['gaussian_counts'][modality] = gs.means.shape[1]
                print(f'Rendering {method}/{modality}: {gs.means.shape[1]} Gaussians', flush=True)
                with torch.inference_mode():
                    predictions[modality] = render(gs, aligned_tests, test_intrs, height, width)
                save_images(folder/modality, predictions[modality])
                del gs
            np.savez_compressed(folder/'rendered_float.npz', **predictions)
            info['methods'][method] = method_info
            record(args.output/'run_config.json', info)
            print(f'{method}: mean camera rotation error {angles.mean():.3f} degrees', flush=True)
            del raw, predictions
            gc.collect()
            torch.cuda.empty_cache()
        del model, normalized
        gc.collect()
        torch.cuda.empty_cache()
    else:
        info = json.loads((args.output/'run_config.json').read_text())

    print('Computing PSNR, SSIM, LPIPS (AlexNet v0.1)', flush=True)
    perceptual = lpips.LPIPS(net='alex', version='0.1').cuda().eval()
    rows, summary = [], []
    with torch.inference_mode():
        for method in METHODS:
            with np.load(args.output/method/'rendered_float.npz') as saved:
                for modality in MODALITIES:
                    x = torch.from_numpy(saved[modality]).permute(0, 3, 1, 2).cuda().float()
                    y = torch.from_numpy(references[modality]).permute(0, 3, 1, 2).cuda().float()
                    psnr = -10*torch.log10((x-y).square().mean(dim=(1, 2, 3)))
                    ssim = ssim_per_image(x, y)
                    distance = perceptual(x, y, normalize=True).flatten()
                    assert torch.isfinite(torch.stack([psnr, ssim, distance])).all()
                    for i, name in enumerate(TEST_NAMES):
                        rows.append({'method': method, 'modality': modality, 'view': name,
                                     'psnr': psnr[i].item(), 'ssim': ssim[i].item(),
                                     'lpips': distance[i].item()})
                    summary.append({'method': method, 'modality': modality, 'views': len(TEST_NAMES),
                                    'psnr': psnr.mean().item(), 'ssim': ssim.mean().item(),
                                    'lpips': distance.mean().item()})
    with (args.output/'per_view_metrics.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    record(args.output/'metrics.json', {'summary': summary, 'per_view': rows})
    previews(args.output, height, width, rows)
    info['status'] = 'complete'
    info['script_sha256'] = file_hash(Path(__file__))
    record(args.output/'run_config.json', info)
    print(json.dumps(summary, indent=2), flush=True)
    print('Complete: '+str(args.output.resolve()), flush=True)


if __name__ == '__main__':
    main()
