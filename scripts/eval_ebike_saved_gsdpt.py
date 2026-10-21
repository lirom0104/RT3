#!/usr/bin/env python3
"""Evaluate a saved Ebike Gaussian reconstruction against the recorded held-out test."""

import argparse
import csv
import gc
import json
from pathlib import Path

from eval_ebike_gsdpt import (
    MODALITIES, TEST_NAMES, align_poses_umeyama, colmap_cameras, file_hash,
    homogeneous, lpips, np, pycolmap, render, save_images, ssim_per_image,
    torch, transform_cameras,
)
from PIL import Image, ImageDraw
from depth_anything_3.specs import Gaussians
from render_sear_gsdpt import record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--render-dir', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--metrics-only', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    source = json.loads((args.render_dir/'run_config.json').read_text())
    baseline = json.loads((args.baseline/'run_config.json').read_text())
    assert source['status'] == baseline['status'] == 'complete'
    assert source['input_sha256'] == baseline['input_sha256']
    assert source['input_hw'] == baseline['input_hw']
    assert baseline['test_views'] == TEST_NAMES
    height, width = source['input_hw']
    views = source['scene']['views']
    n = len(views)//2
    input_names = [Path(v['image']).stem for v in views[:n]]
    assert set(input_names).isdisjoint(TEST_NAMES)
    references = {}
    reference_files = {}
    for modality, directory, suffix in [('rgb', 'images', '.jpg'), ('thermal', 'thermal', '.png')]:
        # Verify that SEAR-deal and the previous SEAR evaluation have identical targets.
        vggt = Path(next(v['image'] for v in views if v['modality']==modality)).parent.parent
        paths = [vggt/directory/(name+suffix) for name in TEST_NAMES]
        references[modality] = np.stack([np.asarray(Image.open(p).convert('RGB')) for p in paths])/255.
        assert references[modality].shape == (4, height, width, 3)
        for i, previous in enumerate(baseline['reference_images'][modality]):
            assert file_hash(previous['path']) == previous['sha256']
            assert np.array_equal(np.asarray(Image.open(previous['path']).convert('RGB')),
                                  np.rint(references[modality][i]*255).astype(np.uint8))
        reference_files[modality] = [{'path': str(p), 'sha256': file_hash(p)} for p in paths]
        save_images(args.output/'reference'/modality, references[modality])
    dataset = Path(baseline['reference_images']['rgb'][0]['path']).parents[2]
    for name, digest in baseline['colmap_files'].items():
        assert file_hash(dataset/'colmap/sparse/0'/name) == digest
    info = {
        'status': 'running', 'source_render': str(args.render_dir.resolve()),
        'source_run_config_sha256': file_hash(args.render_dir/'run_config.json'),
        'scene': source['scene']['scene'], 'test_views': TEST_NAMES,
        'input_sha256': source['input_sha256'], 'test_input_overlap': [],
        'reference_images': reference_files, 'references_match_previous_evaluation': True,
        'baseline': str(args.baseline.resolve()), 'baseline_results_reused': True,
        'baseline_metrics_sha256': file_hash(args.baseline/'all_metrics.json'),
        'adapter': source['adapter'], 'adapter_sha256': source['adapter_sha256'],
        'rgb_detail_refinement': source['rgb_detail_refinement'],
        'metric_protocol': baseline['metric_protocol'],
        'test_cameras': baseline['test_cameras'], 'camera_alignment': baseline['camera_alignment'],
        'reference_format': baseline['reference_format'],
        'gs_depth_source': source['gs_depth_source'],
        'gaussian_filter': source['filter'],
    }
    if not args.metrics_only:
        if (args.output/'rendered_float.npz').exists():
            raise FileExistsError('Use --metrics-only to reuse held-out renders')
        reconstruction = pycolmap.Reconstruction(str(dataset/'colmap/sparse/0'))
        input_exts, _ = colmap_cameras(reconstruction, input_names, height, width)
        test_exts, test_intrs = colmap_cameras(reconstruction, TEST_NAMES, height, width)
        with np.load(args.render_dir/'predictions.npz') as arrays:
            predicted = arrays['extrinsics'][:n]
        r, t, s = align_poses_umeyama(predicted, input_exts, ransac=False)
        aligned_inputs = transform_cameras(input_exts, r, t, s)
        aligned_tests = transform_cameras(test_exts, r, t, s)
        predicted_pose = np.linalg.inv(homogeneous(predicted))
        aligned_pose = np.linalg.inv(aligned_inputs)
        center_errors = np.linalg.norm(predicted_pose[:, :3, 3]-aligned_pose[:, :3, 3], axis=-1)
        relative = predicted_pose[:, :3, :3] @ aligned_pose[:, :3, :3].transpose(0, 2, 1)
        angles = np.degrees(np.arccos(np.clip((np.trace(relative, axis1=1, axis2=2)-1)/2, -1, 1)))
        info['alignment'] = {'rotation': r.tolist(), 'translation': t.tolist(), 'scale': float(s),
            'input_center_rmse_predicted_units': float(np.sqrt(np.mean(center_errors**2))),
            'mean_input_rotation_error_degrees': float(angles.mean()),
            'input_rotation_error_degrees': angles.tolist()}
        np.savez_compressed(args.output/'cameras.npz', aligned_test_extrinsics=aligned_tests,
                            test_intrinsics=test_intrs, predicted_input_extrinsics=predicted)
        record(args.output/'run_config.json', info)
        predictions = {}
        for modality in MODALITIES:
            path = args.render_dir/f'gaussians_{modality}.pt'
            print(f'Loading and rendering saved {modality} Gaussians', flush=True)
            data = torch.load(path, map_location='cuda', weights_only=True)
            gs = Gaussians(**data)
            assert gs.means.shape[1] == source['gaussian_counts'][modality]
            with torch.inference_mode():
                predictions[modality] = render(gs, aligned_tests, test_intrs, height, width)
            save_images(args.output/'rgb_detail_refine'/modality, predictions[modality])
            del gs, data
            gc.collect()
            torch.cuda.empty_cache()
        np.savez_compressed(args.output/'rendered_float.npz', **predictions)
    else:
        info = json.loads((args.output/'run_config.json').read_text())
    print('Computing PSNR, SSIM, LPIPS', flush=True)
    perceptual = lpips.LPIPS(net='alex', version='0.1').cuda().eval()
    previous = json.loads((args.baseline/'all_metrics.json').read_text())
    rows, summary = list(previous['per_view']), list(previous['summary'])
    with np.load(args.output/'rendered_float.npz') as saved, torch.inference_mode():
        for modality in MODALITIES:
            x = torch.from_numpy(saved[modality]).permute(0, 3, 1, 2).cuda().float()
            y = torch.from_numpy(references[modality]).permute(0, 3, 1, 2).cuda().float()
            psnr = -10*torch.log10((x-y).square().mean(dim=(1, 2, 3)))
            ssim = ssim_per_image(x, y)
            distance = perceptual(x, y, normalize=True).flatten()
            assert torch.isfinite(torch.stack([psnr, ssim, distance])).all()
            for i, name in enumerate(TEST_NAMES):
                rows.append({'method': 'rgb_detail_refine', 'modality': modality, 'view': name,
                             'psnr': psnr[i].item(), 'ssim': ssim[i].item(), 'lpips': distance[i].item()})
            summary.append({'method': 'rgb_detail_refine', 'modality': modality, 'views': 4,
                            'psnr': psnr.mean().item(), 'ssim': ssim.mean().item(), 'lpips': distance.mean().item()})
    record(args.output/'metrics.json', {'summary': summary, 'per_view': rows})
    with (args.output/'per_view_metrics.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    methods = ['reference', 'original_separate', 'trained', 'rgb_detail_refine']
    labels = ['GT', 'Original DA3: separate inputs', 'Previous trained: joint inputs', 'RGB detail refine: joint inputs']
    for modality in MODALITIES:
        sheet = Image.new('RGB', (width*len(methods), (height+42)*len(TEST_NAMES)), 'black')
        draw = ImageDraw.Draw(sheet)
        for i, name in enumerate(TEST_NAMES):
            for j, (method, label) in enumerate(zip(methods, labels)):
                directory = args.output if method in ('reference', 'rgb_detail_refine') else args.baseline
                picture = Image.open(directory/method/modality/f'{name}.png')
                x0, y0 = j*width, i*(height+42)
                sheet.paste(picture, (x0, y0+42))
                draw.text((x0+5, y0+3), f'{modality.upper()} {name} | {label}', fill='white')
                if method != 'reference':
                    row = next(row for row in rows if row['method']==method and row['modality']==modality and row['view']==name)
                    draw.text((x0+5, y0+20), f"PSNR {row['psnr']:.2f} SSIM {row['ssim']:.3f} LPIPS {row['lpips']:.3f}", fill='white')
        sheet.save(args.output/f'comparison_{modality}.jpg', quality=94)
    info['status'] = 'complete'
    info['script_sha256'] = file_hash(Path(__file__))
    record(args.output/'run_config.json', info)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
