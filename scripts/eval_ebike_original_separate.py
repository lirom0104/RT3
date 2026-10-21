#!/usr/bin/env python3
"""Supplement the paired Ebike evaluation with original DA3 per modality."""

import argparse
import gc
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import pycolmap
import torch
import lpips

from eval_ebike_gsdpt import (
    MODALITIES, ROOT, TEST_NAMES, colmap_cameras, file_hash, homogeneous,
    InputProcessor, load_giant, record, render, save_images, split_gaussians,
    ssim_per_image, transform_cameras, align_poses_umeyama, gs_weights_hash,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation', type=Path, required=True)
    args = parser.parse_args()
    parent = json.loads((args.evaluation/'run_config.json').read_text())
    output = args.evaluation/'original_separate'
    output.mkdir(exist_ok=True)
    assert not (output/'metrics.json').exists()
    torch.set_num_threads(8)
    height, width = parent['input_hw']
    dataset = ROOT.parent/'data/SEAR'/parent['scene']
    reconstruction = pycolmap.Reconstruction(str(dataset/'colmap/sparse/0'))
    info = {
        'status': 'running', 'protocol': 'Supplementary original DA3 baseline: independent 12-image inference for each modality, no mixed inputs. Each modality aligned using its own 12 input cameras only.',
        'base_checkpoint': parent['base_checkpoint'], 'test_views': TEST_NAMES,
        'gaussian_filter': parent['gaussian_filter'], 'metric_protocol': parent['metric_protocol'],
        'methods': {},
    }
    record(output/'run_config.json', info)
    model = load_giant(parent['base_checkpoint'], include_gaussians=True).cuda().eval()
    assert gs_weights_hash(model) == parent['gs_weights_sha256']
    for modality in MODALITIES:
        views = [v for v in parent['input_views'] if v['modality']==modality]
        names = [Path(v['image']).stem for v in views]
        assert set(names).isdisjoint(TEST_NAMES)
        normalized, _, _ = InputProcessor()([v['image'] for v in views], process_res=518, sequential=True)
        if normalized.ndim == 5:
            normalized = normalized[0]
        input_exts, _ = colmap_cameras(reconstruction, names, height, width)
        test_exts, test_intrs = colmap_cameras(reconstruction, TEST_NAMES, height, width)
        torch.manual_seed(parent['seed'])
        started = time.monotonic()
        print(f'Running original standalone {modality}, {len(views)} inputs', flush=True)
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            raw = model(normalized[None].cuda(), infer_gs=True,
                        ref_view_strategy='saddle_balanced', use_ray_pose=False)
        torch.cuda.synchronize()
        predicted = raw.extrinsics[0].float().cpu().numpy()
        r, t, s = align_poses_umeyama(predicted, input_exts, ransac=False)
        aligned_inputs = transform_cameras(input_exts, r, t, s)
        aligned_tests = transform_cameras(test_exts, r, t, s)
        p, a = np.linalg.inv(homogeneous(predicted)), np.linalg.inv(aligned_inputs)
        relative = p[:, :3, :3] @ a[:, :3, :3].transpose(0, 2, 1)
        angles = np.degrees(np.arccos(np.clip((np.trace(relative, axis1=1, axis2=2)-1)/2, -1, 1)))
        gs = split_gaussians(raw.gaussians, raw.depth, 0, len(views))
        with torch.inference_mode():
            images = render(gs, aligned_tests, test_intrs, height, width)
        save_images(output/modality, images)
        np.savez_compressed(output/f'{modality}_rendered_float.npz', images=images)
        np.savez_compressed(output/f'{modality}_cameras.npz', predicted_input_extrinsics=predicted,
                            aligned_test_extrinsics=aligned_tests, test_intrinsics=test_intrs)
        info['methods'][modality] = {
            'input_views': views, 'input_sha256': hashlib.sha256(normalized.numpy().tobytes()).hexdigest(),
            'elapsed_seconds': time.monotonic()-started,
            'gaussian_count': gs.means.shape[1], 'scale': float(s),
            'rotation': r.tolist(), 'translation': t.tolist(),
            'input_center_rmse_predicted_units': float(np.sqrt(np.mean(np.sum((p[:, :3, 3]-a[:, :3, 3])**2, axis=-1)))),
            'mean_input_rotation_error_degrees': float(angles.mean()),
            'input_rotation_error_degrees': angles.tolist(),
        }
        print(f'{modality}: mean camera rotation error {angles.mean():.3f} degrees', flush=True)
        record(output/'run_config.json', info)
        del raw, gs, normalized
        gc.collect()
        torch.cuda.empty_cache()
    del model
    gc.collect()
    torch.cuda.empty_cache()
    perceptual = lpips.LPIPS(net='alex', version='0.1').cuda().eval()
    rows, summary = [], []
    with torch.inference_mode():
        for modality in MODALITIES:
            images = np.load(output/f'{modality}_rendered_float.npz')['images']
            refs = np.stack([np.asarray(Image.open(args.evaluation/'reference'/modality/f'{name}.png'))
                             for name in TEST_NAMES])/255.
            x = torch.from_numpy(images).permute(0, 3, 1, 2).cuda().float()
            y = torch.from_numpy(refs).permute(0, 3, 1, 2).cuda().float()
            values = {'psnr': -10*torch.log10((x-y).square().mean((1, 2, 3))),
                      'ssim': ssim_per_image(x, y),
                      'lpips': perceptual(x, y, normalize=True).flatten()}
            assert all(torch.isfinite(v).all() for v in values.values())
            for i, name in enumerate(TEST_NAMES):
                rows.append({'method': 'original_separate', 'modality': modality, 'view': name,
                             **{key: value[i].item() for key, value in values.items()}})
            summary.append({'method': 'original_separate', 'modality': modality, 'views': len(TEST_NAMES),
                            **{key: value.mean().item() for key, value in values.items()}})
    record(output/'metrics.json', {'summary': summary, 'per_view': rows})
    info['status'] = 'complete'
    info['script_sha256'] = file_hash(Path(__file__))
    record(output/'run_config.json', info)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
