#!/usr/bin/env python3
"""Recompute matched predictions and apply the Ebike repair to every scene.

Keep official/new output filenames identical to the original comparison. Save
caches, provenance, pixel selections, resumable progress and audits separately.
"""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime
import gc
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import runpy
import shutil
import sys
import time
import traceback
from types import SimpleNamespace
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
import cv2
import matplotlib
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree
import torch
from infer_sear_comparison import (atomic_json, depth_to_world, file_hash, fingerprints,
                                   infer_scene, verify_completed)
from refine_multiview_detail import base_masks
from depth_anything_3.thermal.adapter import load_giant, load_adapter
from depth_anything_3.utils.logger import logger


def read_ply(path):
    with Path(path).open('rb') as stream:
        assert stream.readline() == b'ply\n'
        assert stream.readline() == b'format binary_little_endian 1.0\n'
        count = int(stream.readline().split()[-1])
        while stream.readline() != b'end_header\n':
            if stream.tell() > 4096:
                raise ValueError('Invalid PLY header')
        points = np.fromfile(stream, dtype=[('xyz', '<f4', (3,)), ('rgb', 'u1', (3,))], count=count)
        assert len(points) == count and count > 0 and np.isfinite(points['xyz']).all()
        assert stream.read() == b''
    return points['xyz'], points['rgb']


def verify_png_depth(folder, depth, scene, color_range):
    low, high = color_range
    for i, view in enumerate(scene['views']):
        filename = f"{view['pair_index']:05d}_{Path(view['image']).stem}.png"
        expected = np.rint(matplotlib.colormaps['turbo'](
            np.clip((depth[i] - low) / (high - low), 0, 1))[..., :3] * 255).astype(np.uint8)
        with Image.open(folder / 'depth' / view['modality'] / filename) as picture:
            assert picture.mode == 'RGB' and np.array_equal(expected, np.asarray(picture)), filename
    with Image.open(folder / 'depth.png') as picture:
        picture.load()


def rebuild_clouds(arrays, images, scene, seed, max_points, selection=None):
    clouds = {}
    valid = base_masks(arrays) if selection is None else arrays['export_valid']
    for modality in ('rgb', 'thermal'):
        xyz, colors, per_view = [], [], []
        for i, view in enumerate(scene['views']):
            if view['modality'] != modality:
                continue
            if selection is None:
                ids = np.flatnonzero(valid[i])
                ids = np.random.default_rng(seed + i).choice(ids, min(len(ids), max_points), replace=False)
            else:
                ids = selection[modality + '_pixels'][selection[modality + '_views'] == i]
                assert valid[i].ravel()[ids].all()
            points = depth_to_world(arrays['depth'][i], arrays['extrinsics'][i], arrays['intrinsics'][i])
            xyz.append(points.reshape(-1, 3)[ids].astype(np.float32))
            colors.append(np.rint(images[i].reshape(-1, 3)[ids] * 255).astype(np.uint8))
            per_view.append(len(ids))
        clouds[modality] = (np.concatenate(xyz), np.concatenate(colors), per_view)
    return clouds


def quality_metrics(clouds, extrinsics):
    rgb_indices = np.arange(len(clouds['rgb'][2]))
    e = extrinsics[rgb_indices]
    centers = np.array([-ee[:3, 3] @ ee[:3, :3] for ee in e])
    radius = float(np.sqrt(np.square(centers - centers.mean(0)).sum(1).mean()))
    distances = np.linalg.norm(centers[:, None] - centers[None, :], axis=-1)
    np.fill_diagonal(distances, np.inf)
    neighbors = np.argsort(distances, axis=1)[:, :2]
    rgb, thermal = clouds['rgb'][0], clouds['thermal'][0]
    parts = np.split(rgb, np.cumsum(clouds['rgb'][2])[:-1])
    residuals = []
    for i, points in enumerate(parts):
        other = np.concatenate([parts[j] for j in neighbors[i]])
        if len(other) and len(points):
            residuals.append(cKDTree(other).query(points, workers=1)[0])
    cross = (np.median(cKDTree(thermal).query(rgb, workers=1)[0]) +
             np.median(cKDTree(rgb).query(thermal, workers=1)[0])) / 2
    normalizer = max(radius, 1e-8)
    return {'camera_radius': radius,
            'neighbor_nn_percent_radius_p50_p90': np.percentile(np.concatenate(residuals) / normalizer * 100, [50, 90]).tolist(),
            'rgb_thermal_symmetric_nn_percent_radius_p50': float(cross / normalizer * 100)}


def initialize_worker():
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    logger.level = 1


def process_scene(task):
    scene, config = task['scene'], task['base_config']
    stage_out, stage_record = Path(task['stage_output']), Path(task['stage_record'])
    for owned_path in (stage_out, stage_record):
        if owned_path.exists():
            shutil.rmtree(owned_path)
    log_path = Path(task['log'])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with log_path.open('w') as log, redirect_stdout(log), redirect_stderr(log):
        previous_argv = sys.argv
        try:
            sys.argv = [str(ROOT / 'scripts/refine_multiview_detail.py'),
                        '--base', task['base'], '--teacher', task['teacher'],
                        '--config', task['config_path'], '--scene', scene['scene'],
                        '--output', str(stage_out), '--record', str(stage_record)]
            runpy.run_path(sys.argv[0], run_name='__main__')
        finally:
            sys.argv = previous_argv
        fixed = dict(np.load(stage_record / 'predictions.npz'))
        images = np.load(stage_record / 'processed_images.npy')
        selections = dict(np.load(stage_record / 'export_pixels.npz'))
        record = json.loads((stage_record / 'run.json').read_text())
        base, teacher = dict(np.load(task['base'])), dict(np.load(task['teacher']))
        verify_completed(stage_out, record)
        verify_png_depth(stage_out, fixed['depth'], scene, record['depth_color_range'])
        assert record['input_sha256'] == task['source_records']['new']['input_sha256']
        assert record['input_sha256'] == task['source_records']['official']['input_sha256']
        for key in ('extrinsics', 'intrinsics', 'depth_conf'):
            assert np.array_equal(fixed[key], base[key]), key
        original_clouds = rebuild_clouds(base, images, scene, config['seed'], config['max_points_per_view'])
        repaired_clouds = rebuild_clouds(fixed, images, scene, config['seed'], config['max_points_per_view'], selections)
        for mode, arrays in (('official', teacher), ('new', base)):
            original_folder = Path(config['output_root']) / mode / scene['scene']
            source_record = task['source_records'][mode]
            verify_completed(original_folder, source_record)
            verify_png_depth(original_folder, arrays['depth'], scene, source_record['depth_color_range'])
            expected_clouds = original_clouds if mode == 'new' else rebuild_clouds(
                arrays, images, scene, config['seed'], config['max_points_per_view'])
            for modality in ('rgb', 'thermal'):
                actual = read_ply(original_folder / f'{modality}.ply')
                assert np.array_equal(actual[0], expected_clouds[modality][0]), (mode, modality, 'source XYZ')
                assert np.array_equal(actual[1], expected_clouds[modality][1]), (mode, modality, 'source colors')
        for modality in ('rgb', 'thermal'):
            actual = read_ply(stage_out / f'{modality}.ply')
            assert np.array_equal(actual[0], repaired_clouds[modality][0]), (modality, 'export XYZ')
            assert np.array_equal(actual[1], repaired_clouds[modality][1]), (modality, 'export colors')
        combined = read_ply(stage_out / 'combined.ply')
        for index in (0, 1):
            assert np.array_equal(combined[index], np.concatenate([repaired_clouds[m][index] for m in ('rgb', 'thermal')]))
        recovered = fixed['recovered_thin_valid']
        assert (fixed['support'][recovered] >= 2).all()
        assert fixed['thin_regions'][recovered].all()
        for i in range(len(scene['views'])):
            assert (base['depth_conf'][i][recovered[i]] >= np.percentile(base['depth_conf'][i], 50) * .5).all()
        audit = {'scene': scene['scene'], 'status': 'passed',
                 'checks': {'source_predictions_exactly_reproduce_original_ply_and_24_pngs': True,
                            'original_files_sha256_match_reference': True,
                            'cameras_intrinsics_confidence_exactly_base': True,
                            'all_exported_xyz_and_observed_colors_exact': True,
                            'all_24_pngs_share_exported_depth': True,
                            'combined_exact_concatenation': True,
                            'thin_recovery_requires_two_rgb_supports_and_confidence': True},
                 'retained_valid_pixel_fraction': record['retained_valid_pixel_fraction'],
                 'additional_supported_thin_pixels': record['additional_supported_thin_pixels'],
                 'exported_points': record['cloud_counts'],
                 'points_per_view': {m: repaired_clouds[m][2] for m in ('rgb', 'thermal')},
                 'old_geometry_consistency': quality_metrics(original_clouds, base['extrinsics']),
                 'repaired_geometry_consistency': quality_metrics(repaired_clouds, fixed['extrinsics']),
                 'elapsed_seconds': time.monotonic() - started,
                 'limits': 'Nearest-neighbor metrics measure consistency and depend on point retention; no ground-truth depth.'}
        atomic_json(stage_record / 'validation.json', audit)
        return {'record': record, 'audit': audit}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'artifacts/comparison_all_scenes_20261010/run_config.json')
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--record-dir', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    output, record_dir = args.output_root.resolve(), args.record_dir.resolve()
    base_config = json.loads(args.config.read_text())
    source_progress = json.loads((args.config.parent / 'progress.json').read_text())
    assert source_progress['status'] == 'complete' and not source_progress['failed']
    scenes = base_config['scenes']
    assert len(scenes) == 60 and all(len(s['views']) == 24 for s in scenes)
    assert file_hash(base_config['adapter']) == base_config['adapter_sha256']
    source_files = ['scripts/refine_multiview_detail.py', 'src/depth_anything_3/thermal/multiview_detail.py',
                    'src/depth_anything_3/thermal/detail_refinement.py', 'scripts/infer_sear_comparison.py',
                    'src/depth_anything_3/thermal/adapter.py', 'scripts/refine_all_multiview_detail.py']
    config = {**base_config, 'output_root': str(output), 'source_output_root': base_config['output_root'],
              'base_config_path': str(args.config.resolve()), 'base_config_sha256': file_hash(args.config),
              'source_progress_sha256': file_hash(args.config.parent / 'progress.json'),
              'method': 'Ebike thin surface geometry anchors and corroborated confidence recovery, same parameters',
              'code_sha256': {p: file_hash(ROOT / p) for p in source_files}}
    progress_path = record_dir / 'progress.json'
    if progress_path.exists():
        if not args.resume:
            raise FileExistsError('Existing run requires --resume')
        assert json.loads((record_dir / 'run_config.json').read_text()) == config
        progress = json.loads(progress_path.read_text())
    else:
        if output.exists() or record_dir.exists():
            raise FileExistsError('Both output and record directories must be new')
        output.mkdir(parents=True); record_dir.mkdir(parents=True)
        progress = {'status': 'running', 'completed': {}, 'cached': {}, 'failed': {}}
        atomic_json(record_dir / 'run_config.json', config)
        for src in source_files:
            destination = record_dir / 'code' / src
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / src, destination)
    atomic_json(progress_path, progress)
    def event(name, **data):
        value = {'time': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(), 'event': name, **data}
        with (record_dir / 'run.log').open('a') as stream:
            stream.write(json.dumps(value, ensure_ascii=False) + '\n')
        print(json.dumps(value, ensure_ascii=False), flush=True)
    event('start', scenes=len(scenes), workers=args.workers, physical_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'))
    # Official output bytes remain exactly those of the reference comparison.
    for scene in scenes:
        key = 'official/' + scene['scene']; destination = output / key
        source = Path(base_config['output_root']) / key
        reference = source_progress['completed'][key]
        verify_completed(source, reference)
        if key not in progress['completed']:
            if destination.exists():
                raise FileExistsError(destination)
            shutil.copytree(source, destination)
            verify_completed(destination, reference)
            progress['completed'][key] = reference
        else:
            verify_completed(destination, progress['completed'][key])
    atomic_json(progress_path, progress)
    event('official_complete', scenes=len(scenes))
    for key, value in progress['cached'].items():
        assert file_hash(value['path']) == value['sha256'], key
    torch.set_num_threads(8); logger.level = 1
    if any('official/' + s['scene'] not in progress['cached'] or 'new/' + s['scene'] not in progress['cached'] for s in scenes):
        if not torch.cuda.is_available():
            raise RuntimeError('Prediction recomputation requires CUDA')
        model = load_giant(base_config['base_checkpoint']).cuda().eval()
        infer_args = SimpleNamespace(seed=base_config['seed'], process_res=base_config['process_res'], rgb_detail_sigma=0)
        event('weights_loaded', gpu=torch.cuda.get_device_name(0))
    else:
        model = None
    for scene in scenes:
        key = 'official/' + scene['scene']
        if key in progress['cached']:
            continue
        arrays, _, statistics = infer_scene(model, scene, 'official', infer_args)
        assert statistics['input_sha256'] == source_progress['completed'][key]['input_sha256']
        path = record_dir / 'cache' / key / 'predictions.npz'; path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **arrays)
        progress['cached'][key] = {'path': str(path), 'sha256': file_hash(path), **statistics}
        atomic_json(progress_path, progress); event('teacher_cached', scene=scene['scene'], cached=len(progress['cached']))
        del arrays; gc.collect()
    if model is not None:
        load_adapter(model, base_config['adapter']); model.eval()
    pool = ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn'), initializer=initialize_worker)
    pending = {}
    def consume(done):
        for future in done:
            task = pending.pop(future); key = 'new/' + task['scene']['scene']
            try:
                result = future.result()
                destination = output / key
                if destination.exists():
                    raise FileExistsError(destination)
                destination.parent.mkdir(parents=True, exist_ok=True)
                Path(task['stage_output']).rename(destination)
                published_record = record_dir / 'scenes' / key
                published_record.parent.mkdir(parents=True, exist_ok=True)
                Path(task['stage_record']).rename(published_record)
                progress['completed'][key] = {**result['record'], 'validation': result['audit'], 'record_dir': str(published_record)}
                progress['failed'].pop(key, None)
                event('scene_complete', scene=task['scene']['scene'], completed_new=sum(k.startswith('new/') for k in progress['completed']),
                      retained=result['record']['retained_valid_pixel_fraction'],
                      recovered=result['record']['additional_supported_thin_pixels'],
                      points=result['record']['cloud_counts'], seconds=result['audit']['elapsed_seconds'])
            except Exception as error:
                progress['failed'][key] = {'error': str(error), 'traceback': traceback.format_exc(), 'log': task['log']}
                event('scene_failed', scene=task['scene']['scene'], error=str(error), log=task['log'])
            atomic_json(progress_path, progress)
    for scene in scenes:
        key = 'new/' + scene['scene']
        if key in progress['completed']:
            verify_completed(output / key, progress['completed'][key]); continue
        if key not in progress['cached']:
            arrays, _, statistics = infer_scene(model, scene, 'new', infer_args)
            assert statistics['input_sha256'] == source_progress['completed'][key]['input_sha256']
            path = record_dir / 'cache' / key / 'predictions.npz'; path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, **arrays)
            progress['cached'][key] = {'path': str(path), 'sha256': file_hash(path), **statistics}
            atomic_json(progress_path, progress); event('base_cached', scene=scene['scene'], cached=len(progress['cached']))
            del arrays; gc.collect()
        task = {'scene': scene, 'base_config': base_config, 'config_path': str(args.config.resolve()),
                'base': progress['cached'][key]['path'], 'teacher': progress['cached']['official/' + scene['scene']]['path'],
                'stage_output': str(record_dir / 'staging' / key / 'output'),
                'stage_record': str(record_dir / 'staging' / key / 'record'),
                'log': str(record_dir / 'logs' / (scene['scene'] + '.log')),
                'source_records': {m: source_progress['completed'][m + '/' + scene['scene']] for m in ('official', 'new')}}
        pending[pool.submit(process_scene, task)] = task
        consume({f for f in pending if f.done()})
    del model; gc.collect(); torch.cuda.empty_cache()
    event('prediction_cache_complete', cached=len(progress['cached']), pending=len(pending))
    while pending:
        done, _ = wait(pending, timeout=5, return_when=FIRST_COMPLETED)
        consume(done)
    pool.shutdown()
    expected_keys = {mode + '/' + scene['scene'] for scene in scenes for mode in ('official', 'new')}
    if set(progress['completed']) != expected_keys or progress['failed']:
        progress['status'] = 'incomplete'; atomic_json(progress_path, progress)
        event('finished', status='incomplete', completed=len(progress['completed']), failed=progress['failed'])
        raise SystemExit(1)
    original_files, actual_files = set(), set()
    totals = Counter()
    for key in sorted(expected_keys):
        reference = source_progress['completed'][key]
        current = progress['completed'][key]
        assert {f['path'] for f in current['files']} == {f['path'] for f in reference['files']}
        verify_completed(output / key, current)
        for entry in current['files']:
            actual_files.add(key + '/' + entry['path']); totals[Path(entry['path']).suffix] += 1
        original_files.update(key + '/' + entry['path'] for entry in reference['files'])
    assert actual_files == original_files
    assert actual_files == {p.relative_to(output).as_posix() for p in output.rglob('*') if p.is_file()}
    audits = [progress['completed']['new/' + s['scene']]['validation'] for s in scenes]
    p90_regressions = [a['scene'] for a in audits if a['repaired_geometry_consistency']['neighbor_nn_percent_radius_p50_p90'][1]
                       > a['old_geometry_consistency']['neighbor_nn_percent_radius_p50_p90'][1] * 1.1]
    low_retention = [a['scene'] for a in audits if min(a['retained_valid_pixel_fraction'].values()) < .7]
    summary = {'status': 'passed', 'validated_at': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
               'output_root': str(output), 'scenes': len(scenes), 'scene_model_combinations': len(expected_keys),
               'failed': [], 'files': len(actual_files), 'png_files': totals['.png'], 'ply_files': totals['.ply'],
               'dataset_scene_counts': dict(Counter(s['scene'].split('/')[0] for s in scenes)),
               'checks': ['exact filename and scene set of original comparison', 'official files byte-for-byte original',
                          'all recomputed inputs and source PNG/PLY match original', 'same Ebike method in all scenes',
                          'per-scene camera/PNG/PLY/color/support verification', 'all output sizes and SHA-256'],
               'neighbor_p90_more_than_10percent_worse_scenes': p90_regressions,
               'below_70percent_original_valid_pixel_retention_scenes': low_retention,
               'additional_supported_thin_pixels': {m: sum(a['additional_supported_thin_pixels'][m] for a in audits) for m in ('rgb', 'thermal')},
               'limits': 'Consistency depends on filtering; matching format and passed export audits do not guarantee better ground-truth geometry in every scene.'}
    atomic_json(record_dir / 'output_validation.json', summary)
    atomic_json(record_dir / 'scene_metrics.json', {a['scene']: a for a in audits})
    progress['status'] = 'complete'; atomic_json(progress_path, progress)
    event('finished', **summary)


if __name__ == '__main__':
    main()
