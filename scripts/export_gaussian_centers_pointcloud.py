#!/usr/bin/env python3
"""Export saved 3DGS centers as ordinary XYZ/RGB point clouds.

Retains one point per saved Gaussian, its native coordinate frame, and the
view-independent DC color. PLY files contain no Gaussian or mesh attributes.
"""

import argparse
import gc
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from plyfile import PlyData, PlyElement
import torch

PROPERTIES = ('x', 'y', 'z', 'red', 'green', 'blue')
POINT_DTYPE = [(name, '<f4' if i < 3 else 'u1') for i, name in enumerate(PROPERTIES)]
SH_C0 = 0.28209479177387814


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 ** 2), b''):
            result.update(block)
    return result.hexdigest()


def save_and_verify(path, vertices):
    PlyData([PlyElement.describe(vertices, 'vertex')], text=False, byte_order='<').write(path)
    saved = PlyData.read(path)
    assert [element.name for element in saved.elements] == ['vertex']
    assert saved['vertex'].data.dtype.names == PROPERTIES
    np.testing.assert_array_equal(saved['vertex'].data, vertices)
    return {'path': str(path.resolve()), 'points': len(vertices),
            'bytes': path.stat().st_size, 'sha256': digest(path),
            'verification': 'All XYZ/RGB values equal the exported source values'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    config = json.loads((args.source / 'run_config.json').read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    clouds, sources, files = [], {}, {}
    for modality in ('rgb', 'thermal'):
        path = args.source / f'gaussians_{modality}.pt'
        data = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
        points = data['means'][0].numpy()
        colors = (data['harmonics'][0, :, :, 0] * SH_C0 + 0.5).clamp(0, 1).numpy()
        assert points.shape == colors.shape == (config['gaussian_counts'][modality], 3)
        assert np.isfinite(points).all() and np.isfinite(colors).all()
        vertices = np.empty(len(points), dtype=POINT_DTYPE)
        pixels = np.rint(colors * 255).astype(np.uint8)
        for i, name in enumerate(PROPERTIES[:3]):
            vertices[name] = points[:, i]
        for i, name in enumerate(PROPERTIES[3:]):
            vertices[name] = pixels[:, i]
        files[modality] = save_and_verify(args.output / f'{modality}.ply', vertices)
        sources[modality] = {'path': str(path.resolve()), 'sha256': digest(path),
                             'points': len(points), 'coordinate_min': points.min(axis=0).tolist(),
                             'coordinate_max': points.max(axis=0).tolist()}
        clouds.append(vertices)
        print(f'{modality}: exported and verified {len(points):,} points', flush=True)
        del data, points, colors, pixels
        gc.collect()
    files['combined'] = save_and_verify(args.output / 'combined.ply', np.concatenate(clouds))
    metadata = {'status': 'passed', 'source': str(args.source.resolve()),
                'source_config_sha256': digest(args.source / 'run_config.json'),
                'script_sha256': digest(Path(__file__)), 'sources': sources, 'files': files,
                'properties': list(PROPERTIES), 'coordinate_frame': 'Unchanged native Gaussian world coordinates',
                'sampling': 'All saved Gaussians, exactly one center per Gaussian; no additional filtering or sampling',
                'color': 'uint8 RGB from clamp(SH_C0 * harmonics[..., 0] + 0.5, 0, 1)',
                'combined_order': ['rgb', 'thermal']}
    (args.output / 'export_metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + '\n')
    shutil.copy2(Path(__file__), args.output / 'export_script_snapshot.py')
    (args.output / 'README.md').write_text(
        '高斯中心的普通彩色点云\n\n'
        f'直接从 {args.source.resolve()} 的 gaussians_rgb.pt / gaussians_thermal.pt 导出。'
        '每个已保存高斯对应一个点，坐标逐元素保留，不再抽样或筛选。\n\n'
        f'- rgb.ply：{files["rgb"]["points"]:,} 个 RGB 高斯中心。\n'
        f'- thermal.ply：{files["thermal"]["points"]:,} 个红外高斯中心。\n'
        f'- combined.ply：两者合并，{files["combined"]["points"]:,} 个点。\n\n'
        '三个文件均为普通 binary little-endian PLY，仅有 x/y/z 和 red/green/blue 六个属性。'
        '没有 scale、rotation、opacity、球谐系数、半径或面，按普通点云读取。\n\n'
        '颜色取高斯的基础 DC 颜色（SH_C0 × DC + 0.5），不含随观察方向变化的高阶球谐效果。'
        '坐标保持原模型预测坐标系，未转换到 COLMAP 或真实米制尺度。'
        'pointcloud_replay 是原来从深度反投影、抽样的点云，与本次直接取高斯中心的结果不同。\n\n'
        '所有点数、坐标、颜色与 PLY 属性已通过写出后逐元素读取校验，详情见 export_metadata.json。\n')
    print(f'Combined: {files["combined"]["points"]:,} ordinary points', flush=True)


if __name__ == '__main__':
    main()
