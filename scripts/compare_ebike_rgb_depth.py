#!/usr/bin/env python3
"""Compare corresponding saved input-view depth maps, without confidence filtering."""
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT/'outputs/novel_views/ThermalGaussian/Ebike_gsdpt_20261012'
NEW = ROOT/'outputs/novel_views/ThermalGaussian/Ebike_rgb_detail_refine_gsdpt_20261013'
OUT = ROOT/'artifacts/ebike_rgb_depth_comparison_20261013'


def statistics(a, b):
    relative = np.abs(b-a)/a
    return {
        'mean_absolute_relative_difference': float(relative.mean()),
        'median_absolute_relative_difference': float(np.median(relative)),
        'p90_absolute_relative_difference': float(np.percentile(relative, 90)),
        'fraction_within_5_percent': float(np.mean(relative <= .05)),
        'fraction_within_10_percent': float(np.mean(relative <= .1)),
        'fraction_within_20_percent': float(np.mean(relative <= .2)),
        'pearson_correlation': float(np.corrcoef(a.ravel(), b.ravel())[0, 1]),
    }


def plot_sheet(a, b, views, indices, filename, scale=1.):
    shown = b*scale
    lo, hi = np.percentile(np.concatenate([a.ravel(), shown.ravel()]), [1, 99])
    fig, axes = plt.subplots(len(indices), 4, figsize=(16, 3.1*len(indices)), constrained_layout=True)
    for row, index in enumerate(indices):
        axes[row, 0].imshow(Image.open(views[index]['image']).convert('RGB'))
        axes[row, 1].imshow(a[index], cmap='magma', vmin=lo, vmax=hi)
        depth_im = axes[row, 2].imshow(shown[index], cmap='magma', vmin=lo, vmax=hi)
        diff_im = axes[row, 3].imshow((shown[index]-a[index])/a[index]*100,
                                     cmap='RdBu_r', vmin=-30, vmax=30)
        name = Path(views[index]['image']).stem
        labels = [f'{name}: RGB input', 'Previous trained depth',
                  'Detail refine depth' + (f' x {scale:.4f}' if scale!=1. else ' (raw)'),
                  '(new - old) / old [%]']
        for ax, label in zip(axes[row], labels):
            ax.set_title(label, fontsize=10)
            ax.set_axis_off()
    fig.colorbar(depth_im, ax=axes[:, 1:3].ravel().tolist(), shrink=.4,
                 label='Shared depth color scale (prediction units), 1-99% display clip')
    fig.colorbar(diff_im, ax=axes[:, 3].ravel().tolist(), shrink=.4,
                 label='Signed relative difference (%), display clipped to +/-30%')
    fig.savefig(OUT/filename, dpi=130)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    old_info = json.loads((OLD/'run_config.json').read_text())
    new_info = json.loads((NEW/'run_config.json').read_text())
    assert old_info['input_sha256'] == new_info['input_sha256']
    views = old_info['scene']['views'][:12]
    assert [Path(v['image']).stem for v in views] == [Path(v['image']).stem for v in new_info['scene']['views'][:12]]
    a = np.load(OLD/'predictions.npz')['depth'][:12].astype(np.float64)
    b = np.load(NEW/'predictions.npz')['depth'][:12].astype(np.float64)
    assert a.shape == b.shape == (12, 392, 518)
    assert np.isfinite(a).all() and np.isfinite(b).all() and (a>0).all() and (b>0).all()
    scale = float(np.exp(np.median(np.log(a)-np.log(b))))
    rows = []
    for protocol, values in [('raw', b), ('global_scale_aligned', b*scale)]:
        for i, view in enumerate(views):
            rows.append({'protocol': protocol, 'view': Path(view['image']).stem,
                         **statistics(a[i], values[i])})
    result = {
        'old_prediction': str(OLD/'predictions.npz'), 'new_prediction': str(NEW/'predictions.npz'),
        'old_adapter': old_info['adapter'], 'new_adapter': new_info['adapter'],
        'input_sha256': old_info['input_sha256'], 'views': 12, 'image_hw': [392, 518],
        'depth_source': 'Saved per-input RGB depth used as the GaussianAdapter base, before GSDPT offsets',
        'mask': 'All positive finite pixels, no confidence filter or border crop',
        'relative_difference_denominator': 'Previous trained depth',
        'global_new_to_old_scale': scale,
        'alignment': 'One sequence-wide scale = exp(median(log(old)-log(new))); no per-view scale or shift',
        'raw': statistics(a, b), 'global_scale_aligned': statistics(a, b*scale),
        'per_view': rows,
    }
    (OUT/'metrics.json').write_text(json.dumps(result, indent=2)+'\n')
    with (OUT/'per_view.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plot_sheet(a, b, views, [0, 3, 6, 11], 'comparison_raw.jpg')
    plot_sheet(a, b, views, [0, 3, 6, 11], 'comparison_scale_aligned.jpg', scale)
    plot_sheet(a, b, views, list(range(12)), 'all_views_scale_aligned.jpg', scale)
    print(json.dumps({k: result[k] for k in ['global_new_to_old_scale', 'raw', 'global_scale_aligned']}, indent=2))


if __name__ == '__main__':
    main()
