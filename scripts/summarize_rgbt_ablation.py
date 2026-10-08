#!/usr/bin/env python3
"""Aggregate paired ablations by scene, with scene bootstrap confidence intervals."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

METRICS = [
    "rgb_psnr",
    "thermal_psnr",
    "rgb_ssim",
    "thermal_ssim",
    "rgb_lpips",
    "depth_log_error",
    "fixed_surface_thickness",
]


def summarize(rows):
    reference = {
        (r["data"], r["id"], r["views"]): r for r in rows if r["variant"] == "full"
    }
    paired = []
    for row in rows:
        key = (row["data"], row["id"], row["views"])
        if key not in reference:
            continue
        base = reference[key]
        if row["context_ids"] != base["context_ids"] or row["scene"] != base["scene"]:
            raise ValueError(f"Unpaired evaluation at {key}")
        for metric in METRICS:
            if row.get(metric) is None or base.get(metric) is None:
                continue
            paired.append(
                {
                    "variant": row["variant"],
                    "scene": row["scene"],
                    "split": row["split"],
                    "views": row["views"],
                    "metric": metric,
                    "value": row[metric],
                    "baseline": base[metric],
                    "delta": row[metric] - base[metric],
                }
            )
    grouped = {}
    for r in paired:
        key = (r["variant"], r["scene"], r["split"], r["views"], r["metric"])
        grouped.setdefault(key, []).append(r)
    scenes = []
    for (variant, scene, split, views, metric), group in grouped.items():
        scenes.append(
            {
                "variant": variant,
                "scene": scene,
                "split": split,
                "views": views,
                "metric": metric,
                "value": float(np.mean([r["value"] for r in group])),
                "baseline": float(np.mean([r["baseline"] for r in group])),
                "delta": float(np.mean([r["delta"] for r in group])),
                "count": len(group),
            }
        )
    summary = []
    for variant in sorted({r["variant"] for r in scenes}):
        for split in ["all", "train", "val", "test", "nontrain"]:
            for views in [3, 6]:
                for metric in METRICS:
                    group = [
                        r
                        for r in scenes
                        if r["variant"] == variant
                        and r["views"] == views
                        and r["metric"] == metric
                        and (
                            split == "all"
                            or r["split"] == split
                            or (split == "nontrain" and r["split"] != "train")
                        )
                    ]
                    if not group:
                        continue
                    delta = np.array([r["delta"] for r in group])
                    rng = np.random.default_rng(20260920)
                    bootstrap = rng.choice(
                        delta, (10000, len(delta)), replace=True
                    ).mean(1)
                    direction = (
                        -1
                        if metric
                        in ["rgb_lpips", "depth_log_error", "fixed_surface_thickness"]
                        else 1
                    )
                    summary.append(
                        {
                            "variant": variant,
                            "split": split,
                            "views": views,
                            "metric": metric,
                            "value": float(np.mean([r["value"] for r in group])),
                            "baseline": float(np.mean([r["baseline"] for r in group])),
                            "delta": float(delta.mean()),
                            "ci95_low": float(np.quantile(bootstrap, 0.025)),
                            "ci95_high": float(np.quantile(bootstrap, 0.975)),
                            "scene_count": len(delta),
                            "count": sum(r["count"] for r in group),
                            "improved_scenes": int((direction * delta > 1e-6).sum()),
                            "worse_scenes": int((direction * delta < -1e-6).sum()),
                            "worst_delta": float(delta[np.argmin(delta * direction)]),
                        }
                    )
    return summary, scenes


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("inputs", nargs="+")
    p.add_argument("--output", required=True)
    a = p.parse_args()
    rows, keys = [], set()
    for root in a.inputs:
        for path in sorted(Path(root).glob("rows_*.jsonl")):
            for line in path.read_text().splitlines():
                row = json.loads(line)
                key = (row["data"], row["id"], row["views"], row["variant"])
                if key in keys:
                    raise ValueError(f"Duplicate result {key}")
                keys.add(key)
                rows.append(row)
    summary, scenes = summarize(rows)
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    for name, data in [("summary", summary), ("per_scene", scenes)]:
        (out / (name + ".json")).write_text(json.dumps(data, indent=2) + "\n")
        if data:
            with (out / (name + ".csv")).open("w") as f:
                writer = csv.DictWriter(f, list(data[0]))
                writer.writeheader()
                writer.writerows(data)
    coverage = {
        "rows": len(rows),
        "scenes": len({r["scene"] for r in rows}),
        "targets": len({(r["data"], r["id"]) for r in rows}),
        "cases": len({(r["data"], r["id"], r["views"]) for r in rows}),
        "variants": sorted({r["variant"] for r in rows}),
        "uncertainty": "95% percentile bootstrap of scene means, 10000 draws, seed 20260920; conditional on frozen checkpoint, no retraining seeds; multiple comparisons are exploratory.",
    }
    (out / "coverage.json").write_text(json.dumps(coverage, indent=2) + "\n")
    print(json.dumps(coverage))
    for r in summary:
        if r["split"] == "nontrain" and r["metric"] in ["rgb_psnr", "thermal_psnr"]:
            print(
                f"{r['variant']:35s} {r['views']} {r['metric']:14s} {r['delta']:+.5f} [{r['ci95_low']:+.5f}, {r['ci95_high']:+.5f}] {r['improved_scenes']}/{r['scene_count']}"
            )


if __name__ == "__main__":
    main()
