#!/usr/bin/env python3
"""Render matched 4x3 exposure panels using the official sweep's color ranges."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--official-run", type=Path, required=True)
    parser.add_argument("--title", default="New")
    args = parser.parse_args()
    root, official = args.run.resolve(), args.official_run.resolve()
    config = json.loads((root / "run_config.json").read_text())
    summary = json.loads((root / "summary.json").read_text())
    comparison = json.loads((official / "summary.json").read_text())
    assert summary["status"] == comparison["status"] == "complete"
    reference = json.loads(Path(config["reference_config"]).read_text())
    scene = next(s for s in reference["scenes"] if s["scene"] == "ThermalGaussian/Ebike")
    count = len(scene["selected_pair_indices"])
    views = scene["views"][:count]
    dataset = Path(config["variants"]["normal"]["rgb_root"]).parent
    manifest = json.loads((dataset / "lowlight_processing/manifest.json").read_text())
    names = {Path(i["relative_path"]).name: i["relative_path"] for i in manifest["images"]}
    labels = ("normal", "ev4", "ev6", "ev8")
    protocols = ("rgb_only", "rgb_thermal")
    depths, images = {}, {}
    for label in labels:
        for protocol in protocols:
            with np.load(root / "predictions" / label / protocol / "predictions.npz") as arrays:
                assert arrays["depth"].shape == (count if protocol == "rgb_only" else 2 * count, 392, 518)
                assert np.isfinite(arrays["depth"]).all() and (arrays["depth"] > 0).all()
                depths[label, protocol] = arrays["depth"][:count].copy()
        for index in np.linspace(0, count - 1, 4, dtype=int):
            source = Path(config["variants"][label]["rgb_root"]) / names[Path(views[index]["image"]).name]
            with Image.open(source) as image:
                images[label, int(index)] = np.asarray(image.convert("RGB")).copy()
    with np.load(official / "predictions/normal/rgb_only/predictions.npz") as anchor:
        anchor_median = float(np.median(anchor["depth"]))
    aligned, scales = {}, {}
    for key, depth in depths.items():
        scales["/".join(key)] = float(anchor_median / np.median(depth))
        aligned[key] = depth * scales["/".join(key)]
    ranges = {"aligned": comparison["display_color_range"],
              "raw_scale": comparison["raw_scale_display_color_range"]}
    plots = {"aligned": [], "raw_scale": []}
    for display, displayed in (("aligned", aligned), ("raw_scale", depths)):
        low, high = ranges[display]
        if display == "aligned":
            for label in labels:
                for protocol in protocols:
                    directory = root / "depth" / label / protocol
                    directory.mkdir(parents=True, exist_ok=True)
                    for i, view in enumerate(views):
                        colored = np.rint(matplotlib.colormaps["turbo"](
                            np.clip((displayed[label, protocol][i] - low) / (high - low), 0, 1))[..., :3] * 255).astype(np.uint8)
                        Image.fromarray(colored).save(directory / f"{view['pair_index']:05d}_{Path(view['image']).stem}.png")
        for index in np.linspace(0, count - 1, 4, dtype=int):
            figure, axes = plt.subplots(4, 3, figsize=(13, 12), squeeze=False)
            for row, label in enumerate(labels):
                ev = config["variants"][label]["exposure_ev"]
                axes[row, 0].imshow(images[label, int(index)])
                axes[row, 0].set_title(f"Source RGB | {'normal' if ev == 0 else str(ev) + ' EV'}")
                for column, protocol in enumerate(protocols, 1):
                    artist = axes[row, column].imshow(displayed[label, protocol][index], cmap="turbo", vmin=low, vmax=high)
                    suffix = "RGB only" if protocol == "rgb_only" else "RGB + thermal"
                    axes[row, column].set_title(f"{args.title} | {suffix}")
                for axis in axes[row]:
                    axis.set_axis_off()
            name = Path(views[index]["image"]).stem
            scale_label = "one median scale per sequence" if display == "aligned" else "no scale alignment"
            figure.suptitle(f"{name} | {args.title} | {scale_label} | fixed color range", fontsize=13)
            figure.subplots_adjust(left=.02, right=.98, top=.94, bottom=.065, wspace=.02, hspace=.16)
            color_axis = figure.add_axes([.25, .025, .5, .012])
            figure.colorbar(artist, cax=color_axis, orientation="horizontal",
                           label="Color range shared with official; relative depth; no ground-truth depth")
            filename = f"comparison_{name}.png" if display == "aligned" else f"comparison_raw_scale_{name}.png"
            figure.savefig(root / filename, dpi=130)
            plt.close(figure)
            plots[display].append(filename)
    metrics = []
    for label in labels:
        for protocol in protocols:
            a = aligned["normal", protocol][:, 8:-8, 8:-8]
            b = aligned[label, protocol][:, 8:-8, 8:-8]
            metrics.append({"exposure": label, "protocol": protocol,
                            "mean_relative_change_vs_normal_prediction": float(np.mean(np.abs(a - b) / a))})
    metadata = {"status": "complete", "title": args.title, "plots": plots,
                "color_ranges": ranges, "official_run": str(official),
                "sequence_scales": scales, "metrics": metrics,
                "layout": {"rows": ["normal", "-4 EV", "-6 EV", "-8 EV"],
                           "columns": ["source RGB", "new RGB only", "new RGB + thermal"]},
                "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "metric_interpretation": "Prediction changes relative to each protocol's own normal-light prediction; not ground-truth accuracy."}
    (root / "plot_summary.json").write_text(json.dumps(metadata, indent=2) + "\n")
    shutil.copy2(Path(__file__), root / Path(__file__).name)
    print(json.dumps({"output": str(root), "plots": plots, "color_ranges": ranges}))


if __name__ == "__main__":
    main()
