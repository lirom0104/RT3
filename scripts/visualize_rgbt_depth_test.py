#!/usr/bin/env python3
"""Export test-only target-view depth visualizations, without HTML."""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import cv2
import matplotlib
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

matplotlib.use("Agg")
from matplotlib import colormaps

from ablate_rgbt import VARIANTS, configured
from depth_anything_3.rgbt.checkpoint import load_rgbt_model
from depth_anything_3.rgbt.data import SceneDataset
from depth_anything_3.rgbt.engine import batch_to, depth_loss
from depth_anything_3.rgbt.renderer import render_rgbt
from depth_anything_3.rgbt.geometry import gaussian_depth


FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
DEPTH_CMAP = colormaps["turbo"]


def font(size):
    return ImageFont.truetype(FONT_PATH, size)


def rgb_bitmap(tensor):
    array = tensor.detach().float().clamp(0, 1).cpu().numpy()
    array = np.moveaxis(array, 0, -1)
    return Image.fromarray((array * 255 + 0.5).astype(np.uint8))


def finite_positive(array):
    return np.isfinite(array) & (array > 0)


def depth_color(depth, vmin, vmax):
    """Colorize inverse camera-z depth using one range for the whole panel."""
    depth = np.asarray(depth, np.float32)
    inverse = np.zeros_like(depth)
    positive = finite_positive(depth)
    inverse[positive] = 1.0 / np.maximum(depth[positive], 1e-6)
    normalized = np.clip((inverse - vmin) / max(vmax - vmin, 1e-6), 0, 1)
    image = (DEPTH_CMAP(normalized)[..., :3] * 255 + 0.5).astype(np.uint8)
    image[~positive] = 0
    return Image.fromarray(image, "RGB")


def gray_bitmap(array, mask=None):
    array = np.asarray(array, np.float32)
    if mask is None:
        mask = np.isfinite(array)
    finite = array[mask]
    if finite.size:
        lo, hi = np.percentile(finite, [1, 99])
    else:
        lo, hi = 0, 1
    normalized = np.clip((array - lo) / max(hi - lo, 1e-6), 0, 1)
    image = np.repeat((normalized * 255 + 0.5).astype(np.uint8)[..., None], 3, -1)
    image[~mask] = 0
    return Image.fromarray(image, "RGB")


def panel(path, item, batch, predictions, stats, views, raw_dir):
    h, w = batch["rgb"].shape[-2:]
    header, label, footer, gap = 66, 42, 62, 8
    canvas = Image.new(
        "RGB", (3 * w + 4 * gap, header + 2 * (h + label) + gap + footer), "white"
    )
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (gap, 8), f"{item['scene']} | TEST | {views} source views | TARGET DEPTH",
        fill="#151515", font=font(17)
    )
    draw.text((gap, 36), item["id"], fill="#444444", font=font(12))
    sparse = batch["depth"][0, -1].detach().float().cpu().numpy()
    rgb = batch["rgb"][0, -1]
    images = [
        ("Target RGB reference", rgb_bitmap(rgb)),
        (
            "DA3 rendered target depth",
            depth_color(predictions["da3"]["depth"], stats["vmin"], stats["vmax"]),
        ),
        (
            "SharedRefiner rendered target depth",
            depth_color(predictions["shared"]["depth"], stats["vmin"], stats["vmax"]),
        ),
        (
            "COLMAP sparse depth (target)",
            depth_color(sparse, stats["vmin"], stats["vmax"]),
        ),
        (
            f"DA3 alpha | coverage {stats['da3_alpha_coverage'] * 100:.1f}%",
            gray_bitmap(predictions["da3"]["alpha"], predictions["da3"]["alpha"] > 0.02),
        ),
        (
            f"Shared alpha | coverage {stats['shared_alpha_coverage'] * 100:.1f}%",
            gray_bitmap(predictions["shared"]["alpha"], predictions["shared"]["alpha"] > 0.02),
        ),
    ]
    for index, (title, image) in enumerate(images):
        row, col = divmod(index, 3)
        x, y = gap + col * (w + gap), header + row * (h + label + gap)
        draw.text((x, y), title, fill="#151515", font=font(12))
        canvas.paste(image, (x, y + label))
    draw.text(
        (gap, canvas.height - 51),
        f"Depth color = inverse camera-z; shared range {stats['z_near']:.3g} to {stats['z_far']:.3g}; black = no depth hit.",
        fill="#555555", font=font(11)
    )
    draw.text(
        (gap, canvas.height - 34),
        "COLMAP target depth is sparse and held out from model inputs; it is not a dense ground truth map.",
        fill="#555555", font=font(11)
    )
    draw.text(
        (gap, canvas.height - 17),
        "DA3 = original weights + RGB-T wrapper; SharedRefiner = trained RGB-T model; same target camera.",
        fill="#555555", font=font(11)
    )
    canvas.save(path)
    np.savez_compressed(
        raw_dir / "depth.npz",
        target_rgb=batch["rgb"][0, -1].detach().float().cpu().numpy(),
        depth_sparse=sparse,
        depth_da3=predictions["da3"]["depth"],
        depth_shared=predictions["shared"]["depth"],
        alpha_da3=predictions["da3"]["alpha"],
        alpha_shared=predictions["shared"]["alpha"],
        valid=batch["valid"][0, -1].detach().float().cpu().numpy(),
        inverse_depth_vmin=stats["vmin"],
        inverse_depth_vmax=stats["vmax"],
    )


def overview(root, scene, protocol, rows):
    indices = np.linspace(0, len(rows) - 1, min(3, len(rows))).round().astype(int)
    selected = [rows[i] for i in indices]
    images = [Image.open(root / r["comparison"]).convert("RGB") for r in selected]
    canvas = Image.new("RGB", (images[0].width, sum(i.height for i in images) + 44), "#eeeeee")
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 10), f"{scene} | {protocol} | evenly spaced {len(selected)}/{len(rows)} targets",
              fill="#151515", font=font(16))
    y = 44
    for image in images:
        canvas.paste(image, (0, y))
        y += image.height
        image.close()
    dest = root / "overviews" / f"{scene.replace('/', '_')}_{protocol}.png"
    dest.parent.mkdir(exist_ok=True)
    canvas.save(dest)


def sparse_metrics(prediction, sparse, valid):
    mask = finite_positive(sparse) & finite_positive(prediction) & (valid > 0.5)
    if not mask.any():
        return {"count": 0, "coverage": 0.0, "log_error": None, "abs_relative": None}
    pred, target = prediction[mask], sparse[mask]
    return {
        "count": int(mask.sum()),
        "coverage": float(mask.sum() / max((finite_positive(sparse) & (valid > 0.5)).sum(), 1)),
        "log_error": float(np.abs(np.log(pred) - np.log(target)).mean()),
        "abs_relative": float(np.abs(pred - target).mean() / np.maximum(target, 1e-6).mean()),
    }


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", default="checkpoints/RGBT/model.pt")
    parser.add_argument("--data", default="outputs/rgbt/data_336")
    parser.add_argument("--reference", default="outputs/rgbt/evaluation_20260922")
    parser.add_argument("--dataset", required=True, choices=["RGBT-Scenes", "ThermoScenes"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    torch.set_num_threads(4)
    cv2.setNumThreads(2)
    torch.cuda.set_device(args.device)
    root = args.output
    root.mkdir(parents=True, exist_ok=True)
    rows_path = root / f"rows_{args.dataset}.jsonl"
    if rows_path.exists() or (root / args.dataset).exists():
        parser.error("Use a fresh output directory for this dataset")
    started = time.time()
    model, checkpoint = load_rgbt_model(args.weights, args.device)
    del checkpoint
    dataset_metadata = {
        "dataset": args.dataset,
        "split": "test",
        "views": [3, 6],
        "weights": str(Path(args.weights).resolve()),
        "data": str(Path(args.data).resolve()),
        "target_camera_depth": "render_rgbt depth at the held-out target COLMAP camera",
        "target_sparse_reference": "COLMAP sparse points projected into the held-out target camera; not used as model input",
        "depth_color": "turbo colormap of inverse camera-z with one shared robust range per panel; black means no rendered depth",
        "image_layout": [
            "Target RGB reference", "DA3 target depth", "SharedRefiner target depth",
            "COLMAP sparse target depth", "DA3 alpha", "SharedRefiner alpha",
        ],
        "baseline": "Original DA3 weights with existing RGB-T wrapper, teacher=True, all refiners disabled",
        "source_metrics": "log absolute depth error against sparse source COLMAP depths; target metrics are sparse-only and report coverage",
    }
    (root / f"metadata_{args.dataset}.json").write_text(json.dumps(dataset_metadata, indent=2) + "\n")
    reference = json.loads((Path(args.reference) / args.dataset / "results.json").read_text())
    all_rows = []
    for views in [3, 6]:
        protocol = "nearest3" if views == 3 else "wide6"
        ref_rows = {
            r["id"]: r for r in reference["results"][protocol]["per_view"] if r["split"] == "test"
        }
        dataset = SceneDataset(args.data, "test", views=views, targets_per_scene=100000,
                               sampling="nearest" if views == 3 else "wide")
        dataset.items = [(si, ti) for si, ti in dataset.items
                         if dataset.scenes[si]["domain"] == args.dataset]
        protocol_rows = []
        for index in range(len(dataset)):
            item = dataset[index]
            if item["id"] in item["context_ids"] or item["context_ids"] != ref_rows[item["id"]]["context_ids"]:
                raise ValueError("Source/target pairing differs from previous evaluation")
            batch = batch_to(item, args.device)
            inputs = [batch[k][:, :views] for k in ["rgb", "thermal", "E", "K"]]
            target_E, target_K = batch["E"][:, views:], batch["K"][:, views:]
            h, w = batch["rgb"].shape[-2:]
            with configured(model, VARIANTS["da3_rgbt_wrapper"]), torch.autocast("cuda", dtype=torch.bfloat16):
                baseline = model.reconstruction(*inputs, teacher=True)
            rendered_da3 = render_rgbt(baseline, target_E, target_K, h, w)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                shared = model(*inputs, source_valid=batch["valid"][:, :views])
            rendered_shared = render_rgbt(shared, target_E, target_K, h, w)
            source_gt = batch["depth"][:, :views]
            source_metrics = {}
            for name, output in [("da3", baseline), ("shared", shared)]:
                source_pred = gaussian_depth(output)
                source_metrics[name] = {
                    "log_error": float(depth_loss(source_pred, source_gt)),
                    "valid_points": int(((source_pred > 0) & (source_gt > 0)).sum()),
                }
            sparse = batch["depth"][0, -1].float().cpu().numpy()
            valid = batch["valid"][0, -1].float().cpu().numpy()
            preds = {
                "da3": {
                    "depth": rendered_da3["depth"][0, 0].float().cpu().numpy(),
                    "alpha": rendered_da3["alpha"][0, 0, 0].float().cpu().numpy(),
                },
                "shared": {
                    "depth": rendered_shared["depth"][0, 0].float().cpu().numpy(),
                    "alpha": rendered_shared["alpha"][0, 0, 0].float().cpu().numpy(),
                },
            }
            all_positive = [sparse] + [p["depth"] for p in preds.values()]
            inverse_values = np.concatenate([1.0 / x[finite_positive(x)] for x in all_positive if finite_positive(x).any()])
            vmin, vmax = np.percentile(inverse_values, [2, 98])
            stats = {
                "vmin": float(vmin), "vmax": float(vmax),
                "z_near": float(1 / vmax), "z_far": float(1 / vmin),
                "da3_alpha_coverage": float((preds["da3"]["alpha"] > 0.02).mean()),
                "shared_alpha_coverage": float((preds["shared"]["alpha"] > 0.02).mean()),
                "da3_target_sparse": sparse_metrics(preds["da3"]["depth"], sparse, valid),
                "shared_target_sparse": sparse_metrics(preds["shared"]["depth"], sparse, valid),
            }
            folder = root / item["scene"] / protocol
            raw_dir = folder / "raw" / item["id"]
            raw_dir.mkdir(parents=True, exist_ok=True)
            path = folder / f"{item['id']}.png"
            panel(path, item, batch, preds, stats, views, raw_dir)
            row = {
                "id": item["id"], "scene": item["scene"], "split": "test", "dataset": args.dataset,
                "protocol": protocol, "views": views, "context_ids": item["context_ids"],
                "comparison": path.relative_to(root).as_posix(),
                "raw": raw_dir.relative_to(root).as_posix(),
                "source_depth": source_metrics, "target_sparse": {
                    "points": int((finite_positive(sparse) & (valid > 0.5)).sum()),
                    "da3": stats["da3_target_sparse"], "shared": stats["shared_target_sparse"],
                },
                "target_alpha_coverage": {
                    "da3": stats["da3_alpha_coverage"], "shared": stats["shared_alpha_coverage"],
                },
                "color_range": {"inverse_depth_vmin": stats["vmin"], "inverse_depth_vmax": stats["vmax"],
                                "z_near": stats["z_near"], "z_far": stats["z_far"]},
            }
            protocol_rows.append(row)
            with rows_path.open("a") as stream:
                stream.write(json.dumps(row) + "\n")
            print(args.dataset, protocol, f"{index + 1}/{len(dataset)}", item["id"], flush=True)
            del baseline, shared, rendered_da3, rendered_shared
        if {r["id"] for r in protocol_rows} != set(ref_rows):
            raise ValueError("Missing test targets")
        for scene in sorted({r["scene"] for r in protocol_rows}):
            overview(root, scene, protocol, [r for r in protocol_rows if r["scene"] == scene])
        all_rows.extend(protocol_rows)
    flat = []
    for row in all_rows:
        for method in ["da3", "shared"]:
            flat.append({
                "scene": row["scene"], "protocol": row["protocol"], "target": row["id"], "method": method,
                "source_depth_log_error": row["source_depth"][method]["log_error"],
                "target_sparse_count": row["target_sparse"][method]["count"],
                "target_sparse_coverage": row["target_sparse"][method]["coverage"],
                "target_sparse_log_error": row["target_sparse"][method]["log_error"],
                "target_sparse_abs_relative": row["target_sparse"][method]["abs_relative"],
                "target_alpha_coverage": row["target_alpha_coverage"][method],
            })
    with (root / f"metrics_{args.dataset}.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat[0]))
        writer.writeheader(); writer.writerows(flat)
    dataset_metadata.update(status="complete", groups=len(all_rows), elapsed_seconds=time.time() - started)
    (root / f"metadata_{args.dataset}.json").write_text(json.dumps(dataset_metadata, indent=2) + "\n")
    print("COMPLETE", args.dataset, len(all_rows), "depth comparisons", flush=True)


if __name__ == "__main__":
    main()
