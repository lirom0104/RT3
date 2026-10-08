#!/usr/bin/env python3
"""Export test-only PNG comparisons: ground truth, DA3 wrapper, SharedRefiner."""

import argparse
import csv
import hashlib
import importlib.metadata
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from safetensors import safe_open

from ablate_rgbt import VARIANTS, configured
from depth_anything_3.rgbt.checkpoint import load_rgbt_model
from depth_anything_3.rgbt.data import SceneDataset
from depth_anything_3.rgbt.engine import batch_to, psnr, ssim
from depth_anything_3.rgbt.renderer import render_rgbt


FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
BASELINE_NOTE = (
    "DA3 uses original frozen weights with the existing RGB-T wrapper: source-camera "
    "scale alignment and source thermal sampling. Thermal is not a native DA3 output. "
    "Thermal adapters, learned residuals, depth anchoring, RGB source blend, and all "
    "refiners are disabled. This is not the unmodified official inference API."
)
LPIPS_SETTINGS = {
    "backbone": "alex",
    "version": "0.1",
    "pretrained": True,
    "lower_is_better": True,
    "input": "Clipped float predictions and cached float GT in [0,1]; normalize=True",
    "mask": "Same target valid mask for both images; invalid pixels set to zero before normalization; full feature-map mean",
    "thermal": "Grayscale repeated across three channels; display-image perceptual distance, not temperature error",
    "precision": "float32, eval mode, no autocast",
}


def font(size):
    return ImageFont.truetype(FONT_PATH, size)


def bitmap(tensor):
    array = tensor.detach().float().clamp(0, 1).cpu().numpy()
    array = np.moveaxis(array, 0, -1)
    if array.shape[-1] == 1:
        array = array[..., 0]
    return Image.fromarray((array * 255 + 0.5).astype(np.uint8))


def lpips_score(perceptual, pred, target, mask):
    pred = (pred.float().clamp(0, 1) * mask).flatten(0, 1)
    target = (target.float() * mask).flatten(0, 1)
    if pred.shape[1] == 1:
        pred = pred.repeat(1, 3, 1, 1)
        target = target.repeat(1, 3, 1, 1)
    return float(perceptual(pred, target, normalize=True).mean())


def comparison(path, item, batch, renders, metrics, views, raw_dir):
    h, w = batch["rgb"].shape[-2:]
    header, label, footer, gap = 66, 66, 64, 8
    canvas = Image.new(
        "RGB", (3 * w + 4 * gap, header + 2 * (h + label) + gap + footer), "white"
    )
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (gap, 8),
        f"{item['scene']} | TEST | {views} source views",
        fill="#151515",
        font=font(18),
    )
    draw.text((gap, 36), item["id"], fill="#444444", font=font(12))
    for row, modality in enumerate(["rgb", "thermal"]):
        y = header + row * (h + label + gap)
        entries = [
            ("Ground truth", "gt", batch[modality][0, -1]),
            ("DA3 (RGB-T wrapper)", "da3", renders["da3"][modality][0, 0]),
            ("SharedRefiner", "shared", renders["shared"][modality][0, 0]),
        ]
        for col, (name, key, tensor) in enumerate(entries):
            x = gap + col * (w + gap)
            draw.text(
                (x, y), f"{modality.upper()} | {name}", fill="#151515", font=font(13)
            )
            if key != "gt":
                draw.text(
                    (x, y + 21),
                    f"PSNR {metrics[key][modality + '_psnr']:.2f} dB | SSIM {metrics[key][modality + '_ssim']:.3f}",
                    fill="#555555",
                    font=font(12),
                )
                draw.text(
                    (x, y + 40),
                    f"LPIPS {metrics[key][modality + '_lpips']:.4f} (lower is better)",
                    fill="#333333",
                    font=font(12),
                )
            image = bitmap(tensor)
            image.save(raw_dir / f"{modality}_{key}.png")
            canvas.paste(image.convert("RGB"), (x, y + label))
    draw.text(
        (gap, canvas.height - 53),
        "DA3 thermal: source thermal sampled on DA3 geometry; not a native DA3 output.",
        fill="#555555",
        font=font(12),
    )
    draw.text(
        (gap, canvas.height - 37),
        "Same source images, held-out target and COLMAP cameras for both models.",
        fill="#555555",
        font=font(12),
    )
    draw.text(
        (gap, canvas.height - 21),
        "LPIPS: AlexNet v0.1; float images; same valid mask; thermal grayscale repeated to RGB.",
        fill="#555555",
        font=font(12),
    )
    canvas.save(path)


def overview(root, scene, protocol, rows):
    # Evenly spaced target IDs, with no quality-based selection.
    indices = np.linspace(0, len(rows) - 1, min(3, len(rows))).round().astype(int)
    selected = [rows[i] for i in indices]
    images = [Image.open(root / r["comparison"]).convert("RGB") for r in selected]
    canvas = Image.new(
        "RGB", (images[0].width, sum(i.height for i in images) + 44), "#eeeeee"
    )
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (8, 10),
        f"{scene} | {protocol} | evenly spaced {len(selected)}/{len(rows)} targets",
        fill="#151515",
        font=font(16),
    )
    y = 44
    for image in images:
        canvas.paste(image, (0, y))
        y += image.height
        image.close()
    dest = root / "overviews" / f"{scene.replace('/', '_')}_{protocol}.png"
    dest.parent.mkdir(exist_ok=True)
    canvas.save(dest)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", default="checkpoints/RGBT/model.pt")
    parser.add_argument(
        "--original-weights",
        default="/home/lf/feedforward/DA3base/checkpoints/DA3-GIANT-1.1/model.safetensors",
    )
    parser.add_argument("--data", default="outputs/rgbt/data_336")
    parser.add_argument("--reference", default="outputs/rgbt/evaluation_20260922")
    parser.add_argument(
        "--dataset", required=True, choices=["RGBT-Scenes", "ThermoScenes"]
    )
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
    # Check every foundation tensor against the local official checkpoint.
    prefix = "reconstruction.da3."
    foundation = {
        k[len(prefix) :]: v
        for k, v in checkpoint["state_dict"].items()
        if k.startswith(prefix)
    }
    with safe_open(args.original_weights, framework="pt", device="cpu") as original:
        mapping = {k.removeprefix("model."): k for k in original.keys()}
        if not set(mapping).issubset(foundation):
            raise ValueError("Embedded model is missing official DA3 tensors")
        for key, original_key in mapping.items():
            if not torch.equal(foundation[key], original.get_tensor(original_key)):
                raise ValueError(f"Embedded DA3 weight differs: {key}")
        # torch.save retains aliases of the shared auxiliary LayerNorm, whereas
        # safetensors stores only the first copy. Verify all six aliases too.
        aliases = sorted(set(foundation) - set(mapping))
        expected_aliases = {
            f"head.scratch.output_conv2_aux.{level}.2.{kind}"
            for level in [1, 2, 3]
            for kind in ["weight", "bias"]
        }
        if set(aliases) != expected_aliases:
            raise ValueError(f"Unexpected extra foundation tensors: {aliases}")
        for key in aliases:
            original_key = (
                "model.head.scratch.output_conv2_aux.0.2." + key.rsplit(".", 1)[1]
            )
            if not torch.equal(foundation[key], original.get_tensor(original_key)):
                raise ValueError(f"DA3 shared projection alias differs: {key}")
    del foundation, checkpoint
    import lpips

    perceptual = lpips.LPIPS(net="alex", version="0.1").to(args.device).eval()
    perceptual.requires_grad_(False)
    metadata = {
        "dataset": args.dataset,
        "split": "test",
        "views": [3, 6],
        "weights": str(Path(args.weights).resolve()),
        "official_weights": str(Path(args.original_weights).resolve()),
        "official_tensors_verified_equal": len(mapping),
        "shared_projection_aliases_verified_equal": aliases,
        "baseline": BASELINE_NOTE,
        "baseline_switches": VARIANTS["da3_rgbt_wrapper"],
        "data": str(Path(args.data).resolve()),
        "device": args.device,
        "target_selection": "All reserved test targets; no quality selection",
        "source_selection": {
            "nearest3": "3 nearest source cameras",
            "wide6": "6 broader source cameras using existing wide protocol",
        },
        "image_columns": [
            "Ground truth",
            "Original DA3 weights + RGB-T wrapper",
            "SharedRefiner",
        ],
        "image_rows": ["RGB", "Thermal grayscale"],
        "preprocessing": "Same cached undistorted and resized target for both models",
        "cameras": "Known COLMAP cameras",
        "optimization_steps": 0,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "lpips": {
            **LPIPS_SETTINGS,
            "package_version": importlib.metadata.version("lpips"),
        },
    }
    (root / f"metadata_{args.dataset}.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    reference = json.loads(
        (Path(args.reference) / args.dataset / "results.json").read_text()
    )
    all_rows = []
    for views in [3, 6]:
        protocol = "nearest3" if views == 3 else "wide6"
        ref_rows = {
            r["id"]: r
            for r in reference["results"][protocol]["per_view"]
            if r["split"] == "test"
        }
        dataset = SceneDataset(
            args.data,
            "test",
            views=views,
            targets_per_scene=100000,
            sampling="nearest" if views == 3 else "wide",
        )
        dataset.items = [
            (si, ti)
            for si, ti in dataset.items
            if dataset.scenes[si]["domain"] == args.dataset
        ]
        protocol_rows = []
        for i in range(len(dataset)):
            item = dataset[i]
            if (
                item["id"] in item["context_ids"]
                or item["context_ids"] != ref_rows[item["id"]]["context_ids"]
            ):
                raise ValueError(
                    "Source/target pairing differs from previous evaluation"
                )
            batch = batch_to(item, args.device)
            inputs = [batch[k][:, :views] for k in ["rgb", "thermal", "E", "K"]]
            target_E, target_K = batch["E"][:, views:], batch["K"][:, views:]
            h, w = batch["rgb"].shape[-2:]
            # Restore all switched attributes before running the trained model.
            with (
                configured(model, VARIANTS["da3_rgbt_wrapper"]),
                torch.autocast("cuda", dtype=torch.bfloat16),
            ):
                baseline = model.reconstruction(*inputs, teacher=True)
            renders = {"da3": render_rgbt(baseline, target_E, target_K, h, w)}
            del baseline
            with torch.autocast("cuda", dtype=torch.bfloat16):
                shared = model(*inputs, source_valid=batch["valid"][:, :views])
            renders["shared"] = render_rgbt(shared, target_E, target_K, h, w)
            del shared
            mask = batch["valid"][:, views:]
            metrics = {}
            for method, rendered in renders.items():
                metrics[method] = {}
                for modality in ["rgb", "thermal"]:
                    image = rendered[modality].clamp(0, 1)
                    target = batch[modality][:, views:]
                    metrics[method][modality + "_psnr"] = psnr(image, target, mask)
                    metrics[method][modality + "_ssim"] = float(
                        ssim(image * mask, target * mask)
                    )
                    metrics[method][modality + "_lpips"] = lpips_score(
                        perceptual, image, target, mask
                    )
                if not all(np.isfinite(v) for v in metrics[method].values()):
                    raise ValueError(f"Non-finite metrics for {item['id']}")
            differences = {
                key: abs(value - ref_rows[item["id"]][key])
                for key, value in metrics["shared"].items()
                if key.endswith(("_psnr", "_ssim"))
            }
            if max(differences.values()) > 0.002:
                raise ValueError(
                    f"SharedRefiner no longer reproduces prior run: {differences}"
                )
            folder = root / item["scene"] / protocol
            folder.mkdir(parents=True, exist_ok=True)
            raw_dir = folder / "raw" / item["id"]
            raw_dir.mkdir(parents=True, exist_ok=True)
            path = folder / f"{item['id']}.png"
            comparison(path, item, batch, renders, metrics, views, raw_dir)
            row = {
                "id": item["id"],
                "scene": item["scene"],
                "split": "test",
                "dataset": args.dataset,
                "protocol": protocol,
                "views": views,
                "context_ids": item["context_ids"],
                "comparison": path.relative_to(root).as_posix(),
                "raw": raw_dir.relative_to(root).as_posix(),
                "reference_max_metric_difference": max(differences.values()),
                **metrics,
            }
            protocol_rows.append(row)
            with rows_path.open("a") as stream:
                stream.write(json.dumps(row) + "\n")
            print(
                args.dataset,
                protocol,
                f"{i + 1}/{len(dataset)}",
                item["id"],
                json.dumps(metrics),
                flush=True,
            )
        if {r["id"] for r in protocol_rows} != set(ref_rows):
            raise ValueError("Missing test targets")
        for scene in sorted({r["scene"] for r in protocol_rows}):
            overview(
                root, scene, protocol, [r for r in protocol_rows if r["scene"] == scene]
            )
        all_rows.extend(protocol_rows)
    flat_rows = [
        {
            "scene": r["scene"],
            "protocol": r["protocol"],
            "target": r["id"],
            "method": method,
            **r[method],
        }
        for r in all_rows
        for method in ["da3", "shared"]
    ]
    with (root / f"metrics_{args.dataset}.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat_rows[0]))
        writer.writeheader()
        writer.writerows(flat_rows)
    metadata.update(
        status="complete", groups=len(all_rows), elapsed_seconds=time.time() - started
    )
    (root / f"metadata_{args.dataset}.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print("COMPLETE", args.dataset, len(all_rows), "comparisons", flush=True)


if __name__ == "__main__":
    main()
