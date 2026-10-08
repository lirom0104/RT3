#!/usr/bin/env python3
"""Compare a gated thermal-edge refiner with the unchanged shared refiner."""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw

from cache_rgbt_joint import move
from depth_anything_3.rgbt.engine import batch_to, depth_loss, psnr, ssim
from depth_anything_3.rgbt.geometry import gaussian_depth, thermal_edge_confidence
from depth_anything_3.rgbt.learned_joint import LearnedJointRefiner
from depth_anything_3.rgbt.renderer import render_rgbt


def make_model(seed, device, candidate=None, edge_limit=0.25):
    model = LearnedJointRefiner(
        channels=96,
        attention_pairs=2,
        ordered_updates=True,
        phase_modulation=True,
        preserve_first_update=True,
        shared_residual=True,
        thermal_edge=True,
        thermal_edge_channels=16,
        thermal_edge_geometry_limit=edge_limit,
    )
    warm = torch.load(seed, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(warm["state_dict"], strict=False)
    if unexpected or any(not n.startswith("core.thermal_edge_") for n in missing):
        raise RuntimeError(f"Warm-start mismatch: {missing}, {unexpected}")
    if candidate:
        checkpoint = torch.load(candidate, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model.to(device).eval()


def panel(path, data, baseline, candidate, views):
    h, w = data["rgb"].shape[-2:]
    canvas = Image.new("RGB", (4 * w, 2 * (h + 22)), "white")
    draw = ImageDraw.Draw(canvas)
    for row, modality in enumerate(("rgb", "thermal")):
        images = [
            ("target", data[modality][0, views]),
            ("shared", baseline[modality][0, 0]),
            ("edge gate", candidate[modality][0, 0]),
            ("edge residual", (candidate[modality][0, 0] - baseline[modality][0, 0]).abs() * 4),
        ]
        for col, (label, tensor) in enumerate(images):
            value = tensor.detach().float().cpu().clamp(0, 1)
            if value.shape[0] == 1:
                value = value.repeat(3, 1, 1)
            pixels = (value.permute(1, 2, 0).numpy() * 255 + 0.5).astype(np.uint8)
            draw.text((col * w + 4, row * (h + 22) + 4), f"{modality}: {label}", fill="black")
            canvas.paste(Image.fromarray(pixels), (col * w, row * (h + 22) + 22))
    canvas.save(path)


def depth_panel(path, target, baseline, candidate):
    target = target.detach().float().cpu().numpy()
    baseline = baseline.detach().float().cpu().numpy()
    candidate = candidate.detach().float().cpu().numpy()
    valid = np.isfinite(target) & (target > 0)
    scale_values = np.concatenate(
        [baseline[np.isfinite(baseline)], candidate[np.isfinite(candidate)], target[valid]]
    )
    lo, hi = np.percentile(scale_values, [2, 98])
    hi = max(float(hi), float(lo) + 1e-6)

    def render_depth(value, mask=None):
        normalized = np.clip((value - lo) / (hi - lo), 0, 1)
        image = (normalized * 255 + 0.5).astype(np.uint8)
        if mask is not None:
            image[~mask] = 0
        return Image.fromarray(image).convert("RGB")

    difference = np.abs(candidate - baseline)
    diff_scale = max(float(np.percentile(difference, 99)), 1e-6)
    diff_image = Image.fromarray(
        (np.clip(difference / diff_scale, 0, 1) * 255 + 0.5).astype(np.uint8)
    ).convert("RGB")
    images = [
        ("sparse COLMAP", render_depth(target, valid)),
        ("shared Gaussian z", render_depth(baseline)),
        ("Thermal-edge z", render_depth(candidate)),
        ("absolute delta", diff_image),
    ]
    w, h = baseline.shape[1], baseline.shape[0]
    canvas = Image.new("RGB", (4 * w, h + 24), "white")
    draw = ImageDraw.Draw(canvas)
    for column, (label, image) in enumerate(images):
        draw.text((column * w + 4, 4), label, fill="black")
        canvas.paste(image, (column * w, 24))
    canvas.save(path)


@torch.no_grad()
def evaluate(paths, baseline_model, candidate_model, device, output, visualize):
    rows = []
    for path in paths:
        data = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        if "item" not in data:
            continue
        b = batch_to(data["item"], device)
        initial = move(data["initial"], device)
        views = initial.depth.shape[1]
        rgb, thermal, E, K, valid = [
            b[k][:, :views] for k in ["rgb", "thermal", "E", "K", "valid"]
        ]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            baseline = baseline_model(
                rgb,
                thermal,
                E,
                K,
                initial=initial,
                source_valid=valid,
            )
            candidate = candidate_model(
                rgb,
                thermal,
                E,
                K,
                initial=initial,
                source_valid=valid,
            )
        pred_base = render_rgbt(
            baseline, b["E"][:, views:], b["K"][:, views:], *b["rgb"].shape[-2:]
        )
        pred_candidate = render_rgbt(
            candidate, b["E"][:, views:], b["K"][:, views:], *b["rgb"].shape[-2:]
        )
        mask = b["valid"][:, views:]
        target_depth = b["depth"][:, :views]
        edge_confidence = thermal_edge_confidence(rgb, thermal)[:, :, 0]
        edge_threshold = torch.quantile(
            edge_confidence.flatten(-2), 0.75, dim=-1
        )[..., None, None]
        edge_depth_mask = (target_depth > 0) & (edge_confidence >= edge_threshold)
        row = {
            "id": data["item"]["id"],
            "scene": data["item"]["scene"],
            "views": views,
            "baseline": {},
            "candidate": {},
        }
        for name, output_state, rendered in [
            ("baseline", baseline, pred_base),
            ("candidate", candidate, pred_candidate),
        ]:
            metrics = {}
            for modality in ("rgb", "thermal"):
                target = b[modality][:, views:]
                pred = rendered[modality].clamp(0, 1)
                metrics[modality + "_psnr"] = psnr(pred.detach(), target, mask)
                metrics[modality + "_ssim"] = float(ssim(pred * mask, target * mask))
            metrics["depth_log_error"] = float(
                depth_loss(gaussian_depth(output_state), target_depth)
            )
            edge_depth = gaussian_depth(output_state)
            edge_depth_mask_valid = (
                edge_depth_mask
                & torch.isfinite(edge_depth)
                & (edge_depth > 0)
            )
            edge_diff = (
                edge_depth.clamp_min(1e-6).log()
                - target_depth.clamp_min(1e-6).log()
            ).abs()
            if edge_depth_mask_valid.sum() < 16:
                edge_depth_mask_valid = (
                    (target_depth > 0)
                    & (edge_confidence >= torch.quantile(
                        edge_confidence.flatten(-2), 0.50, dim=-1
                    )[..., None, None])
                    & torch.isfinite(edge_depth)
                    & (edge_depth > 0)
                )
            if edge_depth_mask_valid.sum() == 0:
                edge_depth_mask_valid = (
                    (target_depth > 0)
                    & torch.isfinite(edge_depth)
                    & (edge_depth > 0)
                )
            metrics["thermal_edge_depth_error"] = float(
                edge_diff[edge_depth_mask_valid].mean()
            )
            row[name] = metrics
        row["delta"] = {
            key: row["candidate"][key] - row["baseline"][key]
            for key in row["baseline"]
        }
        rows.append(row)
        if data["item"]["id"] in visualize:
            panel(
                output / f"{data['item']['id']}_{views}.png",
                b,
                pred_base,
                pred_candidate,
                views,
            )
            depth_panel(
                output / f"{data['item']['id']}_{views}_depth.png",
                b["depth"][0, 0],
                gaussian_depth(baseline)[0, 0],
                gaussian_depth(candidate)[0, 0],
            )
        print("EVAL", row["scene"], row["id"], views, row["delta"], flush=True)

    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["scene"], row["views"])].append(row)
    group_rows = []
    for (scene, views), selected in sorted(grouped.items()):
        group_rows.append(
            {
                "scene": scene,
                "views": views,
                "count": len(selected),
                **{
                    tag: {
                        key: float(np.mean([row[tag][key] for row in selected]))
                        for key in rows[0][tag]
                    }
                    for tag in ("baseline", "candidate", "delta")
                },
            }
        )
    summary = {
        "count": len(rows),
        "scene_view_groups": len(group_rows),
        **{
            tag: {
                key: float(np.mean([row[tag][key] for row in group_rows]))
                for key in rows[0][tag]
            }
            for tag in ("baseline", "candidate", "delta")
        },
        "worst_group_rgb_psnr_delta": min(
            row["delta"]["rgb_psnr"] for row in group_rows
        ),
        "worst_group_thermal_psnr_delta": min(
            row["delta"]["thermal_psnr"] for row in group_rows
        ),
    }
    report = {"summary": summary, "per_scene_views": group_rows, "per_target": rows}
    (output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    print("SUMMARY", json.dumps(summary), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--seed", default="checkpoints/RGBT/training_state.pt")
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--edge-limit", type=float, default=None)
    parser.add_argument("--ids", nargs="*")
    parser.add_argument("--visualize", nargs="*", default=["RGBT-Scenes_Building_img_025"])
    args = parser.parse_args()
    torch.set_num_threads(4)
    cv2.setNumThreads(2)
    torch.cuda.set_device(args.device)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.edge_limit is None:
        checkpoint = torch.load(args.candidate, map_location="cpu", weights_only=False)
        args.edge_limit = float(checkpoint.get("refiner_config", {}).get(
            "thermal_edge_geometry_limit", 0.25
        ))
    baseline = make_model(args.seed, args.device, edge_limit=args.edge_limit)
    candidate = make_model(
        args.seed, args.device, args.candidate, edge_limit=args.edge_limit
    )
    paths = sorted(args.cache.glob("*.pt"))
    if args.ids:
        paths = [path for path in paths if any(item_id in path.name for item_id in args.ids)]
    evaluate(paths, baseline, candidate, args.device, args.output, set(args.visualize))


if __name__ == "__main__":
    main()
