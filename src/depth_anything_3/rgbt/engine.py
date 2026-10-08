"""Losses, deterministic validation and compact trainable-weight checkpoints."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from depth_anything_3.rgbt.geometry import gaussian_depth
from depth_anything_3.rgbt.renderer import render_rgbt


def ssim(x, y):
    x, y = x.flatten(0, 1), y.flatten(0, 1)
    mx, my = F.avg_pool2d(x, 7, 1, 3), F.avg_pool2d(y, 7, 1, 3)
    vx = F.avg_pool2d(x * x, 7, 1, 3) - mx * mx
    vy = F.avg_pool2d(y * y, 7, 1, 3) - my * my
    cov = F.avg_pool2d(x * y, 7, 1, 3) - mx * my
    return (
        ((2 * mx * my + 0.01**2) * (2 * cov + 0.03**2))
        / ((mx * mx + my * my + 0.01**2) * (vx + vy + 0.03**2))
    ).mean()


def image_loss(pred, target, mask):
    l1 = ((pred - target).abs() * mask).sum() / (mask.sum() * pred.shape[2]).clamp_min(
        1
    )
    return 0.8 * l1 + 0.2 * (1 - ssim(pred * mask, target * mask))


def depth_loss(pred, target, relative=False):
    mask = (target > 0) & torch.isfinite(target) & torch.isfinite(pred) & (pred > 0)
    if mask.sum() < 16:
        return pred.sum() * 0
    diff = pred[mask].log() - target[mask].log()
    if relative:
        diff = diff - diff.median().detach()
    return diff.clamp(-3, 3).abs().mean()


def batch_to(batch, device):
    return {
        k: (v[None].to(device) if isinstance(v, torch.Tensor) else v)
        for k, v in batch.items()
    }


def forward_batch(model, batch, views):
    return model(
        batch["rgb"][:, :views],
        batch["thermal"][:, :views],
        batch["E"][:, :views],
        batch["K"][:, :views],
        source_valid=batch["valid"][:, :views],
    )


def batch_loss(model, batch, views, depth_weight=0.03):
    o = forward_batch(model, batch, views)
    h, w = batch["rgb"].shape[-2:]
    p = render_rgbt(o, batch["E"][:, views:], batch["K"][:, views:], h, w)
    mask = batch["valid"][:, views:]
    rgb = image_loss(p["rgb"], batch["rgb"][:, views:], mask)
    thermal = image_loss(p["thermal"], batch["thermal"][:, views:], mask)
    depth = depth_loss(o.depth, batch["depth"][:, :views])
    terms = {
        "rgb": rgb,
        "thermal": thermal,
        "depth": depth,
        "regularization": o.regularization,
    }
    terms["total"] = rgb + thermal + depth_weight * depth + 0.001 * o.regularization
    return terms, o, p


def psnr(pred, target, mask):
    mse = ((pred - target).square() * mask).sum() / (
        mask.sum() * pred.shape[-3]
    ).clamp_min(1)
    return float(-10 * mse.clamp_min(1e-10).log10())


def panel(
    path,
    batch,
    pred,
    baseline=None,
    baseline_label="before training",
    prediction_label="prediction",
):
    def image(t):
        a = t.detach().float().cpu().clamp(0, 1).numpy()
        if a.ndim == 3:
            a = a.transpose(1, 2, 0)
        if a.shape[-1] == 1:
            a = np.repeat(a, 3, -1)
        return Image.fromarray((a * 255 + 0.5).astype(np.uint8))

    items = [
        ("RGB target (unseen)", batch["rgb"][0, -1]),
        (f"RGB {prediction_label}", pred["rgb"][0, 0]),
        ("Thermal target (unseen)", batch["thermal"][0, -1]),
        (f"Thermal {prediction_label}", pred["thermal"][0, 0]),
    ]
    if baseline is not None:
        items = [
            items[0],
            (f"RGB {baseline_label}", baseline["rgb"][0, 0]),
            items[1],
            items[2],
            (f"Thermal {baseline_label}", baseline["thermal"][0, 0]),
            items[3],
        ]
    ncols = len(items) // 2
    h, w = items[0][1].shape[-2:]
    out = Image.new("RGB", (ncols * w, 2 * (h + 26)), (18, 21, 30))
    draw = ImageDraw.Draw(out)
    for i, (name, t) in enumerate(items):
        x, y = (i % ncols) * w, (i // ncols) * (h + 26)
        draw.text((x + 5, y + 6), name, fill="white")
        out.paste(image(t), (x, y + 26))
    out.save(path)


@torch.no_grad()
def evaluate(model, dataset, device, output=None):
    model.eval()
    rows = []
    if output:
        Path(output).mkdir(parents=True, exist_ok=True)
    for i in range(len(dataset)):
        item = dataset[i]
        b = batch_to(item, device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            o = forward_batch(model, b, dataset.views)
        h, w = b["rgb"].shape[-2:]
        p = render_rgbt(o, b["E"][:, dataset.views :], b["K"][:, dataset.views :], h, w)
        mask = b["valid"][:, -1:]
        row = {
            "id": item["id"],
            "scene": item["scene"],
            "context_ids": item["context_ids"],
        }
        for mod in ["rgb", "thermal"]:
            pred = p[mod].clamp(0, 1)
            target = b[mod][:, -1:]
            row[f"{mod}_psnr"] = psnr(pred, target, mask)
            row[f"{mod}_ssim"] = float(ssim(pred * mask, target * mask))
        row["source_depth_log_error"] = float(
            depth_loss(o.depth, b["depth"][:, : dataset.views])
        )
        final_z = gaussian_depth(o)
        row["gaussian_depth_log_error"] = float(
            depth_loss(final_z, b["depth"][:, : dataset.views])
        )
        row["gaussian_depth_ratio"] = float(
            (final_z / o.depth.clamp_min(1e-6)).median()
        )
        rows.append(row)
        if output:
            panel(Path(output) / f"{item['id']}.jpg", b, p)
            np.savez_compressed(
                Path(output) / f"{item['id']}.npz",
                rgb=p["rgb"].float().cpu().numpy(),
                thermal=p["thermal"].float().cpu().numpy(),
                depth=p["depth"].cpu().numpy(),
                alpha=p["alpha"].cpu().numpy(),
            )
    keys = [
        "rgb_psnr",
        "thermal_psnr",
        "rgb_ssim",
        "thermal_ssim",
        "source_depth_log_error",
        "gaussian_depth_log_error",
        "gaussian_depth_ratio",
    ]
    byscene = {
        scene: {
            k: float(np.mean([r[k] for r in rows if r["scene"] == scene])) for k in keys
        }
        for scene in sorted({r["scene"] for r in rows})
    }
    summary = {k: float(np.mean([r[k] for r in byscene.values()])) for k in keys}
    result = {
        "summary": summary,
        "per_scene": byscene,
        "per_view": rows,
        "count": len(rows),
        "selection_score": (summary["rgb_psnr"] + summary["thermal_psnr"]) / 2,
    }
    if output:
        (Path(output) / "metrics.json").write_text(json.dumps(result, indent=2))
    return result
