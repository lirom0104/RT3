#!/usr/bin/env python3
"""Train the gated thermal-edge residual on cached DA3 reconstructions."""

import argparse
import json
import random
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from cache_rgbt_joint import move
from depth_anything_3.rgbt.engine import batch_to, depth_loss, image_loss, psnr, ssim
from depth_anything_3.rgbt.geometry import (
    gaussian_depth,
    multiview_depth_loss,
    thermal_edge_confidence,
)
from depth_anything_3.rgbt.learned_joint import LearnedJointRefiner
from depth_anything_3.rgbt.renderer import render_rgbt


@lru_cache(maxsize=24)
def read(path):
    return torch.load(path, map_location="cpu", weights_only=False, mmap=True)


def gradient_loss(pred, target, mask):
    total = pred.new_zeros(())
    for stride in (1, 2, 4):
        for dim in (-2, -1):
            n = pred.shape[dim] - stride
            p0, p1 = pred.narrow(dim, 0, n), pred.narrow(dim, stride, n)
            t0, t1 = target.narrow(dim, 0, n), target.narrow(dim, stride, n)
            m = mask.narrow(dim, 0, n) * mask.narrow(dim, stride, n)
            total = total + (((p1 - p0) - (t1 - t0)).abs() * m).sum() / (
                m.sum() * pred.shape[2]
            ).clamp_min(1)
    return total / 6


def make_model(seed, device, edge_limit):
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
    if unexpected or any(not name.startswith("core.thermal_edge_") for name in missing):
        raise RuntimeError(f"Unexpected warm-start mismatch: missing={missing}, unexpected={unexpected}")
    model.to(device)
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith("core.thermal_edge_"))
    return model


def run(model, path, device, multiview_weight=0.5, train=True):
    data = read(str(path))
    b = batch_to(data["item"], device)
    initial = move(data["initial"], device)
    views = initial.depth.shape[1]
    output = model(
        b["rgb"][:, :views],
        b["thermal"][:, :views],
        b["E"][:, :views],
        b["K"][:, :views],
        initial=initial,
        source_valid=b["valid"][:, :views],
    )
    pred = render_rgbt(
        output,
        b["E"][:, views:],
        b["K"][:, views:],
        *b["rgb"].shape[-2:],
    )
    mask = b["valid"][:, views:]
    rgb_loss = image_loss(pred["rgb"], b["rgb"][:, views:], mask)
    thermal_loss = image_loss(pred["thermal"], b["thermal"][:, views:], mask)
    edge = gradient_loss(pred["thermal"], b["thermal"][:, views:], mask)
    final_depth = gaussian_depth(output)
    target_depth = b["depth"][:, :views]
    depth = depth_loss(final_depth, target_depth)
    edge_confidence = thermal_edge_confidence(
        b["rgb"][:, :views], b["thermal"][:, :views]
    )[:, :, 0]
    depth_valid = (
        (target_depth > 0)
        & torch.isfinite(target_depth)
        & torch.isfinite(final_depth)
        & (final_depth > 0)
    )
    edge_weights = 1 + 8 * edge_confidence
    edge_depth = (
        (final_depth.clamp_min(1e-6).log() - target_depth.clamp_min(1e-6).log())
        .clamp(-3, 3)
        .abs()
    )
    edge_depth = (edge_depth * edge_weights * depth_valid).sum() / (
        edge_weights * depth_valid
    ).sum().clamp_min(1)
    multiview = multiview_depth_loss(output, b["valid"][:, :views], stride=4)
    loss = (
        rgb_loss
        + 1.5 * thermal_loss
        + 0.35 * edge
        + 0.10 * depth
        + 0.50 * edge_depth
        + multiview_weight * multiview
    )
    return loss, {
        "rgb_psnr": psnr(pred["rgb"].detach().clamp(0, 1), b["rgb"][:, views:], mask),
        "thermal_psnr": psnr(pred["thermal"].detach().clamp(0, 1), b["thermal"][:, views:], mask),
        "rgb_ssim": float(ssim(pred["rgb"].detach() * mask, b["rgb"][:, views:] * mask)),
        "thermal_ssim": float(ssim(pred["thermal"].detach() * mask, b["thermal"][:, views:] * mask)),
        "depth_log_error": float(depth.detach()),
        "thermal_edge_depth_error": float(edge_depth.detach()),
        "edge_loss": float(edge.detach()),
        "multiview_depth_error": float(multiview.detach()),
    }


@torch.no_grad()
def evaluate(model, paths, device, multiview_weight=0.5):
    model.eval()
    rows = []
    for path in paths:
        _, row = run(model, path, device, multiview_weight=multiview_weight, train=False)
        rows.append(row)
    keys = rows[0].keys()
    return {key: float(np.mean([row[key] for row in rows])) for key in keys}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-cache", required=True, type=Path)
    parser.add_argument("--val-cache", required=True, type=Path)
    parser.add_argument("--seed", default="checkpoints/RGBT/training_state.pt")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--edge-limit", type=float, default=0.25)
    parser.add_argument("--multiview-weight", type=float, default=0.5)
    parser.add_argument("--seed-number", type=int, default=20260930)
    args = parser.parse_args()
    torch.set_num_threads(4)
    cv2.setNumThreads(2)
    torch.cuda.set_device(args.device)
    torch.manual_seed(args.seed_number)
    np.random.seed(args.seed_number)
    random.seed(args.seed_number)
    train_paths = [
        p for p in sorted(args.train_cache.glob("*.pt")) if "item" in read(str(p))
    ]
    val_paths = [
        p for p in sorted(args.val_cache.glob("*.pt")) if "item" in read(str(p))
    ]
    if not train_paths or not val_paths:
        raise ValueError("Training and validation caches must contain .pt files")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "config.json").write_text(json.dumps(vars(args), default=str, indent=2) + "\n")
    model = make_model(args.seed, args.device, args.edge_limit)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=1e-4
    )
    baseline = evaluate(model, val_paths, args.device, args.multiview_weight)
    (args.output / "val_0.json").write_text(json.dumps(baseline, indent=2) + "\n")
    initial_checkpoint = {
        "refiner_config": model.config,
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "step": 0,
    }
    torch.save(initial_checkpoint, args.output / "step_0.pt")
    print("VAL 0", json.dumps(baseline), flush=True)
    def score(metrics):
        return (
            metrics["rgb_psnr"]
            + metrics["thermal_psnr"]
            - 10 * metrics["edge_loss"]
            - 10 * metrics["depth_log_error"]
            - 10 * metrics["thermal_edge_depth_error"]
            - 10 * metrics["multiview_depth_error"]
        )

    best = score(baseline)
    torch.save(initial_checkpoint, args.output / "best.pt")
    for step in range(1, args.steps + 1):
        model.train()
        path = random.choice(train_paths)
        optimizer.zero_grad(set_to_none=True)
        loss, row = run(model, path, args.device, args.multiview_weight)
        if not torch.isfinite(loss):
            raise FloatingPointError(row)
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(grad) or grad == 0:
            raise FloatingPointError(f"invalid gradient {grad}")
        optimizer.step()
        if step == 1 or step % 25 == 0:
            print("TRAIN", step, float(loss.detach()), row, flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            result = evaluate(model, val_paths, args.device, args.multiview_weight)
            result["step"] = step
            (args.output / f"val_{step}.json").write_text(json.dumps(result, indent=2) + "\n")
            current_score = score(result)
            torch.save(
                {"refiner_config": model.config, "state_dict": model.state_dict(), "step": step},
                args.output / "last.pt",
            )
            if current_score > best:
                best = current_score
                torch.save(
                    {"refiner_config": model.config, "state_dict": model.state_dict(), "step": step},
                    args.output / "best.pt",
                )
            print("VAL", step, json.dumps(result), "score", current_score, flush=True)


if __name__ == "__main__":
    main()
