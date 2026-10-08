#!/usr/bin/env python3
"""Train a single residual Gaussian decoder using held-out-view image losses."""

import argparse
import json
import random
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from cache_rgbt_joint import move
from PIL import Image, ImageDraw

from depth_anything_3.rgbt.engine import batch_to, image_loss, psnr, ssim
from depth_anything_3.rgbt.native_refiner import NativeGaussianRefiner
from depth_anything_3.rgbt.renderer import render_rgbt


@lru_cache(maxsize=12)
def read(path):
    return torch.load(path, map_location="cpu", weights_only=False, mmap=True)


def gradient_loss(pred, truth, mask):
    loss = pred.new_zeros(())
    for stride in [1, 2, 4]:
        for dim in [-2, -1]:
            n = pred.shape[dim] - stride
            p0, p1 = pred.narrow(dim, 0, n), pred.narrow(dim, stride, n)
            t0, t1 = truth.narrow(dim, 0, n), truth.narrow(dim, stride, n)
            m = mask.narrow(dim, 0, n) * mask.narrow(dim, stride, n)
            loss = loss + (((p1 - p0) - (t1 - t0)).abs() * m).sum() / (
                m.sum() * pred.shape[2]
            ).clamp_min(1)
    return loss / 6


def lpips(perceptual, pred, truth, mask):
    x, y = (pred * mask).flatten(0, 1), (truth * mask).flatten(0, 1)
    if x.shape[1] == 1:
        x, y = x.repeat(1, 3, 1, 1), y.repeat(1, 3, 1, 1)
    return perceptual(x, y, normalize=True).mean()


def predict(model, data, device):
    b = batch_to(data["item"], device)
    initial = move(data["initial"], device)
    v = initial.depth.shape[1]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = model(
            *[b[k][:, :v] for k in ["rgb", "thermal", "E", "K"]],
            initial=initial,
            source_valid=b["valid"][:, :v],
        )
    rendered = render_rgbt(output, b["E"][:, v:], b["K"][:, v:], *b["rgb"].shape[-2:])
    return b, output, rendered


def panel(path, b, methods):
    h, w = b["rgb"].shape[-2:]
    canvas = Image.new("RGB", (4 * w, 2 * (h + 24)), "white")
    draw = ImageDraw.Draw(canvas)
    for row, mod in enumerate(["rgb", "thermal"]):
        for col, (label, r) in enumerate([("GT", None), *methods.items()]):
            t = b[mod][0, -1] if r is None else r[mod][0, 0]
            t = t.detach().float().cpu().clamp(0, 1)
            if t.shape[0] == 1:
                t = t.repeat(3, 1, 1)
            im = Image.fromarray(
                (t.permute(1, 2, 0).numpy() * 255 + 0.5).astype("uint8")
            )
            draw.text(
                (col * w + 4, row * (h + 24) + 5), f"{mod}: {label}", fill="black"
            )
            canvas.paste(im, (col * w, row * (h + 24) + 24))
    canvas.save(path)


@torch.no_grad()
def evaluate(model, paths, perceptual, device, dest):
    model.eval()
    dest.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in paths:
        d = read(str(path))
        b, o, pred = predict(model, d, device)
        v = o.depth.shape[1]
        mask = b["valid"][:, v:]
        methods = {
            "native": move(d["baseline"], device),
            "previous": move(d["previous"], device),
            "single": pred,
        }
        row = {"id": d["item"]["id"], "scene": d["item"]["scene"], "views": v}
        for name, r in methods.items():
            metrics = {}
            for mod in ["rgb", "thermal"]:
                x, y = r[mod].clamp(0, 1), b[mod][:, v:]
                metrics[mod + "_psnr"] = psnr(x, y, mask)
                metrics[mod + "_ssim"] = float(ssim(x * mask, y * mask))
                metrics[mod + "_lpips"] = float(lpips(perceptual, x, y, mask))
                metrics[mod + "_edge"] = float(gradient_loss(x, y, mask))
            for threshold in [0.01, 0.5, 0.9]:
                metrics[f"holes_{threshold}"] = float(
                    ((r["alpha"] < threshold) * mask).sum() / mask.sum()
                )
            row[name] = metrics
        rows.append(row)
        panel(dest / (path.stem + ".png"), b, methods)
    keys = list(rows[0]["single"])
    # Scene and view-count balanced, with validation only used for selection.
    groups = sorted({(r["scene"], r["views"]) for r in rows})
    summary = {
        name: {
            k: float(
                np.mean(
                    [
                        np.mean(
                            [r[name][k] for r in rows if (r["scene"], r["views"]) == g]
                        )
                        for g in groups
                    ]
                )
            )
            for k in keys
        }
        for name in ["native", "previous", "single"]
    }
    (dest / "metrics.json").write_text(
        json.dumps({"summary": summary, "rows": rows}, indent=2)
    )
    score = sum(
        summary["single"][mod + "_lpips"] + summary["single"][mod + "_edge"]
        for mod in ["rgb", "thermal"]
    )
    score += 0.2 * summary["single"]["holes_0.5"]
    return score, summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-cache", type=Path, required=True)
    p.add_argument("--val-cache", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--device", default="cuda:1")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--resume", type=Path)
    p.add_argument("--bounds", type=Path)
    p.add_argument("--rgb-weight", type=float, default=1.0)
    p.add_argument("--preservation-weight", type=float, default=0.0)
    p.add_argument("--eval-only", action="store_true")
    a = p.parse_args()
    torch.set_num_threads(4)
    torch.cuda.set_device(a.device)
    torch.manual_seed(20260922)
    random.seed(20260922)
    a.output.mkdir(parents=True, exist_ok=True)
    (a.output / "config.json").write_text(
        json.dumps(
            {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()},
            indent=2,
        )
    )
    import lpips as lpips_module

    perceptual = (
        lpips_module.LPIPS(net="alex").to(a.device).eval().requires_grad_(False)
    )
    seed = torch.load(
        a.resume or a.train_cache / "seed.pt", map_location="cpu", weights_only=False
    )
    if a.bounds:
        seed["config"].update(json.loads(a.bounds.read_text()))
    model = NativeGaussianRefiner(**seed["config"]).to(a.device)
    model.load_state_dict(seed["state_dict"])
    val_paths = [
        a.val_cache / r["file"]
        for r in json.loads((a.val_cache / "index.json").read_text())
    ]
    if a.eval_only:
        print(
            evaluate(model, val_paths, perceptual, a.device, a.output / "eval"),
            flush=True,
        )
        return
    train_index = json.loads((a.train_cache / "index.json").read_text())
    scenes = sorted({r["scene"] for r in train_index})
    val_scenes = {
        r["scene"] for r in json.loads((a.val_cache / "index.json").read_text())
    }
    assert not set(scenes) & val_scenes
    groups = {
        (scene, v): [
            a.train_cache / r["file"]
            for r in train_index
            if r["scene"] == scene and r["views"] == v
        ]
        for scene in scenes
        for v in [3, 6]
    }
    optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    if a.resume and "optimizer" in seed:
        optimizer.load_state_dict(seed["optimizer"])
    best = float("inf")
    started = time.time()
    for step in range(1, a.steps + 1):
        model.train()
        d = read(str(random.choice(groups[random.choice(list(groups))])))
        assert d["split"] == "train"
        optimizer.zero_grad(set_to_none=True)
        b, o, r = predict(model, d, a.device)
        v = o.depth.shape[1]
        mask = b["valid"][:, v:]
        weights = {"rgb": a.rgb_weight, "thermal": 1.0}
        image = sum(
            weights[mod] * image_loss(r[mod], b[mod][:, v:], mask) for mod in weights
        )
        edge = sum(
            weights[mod] * gradient_loss(r[mod], b[mod][:, v:], mask) for mod in weights
        )
        percept = sum(
            weights[mod] * lpips(perceptual, r[mod], b[mod][:, v:], mask)
            for mod in weights
        )
        coverage = (
            -r["alpha"].clamp_min(0.001).log() * mask
        ).sum() / mask.sum().clamp_min(1)
        loss = (
            image
            + 0.3 * percept
            + 0.5 * edge
            + 0.03 * coverage
            + 0.002 * o.regularization
        )
        if a.preservation_weight:
            baseline = move(d["baseline"], a.device)["rgb"]
            truth = b["rgb"][:, v:]
            errors = ((r["rgb"] - truth).abs() - (baseline - truth).abs()) * mask
            preservation = F.avg_pool2d(errors.flatten(0, 1), 8, 4).relu().mean()
            loss = loss + a.preservation_weight * preservation
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite training loss")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), 1.0, error_if_nonfinite=True
        )
        optimizer.step()
        for group in optimizer.param_groups:
            group["lr"] = a.lr * (0.2 + 0.8 * (1 - step / a.steps))
        if step % 10 == 0:
            record = {
                "step": step,
                "loss": float(loss.detach()),
                "image": float(image.detach()),
                "edge": float(edge.detach()),
                "lpips": float(percept.detach()),
                "coverage": float(coverage.detach()),
                "grad": float(norm),
                "seconds": time.time() - started,
            }
            print(json.dumps(record), flush=True)
            with (a.output / "train.jsonl").open("a") as f:
                f.write(json.dumps(record) + "\n")
        if step % a.eval_every == 0 or step == a.steps:
            score, summary = evaluate(
                model, val_paths, perceptual, a.device, a.output / f"val_{step:04d}"
            )
            payload = {
                "config": model.config,
                "state_dict": {
                    k: v.detach().cpu() for k, v in model.state_dict().items()
                },
                "optimizer": optimizer.state_dict(),
                "step": step,
                "score": score,
                "summary": summary,
            }
            torch.save(payload, a.output / "last.pt")
            torch.save(payload, a.output / f"step_{step:04d}.pt")
            if score < best:
                best = score
                torch.save(payload, a.output / "best.pt")
            print("VALIDATION", step, score, json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
