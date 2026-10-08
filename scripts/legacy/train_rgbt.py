#!/usr/bin/env python3
"""Legacy three-stage surface training. Use train_rgbt_joint.py for the shared module."""

import argparse
import json
import random
import shutil
import sys
import time
from dataclasses import fields, replace
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np
import torch

from depth_anything_3.rgbt.checkpoint import load_rgbt_model, save_checkpoint
from depth_anything_3.rgbt.engine import (
    batch_to,
    depth_loss,
    image_loss,
    psnr,
    ssim,
)
from depth_anything_3.rgbt.geometry import gaussian_depth, multiview_depth_loss
from depth_anything_3.rgbt.renderer import render_rgbt
from depth_anything_3.rgbt.surface_consensus import surface_metrics


@lru_cache(maxsize=16)
def read_cache(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def prepare(path, device, associations=None):
    c = read_cache(str(path))
    o = c["output"]
    gs = replace(
        o.gaussians,
        **{
            f.name: getattr(o.gaussians, f.name).to(device) for f in fields(o.gaussians)
        },
    )
    o = replace(
        o,
        gaussians=gs,
        **{
            f.name: getattr(o, f.name).to(device)
            for f in fields(o)
            if isinstance(getattr(o, f.name), torch.Tensor)
        },
    )
    assoc = {
        k: (x.to(device) if isinstance(x, torch.Tensor) else x)
        for k, x in c["association"].items()
    }
    if (
        associations
        and (Path(associations) / Path(path).parent.name / Path(path).name).is_file()
    ):
        override = read_cache(
            str(Path(associations) / Path(path).parent.name / Path(path).name)
        )
        assoc.update(
            {
                k: (x.to(device) if isinstance(x, torch.Tensor) else x)
                for k, x in override.items()
            }
        )
    return batch_to(c["item"], device), o, assoc


def run(model, b, base, assoc):
    v = base.depth.shape[1]
    return model(
        *[b[k][:, :v] for k in ["rgb", "thermal", "E", "K"]],
        source_valid=b["valid"][:, :v],
        initial=base,
        association=assoc,
    )


@torch.no_grad()
def evaluate(model, paths, device, perceptual, associations=None):
    model.eval()
    rows = []
    for path in paths:
        b, base, assoc = prepare(path, device, associations)
        v = base.depth.shape[1]
        h, w = base.depth.shape[-2:]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            o = run(model, b, base, assoc)
        row = {"id": b["id"], "scene": b["scene"], "views": v}
        for name, output in [("prior", base), ("model", o)]:
            pred = render_rgbt(output, b["E"][:, v:], b["K"][:, v:], h, w)
            mask = b["valid"][:, v:]
            row[name] = {
                m + "_psnr": psnr(pred[m].clamp(0, 1), b[m][:, v:], mask)
                for m in ["rgb", "thermal"]
            }
            row[name].update(
                {
                    m + "_ssim": float(
                        ssim(pred[m].clamp(0, 1) * mask, b[m][:, v:] * mask)
                    )
                    for m in ["rgb", "thermal"]
                }
            )
            row[name]["depth"] = float(
                depth_loss(gaussian_depth(output), b["depth"][:, :v])
            )
            row[name]["lpips"] = float(
                perceptual(
                    (pred["rgb"].clamp(0, 1) * mask).flatten(0, 1),
                    (b["rgb"][:, v:] * mask).flatten(0, 1),
                    normalize=True,
                ).mean()
            )
            row[name]["surface"] = surface_metrics(output, assoc)
            if (model.last_association["ids"] >= 0).any():
                row[name]["supported_surface"] = surface_metrics(
                    output, model.last_association
                )
        rows.append(row)
    per_scene = {
        str(v): {
            name: {
                scene: {
                    k: float(
                        np.mean(
                            [
                                r[name][k]
                                for r in rows
                                if r["views"] == v and r["scene"] == scene
                            ]
                        )
                    )
                    for k in [
                        "rgb_psnr",
                        "thermal_psnr",
                        "rgb_ssim",
                        "thermal_ssim",
                        "depth",
                        "lpips",
                    ]
                }
                for scene in sorted({r["scene"] for r in rows if r["views"] == v})
            }
            for name in ["prior", "model"]
        }
        for v in sorted({r["views"] for r in rows})
    }
    summary = {
        protocol: {
            name: {
                k: float(np.mean([s[k] for s in scenes.values()]))
                for k in next(iter(scenes.values()))
            }
            for name, scenes in versions.items()
        }
        for protocol, versions in per_scene.items()
    }
    return {
        "summary": summary,
        "per_scene": per_scene,
        "per_view": rows,
        "aggregation": "Equal scene weights within each input-view protocol",
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    p.add_argument("--device", required=True)
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument(
        "--weights", default="outputs/rgbt/joint_training_20260920/before/model.pt"
    )
    p.add_argument("--seed", type=int, default=20260920)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument(
        "--smoke",
        action="store_true",
        help="One real gradient update, without changing the input checkpoint",
    )
    p.add_argument("--resume", help="Unified last.pt with optimizer and RNG state")
    p.add_argument("--associations", default="outputs/rgbt/cache/anchors")
    p.add_argument(
        "--train-cache",
        nargs="+",
        default=["outputs/rgbt/cache/train_seed", "outputs/rgbt/cache/train_extended"],
    )
    p.add_argument(
        "--val-cache",
        nargs="+",
        default=[
            "outputs/rgbt/cache/validation_main",
            "outputs/rgbt/cache/validation_external",
        ],
    )
    p.add_argument("--thermal-weight", type=float, default=4.0)
    p.add_argument("--thermal-structure", type=float, default=3.0)
    p.add_argument("--perceptual-weight", type=float, default=0.5)
    p.add_argument("--edge-weight", type=float, default=0.2)
    p.add_argument("--depth-weight", type=float, default=3.0)
    p.add_argument("--multiview-weight", type=float, default=0.05)
    a = p.parse_args()
    torch.set_num_threads(4)
    torch.cuda.set_device(a.device)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    random.seed(a.seed)
    resume = None
    if a.resume:
        resume = torch.load(a.resume, map_location="cpu", weights_only=False, mmap=True)
        if resume.get("optimizer") is None:
            raise ValueError("Resume requires last.pt with optimizer and RNG state")
    out = Path(a.output)
    if (out / "config.json").exists() and not a.resume:
        raise ValueError("Fresh output directory required")
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(a), indent=2))
    (out / "source").mkdir(exist_ok=True)
    source = ROOT / "src/depth_anything_3/rgbt"
    for file in source.rglob("*.py"):
        destination = out / "source" / file.relative_to(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(file, destination)
    shutil.copy2(__file__, out / "source" / Path(__file__).name)
    paths = [p for d in a.train_cache for p in sorted(Path(d).glob("*.pt"))]
    if not paths:
        raise ValueError("Empty training cache")
    byscene = {}
    for path in paths:
        scene = path.name.split("_", 1)[0]
        byscene.setdefault(scene, []).append(path)
    val = [p for d in a.val_cache for p in sorted(Path(d).glob("*.pt"))]
    if not val and not a.smoke:
        raise ValueError("Empty validation cache")
    model, checkpoint = load_rgbt_model(a.resume or a.weights, a.device)
    if model.learned_joint:
        raise ValueError(
            "Shared refiners use scripts/train_rgbt_joint.py --config configs/rgbt_joint_train.json"
        )
    surface_config = model.config["surface"]
    del checkpoint
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.01
    )
    import lpips

    perceptual = lpips.LPIPS(net="alex").to(a.device).eval().requires_grad_(False)
    best = -1e10
    step0 = 0
    if resume:
        optimizer.load_state_dict(resume["optimizer"])
        step0 = resume["step"]
        best = resume.get("best_score", best)
        random.setstate(resume["python_rng"])
        np.random.set_state(resume["numpy_rng"])
        torch.set_rng_state(resume["torch_rng"])
        torch.cuda.set_rng_state(resume["cuda_rng"], a.device)
    start = time.time()
    for step in range(step0 + 1, a.steps + 1):
        model.train()
        path = random.choice(byscene[random.choice(sorted(byscene))])
        b, base, assoc = prepare(path, a.device, a.associations)
        v = base.depth.shape[1]
        h, w = base.depth.shape[-2:]
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            o = run(model, b, base, assoc)
        pred = render_rgbt(o, b["E"][:, v:], b["K"][:, v:], h, w)
        mask = b["valid"][:, v:]
        terms = {}
        with torch.no_grad():
            teacher = render_rgbt(base, b["E"][:, v:], b["K"][:, v:], h, w)
        for m in ["rgb", "thermal"]:
            terms[m] = 2 * ((pred[m] - b[m][:, v:]).square() * mask).sum() / (
                mask.sum() * pred[m].shape[2]
            ) + 0.3 * image_loss(pred[m], b[m][:, v:], mask)
        terms["edges"] = sum(
            (
                (
                    pred[m][..., 1:, :]
                    - pred[m][..., :-1, :]
                    - b[m][:, v:, ..., 1:, :]
                    + b[m][:, v:, ..., :-1, :]
                ).abs()
                * mask[..., 1:, :]
                * mask[..., :-1, :]
            ).mean()
            + (
                (
                    pred[m][..., :, 1:]
                    - pred[m][..., :, :-1]
                    - b[m][:, v:, ..., :, 1:]
                    + b[m][:, v:, ..., :, :-1]
                ).abs()
                * mask[..., :, 1:]
                * mask[..., :, :-1]
            ).mean()
            for m in ["rgb", "thermal"]
        )
        terms["multiview"] = (
            multiview_depth_loss(o, b["valid"][:, :v], stride=8)
            if surface_config["dense_depth_limit"] or surface_config["learn_planes"]
            else pred["rgb"].sum() * 0
        )
        terms["depth"] = depth_loss(gaussian_depth(o), b["depth"][:, :v])
        terms["perceptual"] = perceptual(
            (pred["rgb"].clamp(0, 1) * mask).flatten(0, 1),
            (b["rgb"][:, v:] * mask).flatten(0, 1),
            normalize=True,
        ).mean()
        terms["teacher"] = sum(
            (
                (
                    (pred[m] - b[m][:, v:]).abs().mean(2, keepdim=True)
                    - (teacher[m] - b[m][:, v:]).abs().mean(2, keepdim=True)
                    - 0.003
                ).clamp_min(0)
                * mask
            ).sum()
            / mask.sum()
            for m in ["rgb", "thermal"]
        )
        terms["coverage"] = ((0.98 - pred["alpha"]).clamp_min(0) * mask).mean()
        terms["thermal_structure"] = 1 - ssim(
            pred["thermal"] * mask, b["thermal"][:, v:] * mask
        )
        si = (step - 1) % v
        src = render_rgbt(o, b["E"][:, si : si + 1], b["K"][:, si : si + 1], h, w)
        terms["source"] = sum(
            image_loss(src[m], b[m][:, si : si + 1], b["valid"][:, si : si + 1])
            for m in ["rgb", "thermal"]
        )
        loss = (
            terms["rgb"]
            + a.thermal_weight * terms["thermal"]
            + a.thermal_structure * terms["thermal_structure"]
            + a.depth_weight * terms["depth"]
            + a.perceptual_weight * terms["perceptual"]
            + a.edge_weight * terms["edges"]
            + a.multiview_weight * terms["multiview"]
            + 0.3 * terms["teacher"]
            + 0.08 * terms["coverage"]
            + 0.1 * terms["source"]
            + 0.001 * o.regularization
        )
        if not torch.isfinite(loss):
            raise FloatingPointError(terms)
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0
        )
        if not torch.isfinite(grad):
            raise FloatingPointError(grad)
        optimizer.step()
        for g in optimizer.param_groups:
            g["lr"] = a.lr * (0.25 + 0.75 * (1 - step / a.steps))
        row = {
            "step": step,
            "scene": b["scene"],
            "views": v,
            "total": float(loss.detach()),
            "terms": {k: float(t.detach()) for k, t in terms.items()},
            "gradient": float(grad),
            "seconds": time.time() - start,
        }
        with (out / "train.jsonl").open("a") as f:
            f.write(json.dumps(row) + "\n")
        if step % 25 == 0 or step == 1:
            print("TRAIN", json.dumps(row), flush=True)
        if a.smoke:
            print("SMOKE PASSED", flush=True)
            break
        if step % a.eval_every == 0 or step == a.steps:
            result = evaluate(model, val, a.device, perceptual, a.associations)
            (out / f"val_{step}.json").write_text(json.dumps(result, indent=2))
            score = float(
                np.mean(
                    [
                        (s["model"]["rgb_psnr"] - s["prior"]["rgb_psnr"])
                        + (s["model"]["thermal_psnr"] - s["prior"]["thermal_psnr"])
                        + 10 * (s["model"]["rgb_ssim"] - s["prior"]["rgb_ssim"])
                        + 15 * (s["model"]["thermal_ssim"] - s["prior"]["thermal_ssim"])
                        - 15 * (s["model"]["lpips"] - s["prior"]["lpips"])
                        - 2 * (s["model"]["depth"] - s["prior"]["depth"])
                        for s in result["summary"].values()
                    ]
                )
            )
            save_checkpoint(
                out / "last.pt",
                model,
                optimizer,
                step=step,
                python_rng=random.getstate(),
                numpy_rng=np.random.get_state(),
                torch_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state(),
                best_score=max(best, score),
            )
            if score > best:
                best = score
                save_checkpoint(out / "best.pt", model, step=step)
            print("VAL", step, json.dumps(result["summary"]), flush=True)


if __name__ == "__main__":
    main()
