#!/usr/bin/env python3
"""Paired full-forward evaluation of a shared refiner against the frozen teacher."""

import argparse
import hashlib
import html
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import cv2
import numpy as np
import torch
from ablate_rgbt import cases

from depth_anything_3.rgbt.checkpoint import load_rgbt_model
from depth_anything_3.rgbt.engine import batch_to, depth_loss, panel, psnr, ssim
from depth_anything_3.rgbt.geometry import gaussian_depth
from depth_anything_3.rgbt.learned_joint import LearnedJointRefiner
from depth_anything_3.rgbt.renderer import render_rgbt
from depth_anything_3.rgbt.surface_consensus import surface_metrics


def aggregate(rows, policy):
    tags = ["teacher", "student", "delta"]
    if all("base" in row for row in rows):
        tags += ["base", "delta_base"]
    scenes = []
    for scene, views in sorted({(r["scene"], r["views"]) for r in rows}):
        group = [r for r in rows if r["scene"] == scene and r["views"] == views]
        scenes.append(
            {
                "scene": scene,
                "views": views,
                **{
                    tag: {
                        k: float(np.mean([r[tag][k] for r in group]))
                        for k in group[0][tag]
                    }
                    for tag in tags
                },
            }
        )
    summary = {
        str(v): {
            k: float(np.mean([r["delta"][k] for r in scenes if r["views"] == v]))
            for k in scenes[0]["delta"]
        }
        for v in sorted({r["views"] for r in scenes})
    }
    failures = []
    for level, records, rules in [
        ("scene", scenes, policy["required_scene_protocol_deltas"]),
        ("target", rows, policy["required_per_target_deltas"]),
    ]:
        for row in records:
            for rule, limit in rules.items():
                key, bound = rule.rsplit("_", 1)
                value = row["delta"][key]
                if (bound == "min" and value < limit) or (
                    bound == "max" and value > limit
                ):
                    failures.append(
                        {
                            "level": level,
                            "scene": row["scene"],
                            "views": row["views"],
                            "target": row.get("target"),
                            "metric": key,
                            "delta": value,
                            "limit": limit,
                        }
                    )
    return {
        "summary_delta": summary,
        "summary": {
            str(v): {
                tag: {
                    k: float(np.mean([r[tag][k] for r in scenes if r["views"] == v]))
                    for k in scenes[0][tag]
                }
                for tag in tags
            }
            for v in sorted({r["views"] for r in scenes})
        },
        "per_scene": scenes,
        "failures": failures,
        "passed": not failures,
        "cases": len(rows),
        "scenes": len({r["scene"] for r in rows}),
    }


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--teacher", default="outputs/rgbt/joint_training_20260920/before/model.pt"
    )
    parser.add_argument("--refiner", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--data",
        nargs="+",
        default=[
            "outputs/rgbt/data_336",
            "outputs/rgbt/datasets/external_336",
            "outputs/rgbt/datasets/heldout_336",
            "outputs/rgbt/datasets/plant_336",
        ],
    )
    parser.add_argument("--split", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--views", nargs="+", type=int, default=[3, 6])
    parser.add_argument("--targets", type=int, default=999)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument(
        "--include-base",
        action="store_true",
        help="Also evaluate frozen reconstruction without any of the three refiners",
    )
    args = parser.parse_args()
    torch.set_num_threads(4)
    cv2.setNumThreads(2)
    torch.cuda.set_device(args.device)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if (out / f"results_{args.shard}.json").exists():
        raise ValueError("Use a fresh output directory")
    (out / f"config_{args.shard}.json").write_text(
        json.dumps(vars(args), indent=2) + "\n"
    )
    teacher, checkpoint = load_rgbt_model(args.teacher, args.device)
    del checkpoint
    checkpoint = torch.load(args.refiner, map_location="cpu", weights_only=False)
    # Keep the evaluated weights even when the training run later replaces best.pt.
    snapshot = out / f"refiner_{args.shard}.pt"
    torch.save(
        {
            key: checkpoint[key]
            for key in ["format", "refiner_config", "state_dict", "step"]
        },
        snapshot,
    )
    metadata = {
        **vars(args),
        "evaluated_step": checkpoint["step"],
        "evaluated_refiner_sha256": hashlib.sha256(snapshot.read_bytes()).hexdigest(),
    }
    (out / f"config_{args.shard}.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    student = LearnedJointRefiner(**checkpoint["refiner_config"]).to(args.device).eval()
    student.load_state_dict(checkpoint["state_dict"], strict=True)
    del checkpoint
    import lpips

    perceptual = lpips.LPIPS(net="alex").to(args.device).eval().requires_grad_(False)
    policy = json.loads(
        (ROOT / "outputs/rgbt/joint_training_20260920/quality_policy.json").read_text()
    )
    rows, cards = [], []
    start = time.time()
    for number, (root, split, views, dataset, index, scene) in enumerate(
        cases(args.data, args.targets, args.views, args.split)
    ):
        if number % args.shards != args.shard:
            continue
        item = dataset[index]
        batch = batch_to(item, args.device)
        inputs = [batch[k][:, :views] for k in ["rgb", "thermal", "E", "K"]]
        valid = batch["valid"][:, :views]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            initial = teacher.reconstruction(*inputs)
            original = teacher.refiner(*inputs, initial=initial, source_valid=valid)
            candidate = student(*inputs, initial=initial, source_valid=valid)
        target_E, target_K = batch["E"][:, views:], batch["K"][:, views:]
        h, w = initial.depth.shape[-2:]
        outputs = [original, candidate] + ([initial] if args.include_base else [])
        labels = ["teacher", "student"] + (["base"] if args.include_base else [])
        renders = [render_rgbt(o, target_E, target_K, h, w) for o in outputs]
        mask = batch["valid"][:, views:]
        row = {
            "scene": scene,
            "split": split,
            "views": views,
            "target": item["id"],
            "source_ids": item["context_ids"],
            "root": root,
        }
        for label, output, rendered in zip(labels, outputs, renders):
            metrics = {}
            for m in ["rgb", "thermal"]:
                image = rendered[m].clamp(0, 1)
                metrics[m + "_psnr"] = psnr(image, batch[m][:, views:], mask)
                metrics[m + "_ssim"] = float(
                    ssim(image * mask, batch[m][:, views:] * mask)
                )
            metrics["lpips"] = float(
                perceptual(
                    (rendered["rgb"].clamp(0, 1) * mask).flatten(0, 1),
                    (batch["rgb"][:, views:] * mask).flatten(0, 1),
                    normalize=True,
                ).mean()
            )
            metrics["depth_log_error"] = float(
                depth_loss(gaussian_depth(output), batch["depth"][:, :views])
            )
            metrics["alpha"] = float(
                (rendered["alpha"] * mask).sum() / mask.sum().clamp_min(1)
            )
            row[label] = metrics
            row[label + "_surface"] = surface_metrics(output, teacher.last_association)
        row["delta"] = {
            k: row["student"][k] - row["teacher"][k] for k in row["teacher"]
        }
        if args.include_base:
            row["delta_base"] = {
                k: row["student"][k] - row["base"][k] for k in row["base"]
            }
        rows.append(row)
        filename = f"{views}_{item['id']}.jpg"
        panel(
            out / filename,
            batch,
            renders[1],
            baseline=renders[0],
            baseline_label="original model",
            prediction_label="shared refiner",
        )
        if args.include_base:
            panel(
                out / ("base_" + filename),
                batch,
                renders[1],
                baseline=renders[2],
                baseline_label="base reconstruction",
                prediction_label="shared refiner",
            )
        cards.append(
            f'<article><h3>{html.escape(str(views) + " views / " + item["id"])}</h3><img loading="lazy" src="{html.escape(filename)}"></article>'
        )
        with (out / f"rows_{args.shard}.jsonl").open("a") as f:
            f.write(json.dumps(row) + "\n")
        print(
            "EVAL",
            number,
            scene,
            views,
            json.dumps(row["delta"]),
            "seconds",
            round(time.time() - start, 1),
            flush=True,
        )
        result = aggregate(rows, policy)
        (out / f"results_{args.shard}.json").write_text(
            json.dumps(result, indent=2) + "\n"
        )
    assert rows
    (out / f"index_{args.shard}.html").write_text(
        '<!doctype html><html><meta charset="utf-8"><title>Shared refiner comparison</title><style>body{max-width:1200px;margin:30px auto;font:16px system-ui}img{max-width:100%}</style><h1>Shared refiner vs original teacher</h1><p>Each row: target / teacher / shared refiner.</p>'
        + "".join(cards)
        + "</html>"
    )
    print(
        "COMPLETE",
        json.dumps(
            {k: result[k] for k in ["cases", "scenes", "passed", "summary_delta"]}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
