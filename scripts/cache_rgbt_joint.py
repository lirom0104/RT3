#!/usr/bin/env python3
"""Cache raw reconstruction, source-only student preparation and teacher supervision."""

import argparse
import hashlib
import json
import random
import sys
from dataclasses import fields, is_dataclass, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import cv2
import numpy as np
import torch

from depth_anything_3.rgbt.checkpoint import load_rgbt_model
from depth_anything_3.rgbt.engine import batch_to
from depth_anything_3.rgbt.learned_joint import LearnedJointRefiner
from depth_anything_3.rgbt.renderer import render_rgbt


def move(value, device):
    if isinstance(value, torch.Tensor):
        return value.detach().to(device)
    if is_dataclass(value):
        return replace(
            value,
            **{f.name: move(getattr(value, f.name), device) for f in fields(value)},
        )
    if isinstance(value, dict):
        return {k: move(v, device) for k, v in value.items()}
    if isinstance(value, list):
        return [move(v, device) for v in value]
    return value


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--weights", default="outputs/rgbt/joint_training_20260920/before/model.pt"
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--cache", nargs="+", required=True)
    parser.add_argument("--per-scene-views", type=int, default=8)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--split", choices=["train", "val"], required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    cv2.setNumThreads(2)
    torch.cuda.set_device(args.device)
    torch.manual_seed(20260920)
    random.seed(20260920)
    np.random.seed(20260920)
    paths = sorted(p for d in args.cache for p in Path(d).glob("*.pt"))
    byscene = {}
    for path in paths:
        data = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        item = data["item"]
        views = data["output"].depth.shape[1]
        key = (item["scene"], views)
        byscene.setdefault(key, []).append(path)
    selected = []
    for key, group in sorted(byscene.items()):
        indices = (
            np.linspace(0, len(group) - 1, min(len(group), args.per_scene_views))
            .round()
            .astype(int)
        )
        selected.extend(group[i] for i in indices)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"config_{args.shard}.json").write_text(
        json.dumps(vars(args), indent=2) + "\n"
    )
    model, checkpoint = load_rgbt_model(args.weights, args.device)
    del checkpoint
    student = LearnedJointRefiner().to(args.device).eval()
    records = []
    for index, path in enumerate(selected):
        if index % args.shards != args.shard:
            continue
        data = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        item = data["item"]
        views = data["output"].depth.shape[1]
        name = hashlib.sha256(str(path.resolve()).encode()).hexdigest()[:16]
        dest = out / (name + ".pt")
        if dest.exists():
            print("SKIP", dest, flush=True)
            continue
        b = batch_to(item, args.device)
        rgb, t, E, K = [b[k][:, :views] for k in ["rgb", "thermal", "E", "K"]]
        valid = b["valid"][:, :views]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            initial = model.reconstruction(rgb, t, E, K)
            teacher = model.refiner(rgb, t, E, K, initial=initial, source_valid=valid)
        with torch.autocast("cuda", enabled=False):
            prepared = student.prepare(rgb, t, initial, valid, True)
        teacher_render = render_rgbt(
            teacher, b["E"][:, views:], b["K"][:, views:], *initial.depth.shape[-2:]
        )
        # Reconstruct a teacher output from these dense fields and the unchanged SH tail.
        supervision = {
            name: getattr(teacher.gaussians, name)
            for name in ["means", "scales", "rotations", "opacities"]
        }
        supervision.update(
            dc=teacher.gaussians.harmonics[..., 0], thermal=teacher.thermal_colors
        )
        payload = {
            "format": "da3-rgbt-joint-training-v1",
            "split": args.split,
            "item": item,
            "initial": move(initial, "cpu"),
            "prepared": move(prepared, "cpu"),
            "teacher": move(supervision, "cpu"),
            "teacher_render": move(teacher_render, "cpu"),
            "source_cache": str(path),
            "teacher_weights": args.weights,
        }
        temp = dest.with_suffix(".tmp")
        torch.save(payload, temp)
        temp.replace(dest)
        record = {
            "file": dest.name,
            "scene": item["scene"],
            "views": views,
            "target": item["id"],
            "split": args.split,
        }
        records.append(record)
        with (out / f"index_{args.shard}.jsonl").open("a") as f:
            f.write(json.dumps(record) + "\n")
        print("CACHED", index + 1, len(selected), json.dumps(record), flush=True)
        del initial, prepared, teacher, teacher_render, payload, data, supervision, b
    print("COMPLETE", args.shard, len(records), "cases", flush=True)


if __name__ == "__main__":
    main()
