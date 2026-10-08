#!/usr/bin/env python3
"""Cache single-forward native priors, with scene-disjoint supervision."""

import argparse
import json
from pathlib import Path

import cv2
import torch
from cache_rgbt_joint import move

from depth_anything_3.rgbt.checkpoint import load_rgbt_model
from depth_anything_3.rgbt.data import SceneDataset
from depth_anything_3.rgbt.engine import batch_to
from depth_anything_3.rgbt.native_refiner import NativeGaussianRefiner
from depth_anything_3.rgbt.renderer import render_rgbt


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights", default="checkpoints/RGBT/model.pt")
    p.add_argument("--data", default="outputs/rgbt/data_336")
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--split", choices=["train", "val", "test"], required=True)
    p.add_argument("--targets", type=int, default=6)
    p.add_argument("--ids", nargs="+")
    p.add_argument("--device", default="cuda:0")
    a = p.parse_args()
    torch.set_num_threads(4)
    cv2.setNumThreads(2)
    torch.cuda.set_device(a.device)
    torch.manual_seed(20260922)
    a.output.mkdir(parents=True, exist_ok=True)
    model, ckpt = load_rgbt_model(a.weights, a.device)
    refiner = NativeGaussianRefiner()
    state = refiner.state_dict()
    for k in state:
        if not k.startswith("core.head.") and "refiner." + k in ckpt["state_dict"]:
            state[k] = ckpt["state_dict"]["refiner." + k].cpu()
    refiner.load_state_dict(state)
    torch.save(
        {"config": refiner.config, "state_dict": refiner.state_dict()},
        a.output / "seed.pt",
    )
    del ckpt, refiner
    old_mode = model.reconstruction.geometry_mode
    records = []
    for views in [3, 6]:
        ds = SceneDataset(
            a.data,
            a.split,
            views=views,
            targets_per_scene=a.targets,
            sampling="nearest" if views == 3 else "wide",
        )
        if a.ids:
            ds.items = [
                (si, ti)
                for si, ti in ds.items
                if ds.scenes[si]["frames"][ti]["id"] in a.ids
            ]
        for index in range(len(ds)):
            item = ds[index]
            assert item["id"] not in item["context_ids"]
            name = f"{item['id']}_{views}.pt"
            records.append(
                {"file": name, "id": item["id"], "scene": item["scene"], "views": views}
            )
            if (a.output / name).exists():
                continue
            b = batch_to(item, a.device)
            args = [b[k][:, :views] for k in ["rgb", "thermal", "E", "K"]]
            model.reconstruction.geometry_mode = "native"
            with torch.autocast("cuda", dtype=torch.bfloat16):
                initial = model.compute_prior(*args)
            target_args = [b["E"][:, views:], b["K"][:, views:], *b["rgb"].shape[-2:]]
            base = render_rgbt(initial, *target_args)
            previous = None
            if a.split != "train":
                model.reconstruction.geometry_mode = old_mode
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    old = model(*args, source_valid=b["valid"][:, :views])
                previous = render_rgbt(old, *target_args)
                del old
            payload = {
                "format": "rgbt-native-cache-v1",
                "split": a.split,
                "item": item,
                "initial": move(initial, "cpu"),
                "baseline": move(base, "cpu"),
                "previous": move(previous, "cpu"),
            }
            torch.save(payload, a.output / name)
            print(a.split, views, index + 1, len(ds), item["id"], flush=True)
    (a.output / "index.json").write_text(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
