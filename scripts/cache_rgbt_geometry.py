#!/usr/bin/env python3
"""Cache anchored source geometry and baseline renders on disjoint scene splits."""

import argparse, json
from pathlib import Path
import cv2, torch
from cache_rgbt_joint import move
from depth_anything_3.rgbt.checkpoint import load_rgbt_model
from depth_anything_3.rgbt.data import SceneDataset
from depth_anything_3.rgbt.engine import batch_to
from depth_anything_3.rgbt.renderer import render_rgbt


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--weights", default="checkpoints/RGBT/model.pt")
    p.add_argument("--data", default="outputs/rgbt/data_336")
    p.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    p.add_argument("--train-targets", type=int, default=4)
    p.add_argument("--val-targets", type=int, default=4)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--ids", nargs="+")
    a = p.parse_args()
    torch.set_num_threads(4)
    cv2.setNumThreads(2)
    torch.cuda.set_device(a.device)
    model, ckpt = load_rgbt_model(a.weights, a.device)
    del ckpt
    for split in a.splits:
        folder = a.output / split
        folder.mkdir(parents=True, exist_ok=True)
        records = []
        for v in (3, 6):
            count = (
                a.train_targets
                if split == "train"
                else a.val_targets
                if split == "val"
                else 100000
            )
            ds = SceneDataset(
                a.data,
                split,
                views=v,
                sampling="nearest" if v == 3 else "wide",
                targets_per_scene=count,
            )
            if a.ids:
                ds.items = [
                    (si, ti)
                    for si, ti in ds.items
                    if ds.scenes[si]["frames"][ti]["id"] in a.ids
                ]
            for idx in range(len(ds)):
                item = ds[idx]
                dest = folder / f"{item['id']}_{v}.pt"
                records.append(
                    {
                        "id": item["id"],
                        "scene": item["scene"],
                        "views": v,
                        "file": dest.name,
                    }
                )
                if dest.exists():
                    continue
                b = batch_to(item, a.device)
                inputs = [b[k][:, :v] for k in ("rgb", "thermal", "E", "K")]
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    initial = model.compute_prior(*inputs)
                    baseline = model(
                        *inputs, initial=initial, source_valid=b["valid"][:, :v]
                    )
                previous = render_rgbt(
                    baseline, b["E"][:, v:], b["K"][:, v:], *b["rgb"].shape[-2:]
                )
                torch.save(
                    {
                        "format": "rgbt-geometry-cache-v1",
                        "split": split,
                        "item": item,
                        "initial": move(initial, "cpu"),
                        "baseline": move(baseline, "cpu"),
                        "previous": move(previous, "cpu"),
                    },
                    dest,
                )
                print(split, v, idx + 1, len(ds), item["id"], flush=True)
        (folder / "index.json").write_text(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
