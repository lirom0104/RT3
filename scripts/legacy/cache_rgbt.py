#!/usr/bin/env python3
"""Cache frozen priors and source-only surfaces for unified-model fine-tuning."""

import argparse
import json
import random
import sys
from dataclasses import fields, is_dataclass, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import cv2
import numpy as np
import torch

from depth_anything_3.rgbt.checkpoint import load_rgbt_model
from depth_anything_3.rgbt.data import SceneDataset
from depth_anything_3.rgbt.engine import batch_to
from depth_anything_3.rgbt.source_anchors import source_anchor_field
from depth_anything_3.rgbt.surface_consensus import anchor_consensus, build_consensus


def cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if is_dataclass(value):
        return replace(
            value, **{f.name: cpu(getattr(value, f.name)) for f in fields(value)}
        )
    if isinstance(value, dict):
        return {k: cpu(v) for k, v in value.items()}
    return value


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights", default="checkpoints/RGBT/model.pt")
    p.add_argument("--data", default="outputs/rgbt/data_336")
    p.add_argument("--split", choices=["train", "val", "test"], default="train")
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--targets", type=int, default=8)
    p.add_argument("--views", nargs="+", type=int, default=[3, 6])
    p.add_argument("--seed", type=int, default=20260920)
    a = p.parse_args()
    torch.set_num_threads(4)
    cv2.setNumThreads(4)
    torch.cuda.set_device(a.device)
    torch.manual_seed(a.seed)
    random.seed(a.seed)
    np.random.seed(a.seed)
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    model, checkpoint = load_rgbt_model(a.weights, a.device)
    del checkpoint
    (out / "config.json").write_text(json.dumps(vars(a), indent=2) + "\n")
    surface = model.config["surface"]
    for views in a.views:
        samplings = (
            ["nearest"]
            if views == 3
            else (["diverse", "wide"] if a.split == "train" else ["wide"])
        )
        for sampling in samplings:
            ds = SceneDataset(
                a.data,
                a.split,
                views=views,
                sampling=sampling,
                training=a.split == "train",
                targets_per_scene=a.targets,
            )
            for si, scene in enumerate(ds.scenes):
                indices = [i for i, (s, _) in enumerate(ds.items) if s == si]
                if len(indices) > a.targets:
                    indices = [
                        indices[i]
                        for i in np.linspace(0, len(indices) - 1, a.targets)
                        .round()
                        .astype(int)
                    ]
                for n, index in enumerate(indices):
                    dest = out / f"{si:02d}_{views}_{sampling}_{n:03d}.pt"
                    item = ds[index]
                    b = batch_to(item, a.device)
                    rgb, thermal, E, K = [
                        b[k][:, :views] for k in ["rgb", "thermal", "E", "K"]
                    ]
                    valid = b["valid"][:, :views]
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        prior = model.compute_prior(rgb, thermal, E, K, valid)
                    assoc = build_consensus(prior, rgb, valid)
                    if surface["anchored"]:
                        assoc = anchor_consensus(prior, rgb, assoc)
                    if surface["anchor_strength"]:
                        delta, counts = source_anchor_field(
                            prior, rgb, valid, surface["anchor_radius"]
                        )
                        assoc.update(
                            source_anchor_delta=delta,
                            source_anchor_radius=surface["anchor_radius"],
                            source_anchor_counts=counts,
                        )
                    torch.save(
                        {"item": item, "output": cpu(prior), "association": cpu(assoc)},
                        dest,
                    )
                    print(
                        json.dumps(
                            {
                                "file": str(dest),
                                "scene": scene["id"],
                                "planes": len(assoc["groups"]),
                            }
                        ),
                        flush=True,
                    )


if __name__ == "__main__":
    main()
