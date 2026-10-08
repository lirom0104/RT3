#!/usr/bin/env python3
"""Export a standalone single-forward RGB-T checkpoint."""

import argparse
from copy import deepcopy

import torch

from depth_anything_3.rgbt.checkpoint import load_rgbt_model, save_checkpoint
from depth_anything_3.rgbt.native_refiner import NativeGaussianRefiner


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", default="checkpoints/RGBT/model.pt")
    p.add_argument("--refiner", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    model, _ = load_rgbt_model(a.base, "cpu")
    residual = torch.load(a.refiner, map_location="cpu", weights_only=False)
    model.refiner = NativeGaussianRefiner(**residual["config"])
    model.refiner.load_state_dict(residual["state_dict"], strict=True)
    model.reconstruction.geometry_mode = "native"
    model.reconstruction.rgb_source_blend = 0.0
    model.config = deepcopy(model.config)
    model.config["reconstruction"].update(geometry_mode="native", rgb_source_blend=0.0)
    model.config["learned_joint"] = model.refiner.config
    save_checkpoint(
        a.output,
        model.eval(),
        training={
            "refiner": a.refiner,
            "step": residual.get("step"),
            "validation_score": residual.get("score"),
        },
    )
    print(a.output)


if __name__ == "__main__":
    main()
