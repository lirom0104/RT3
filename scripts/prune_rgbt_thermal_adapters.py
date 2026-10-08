#!/usr/bin/env python3
"""Export a shared RGB-T checkpoint without its ViT thermal adapters."""

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

from depth_anything_3.rgbt.checkpoint import FORMAT, save_checkpoint
from depth_anything_3.rgbt.model import RGBTGaussianNetwork


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", default="checkpoints/RGBT/model.pt")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    source = Path(args.weights)
    destination = Path(args.output)
    if destination.exists():
        parser.error("Use a new output path")

    torch.set_num_threads(4)
    checkpoint = torch.load(source, map_location="cpu", weights_only=False, mmap=True)
    if checkpoint.get("format") != FORMAT:
        raise ValueError("Expected a complete unified RGB-T checkpoint")
    config = deepcopy(checkpoint["model_config"])
    if not config.get("learned_joint") or not config["reconstruction"].get("adapter_layers"):
        raise ValueError("Expected a shared-refiner checkpoint with thermal adapters")
    if not config["reconstruction"].get("compact"):
        raise ValueError("Single-level thermal encoding requires compact reconstruction")
    removed_layers = config["reconstruction"]["adapter_layers"]
    config["reconstruction"]["adapter_layers"] = []

    model = RGBTGaussianNetwork(**config).eval()
    expected = set(model.state_dict())
    original = checkpoint["state_dict"]
    removed = {name: tensor for name, tensor in original.items() if name not in expected}
    allowed = (
        "reconstruction.adapters.",
        "reconstruction.thermal_encoder.stages.",
    )
    if not removed or any(not name.startswith(allowed) for name in removed):
        raise ValueError("Unexpected parameter removal")
    model.load_state_dict(
        {name: tensor for name, tensor in original.items() if name in expected},
        strict=True,
    )
    removed_parameters = sum(tensor.numel() for tensor in removed.values())
    save_checkpoint(
        destination,
        model,
        step=checkpoint.get("step"),
        training=checkpoint.get("training"),
        pruning={
            "source_checkpoint": str(source.resolve()),
            "adapter_layers": removed_layers,
            "removed_parameters": removed_parameters,
            "removed_keys": sorted(removed),
            "validation": "Compare against inference with adapters disabled",
        },
    )
    print(
        json.dumps(
            {
                "output": str(destination),
                "removed_parameters": removed_parameters,
                "remaining_parameters": sum(p.numel() for p in model.parameters()),
                "remaining_adapter_keys": [
                    name for name in model.state_dict() if name.startswith("reconstruction.adapters.")
                ],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
