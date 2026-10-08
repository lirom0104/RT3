#!/usr/bin/env python3
"""Migrate trained sequential refiners into the shared JointRefiner without retraining."""

import argparse
import hashlib
import json
import sys
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import torch

from depth_anything_3.rgbt.checkpoint import FORMAT, save_checkpoint
from depth_anything_3.rgbt.legacy.model import LegacyRGBTGaussianNetwork


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    path = Path(args.output)
    if path.exists():
        raise ValueError("Use a new output path; preserve the original checkpoint")
    before = torch.load(args.weights, map_location="cpu", weights_only=False, mmap=True)
    if before.get("format") != FORMAT:
        raise ValueError("Expected unified RGB-T checkpoint")
    config = deepcopy(before["model_config"])
    if config.get("joint_refinement"):
        raise ValueError("Checkpoint already uses JointRefiner")
    config["joint_refinement"] = True
    prefixes = {
        "neighborhood_refiner.": "refiner.local.",
        "crossview_refiner.": "refiner.crossview.",
        "surface_refiner.": "refiner.surface.",
    }
    state = {}
    for name, value in before["state_dict"].items():
        mapped = next(
            (
                new + name[len(old) :]
                for old, new in prefixes.items()
                if name.startswith(old)
            ),
            name,
        )
        if mapped in state:
            raise ValueError(f"Duplicate migrated parameter: {mapped}")
        state[mapped] = value
    model = LegacyRGBTGaussianNetwork(**config).eval()
    model.load_state_dict(state, strict=True)
    for name, value in model.state_dict().items():
        assert torch.equal(value, state[name]), name
    metadata = {
        "mode": "shared_source_geometry_and_attention_projection",
        "learned_weights_unchanged": True,
        "source_checkpoint_sha256": sha256(args.weights),
        "optimizer_reset": True,
        "parameters": sum(p.numel() for p in model.parameters()),
        "prefix_mapping": prefixes,
    }
    save_checkpoint(path, model, step=before.get("step"), fusion=metadata)
    path.with_suffix(".config.json").write_text(json.dumps(config, indent=2) + "\n")
    metadata["checkpoint_sha256"] = sha256(path)
    path.with_suffix(".fusion.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
