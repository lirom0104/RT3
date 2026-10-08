#!/usr/bin/env python3
"""Write a self-contained pruned checkpoint after paired ablation confirmation.

Only explicitly configured branches are removed. All remaining tensors must
load exactly; arbitrary missing or unexpected weights are rejected.
"""

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


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights", required=True)
    p.add_argument(
        "--changes",
        required=True,
        help="JSON mapping variant names to tested overrides",
    )
    p.add_argument("--variant", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    torch.set_num_threads(4)
    old = torch.load(a.weights, map_location="cpu", weights_only=False, mmap=True)
    if old["format"] != FORMAT:
        raise ValueError("Expected unified checkpoint")
    changes = json.loads(Path(a.changes).read_text())[a.variant]
    cfg = deepcopy(old["model_config"])
    sections = {
        "reconstruction": "reconstruction",
        "neighborhood_refiner": "neighborhood",
        "crossview_refiner": "crossview",
        "surface_refiner": "surface",
    }
    for key, value in changes.items():
        module, attr = key.split(".", 1)
        section = cfg[sections[module]]
        if attr.startswith("adapters.") and attr.endswith(".gamma") and value == 0:
            index = int(attr.split(".")[1])
            section["adapter_layers"] = [
                i for i in section["adapter_layers"] if i != index
            ]
        elif attr == "adapters_enabled" and value is False:
            section["adapter_layers"] = []
        elif attr in section:
            section[attr] = value
        else:
            raise ValueError(f"Cannot serialize intervention: {key}")
    for section in sections.values():
        cfg[section]["compact"] = True
    model = LegacyRGBTGaussianNetwork(**cfg).eval()
    expected = set(model.state_dict())
    omitted = {k: v for k, v in old["state_dict"].items() if k not in expected}
    allowed = [
        "reconstruction.rgb_head.",
        "neighborhood_refiner.state_conv.",
        "crossview_refiner.state_conv.",
    ]
    if not cfg["surface"]["learn_planes"]:
        allowed.append("surface_refiner.plane_head.")
    for index in old["model_config"]["reconstruction"]["adapter_layers"]:
        if index not in cfg["reconstruction"]["adapter_layers"]:
            allowed.append(f"reconstruction.adapters.{index}.")
    if not cfg["reconstruction"]["adapter_layers"]:
        allowed.append("reconstruction.thermal_encoder.stages.")
    if any(not k.startswith(tuple(allowed)) for k in omitted):
        raise ValueError("Unexpected parameter removal")
    model.load_state_dict(
        {k: v for k, v in old["state_dict"].items() if k in expected}, strict=True
    )
    out = Path(a.output)
    if out.exists():
        raise ValueError(
            "Use a new output path; existing checkpoints are not overwritten"
        )
    metadata = {
        "source_checkpoint": str(Path(a.weights).resolve()),
        "variant": a.variant,
        "changes": changes,
        "removed_parameters": sum(v.numel() for v in omitted.values()),
        "removed_keys": sorted(omitted),
        "optimizer_reset": True,
    }
    save_checkpoint(out, model, step=old.get("step"), pruning=metadata)
    out.with_suffix(".config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    metadata["checkpoint_sha256"] = sha256(out)
    out.with_suffix(".pruning.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
