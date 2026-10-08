#!/usr/bin/env python3
"""Average compatible trained shared-refiner weights; inference stays one network."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import torch

from depth_anything_3.rgbt.learned_joint import LearnedJointRefiner


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--coefficients", nargs="+", type=float)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    weights = args.coefficients or [1 / len(args.inputs)] * len(args.inputs)
    assert len(weights) == len(args.inputs) and min(weights) >= 0 and sum(weights) > 0
    weights = [x / sum(weights) for x in weights]
    states = [
        torch.load(p, map_location="cpu", weights_only=False) for p in args.inputs
    ]
    configs = [LearnedJointRefiner(**s["refiner_config"]).config for s in states]
    assert all(c == configs[0] for c in configs)
    keys = set(states[0]["state_dict"])
    assert all(set(s["state_dict"]) == keys for s in states)
    merged = {
        k: sum(w * s["state_dict"][k] for w, s in zip(weights, states))
        for k in states[0]["state_dict"]
    }
    model = LearnedJointRefiner(**configs[0])
    model.load_state_dict(merged, strict=True)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    assert not out.exists()
    torch.save(
        {
            "format": "da3-rgbt-learned-refiner-training",
            "refiner_config": configs[0],
            "state_dict": merged,
            "step": max(s["step"] for s in states),
            "weight_averaging": {"inputs": args.inputs, "coefficients": weights},
            "optimizer": None,
            "args": states[0].get("args", {}),
        },
        out,
    )
    print("SAVED", out)


if __name__ == "__main__":
    main()
