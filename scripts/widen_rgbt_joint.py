#!/usr/bin/env python3
"""Double a shared refiner's channels while approximately preserving its function."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import torch

from depth_anything_3.rgbt.learned_joint import LearnedJointRefiner


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--noise", type=float, default=0.00001)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(20260927)
    c = torch.load(args.weights, map_location="cpu", weights_only=False)
    config = LearnedJointRefiner(**c["refiner_config"]).config
    old_channels = config["channels"]
    config["channels"] *= 2
    model = LearnedJointRefiner(**config)
    state = {}
    for key, target in model.state_dict().items():
        value = c["state_dict"][key]
        if value.shape == target.shape:
            state[key] = value
            continue
        if value.ndim == 1:
            value = value.repeat_interleave(2, 0)
        elif key == "core.phase_embedding.weight":
            value = value.repeat_interleave(2, 1)
        elif value.ndim == 4:
            if target.shape[0] == value.shape[0] * 2:
                value = value.repeat_interleave(2, 0)
            if target.shape[1] == value.shape[1] * 2:
                value = value.repeat_interleave(2, 1) * 0.5
            elif key == "core.head.0.weight":
                assert value.shape[1] == old_channels + 4
                value = torch.cat(
                    [
                        value[:, :old_channels].repeat_interleave(2, 1) * 0.5,
                        value[:, old_channels:],
                    ],
                    1,
                )
            if args.noise:
                value = value + torch.randn_like(value) * args.noise
        else:
            raise ValueError(key)
        assert value.shape == target.shape, (key, value.shape, target.shape)
        state[key] = value
    model.load_state_dict(state, strict=True)
    path = Path(args.output)
    assert not path.exists()
    torch.save(
        {
            "format": "da3-rgbt-learned-refiner-training",
            "refiner_config": config,
            "state_dict": state,
            "step": c["step"],
            "optimizer": None,
            "widened_from": args.weights,
            "noise": args.noise,
        },
        path,
    )
    print("SAVED", path, "parameters", sum(p.numel() for p in model.parameters()))


if __name__ == "__main__":
    main()
