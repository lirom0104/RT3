#!/usr/bin/env python3
"""Package a trained shared refiner with its frozen reconstruction for evaluation."""

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import torch

from depth_anything_3.rgbt.checkpoint import save_checkpoint
from depth_anything_3.rgbt.model import RGBTGaussianNetwork


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--teacher", default="outputs/rgbt/joint_training_20260920/before/model.pt"
    )
    parser.add_argument("--refiner", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    path = Path(args.output)
    if path.exists():
        raise ValueError("Use a new output path")
    teacher = torch.load(
        args.teacher, map_location="cpu", weights_only=False, mmap=True
    )
    student = torch.load(args.refiner, map_location="cpu", weights_only=False)
    if student.get("format") != "da3-rgbt-learned-refiner-training":
        raise ValueError("Expected trained shared refiner")
    config = deepcopy(teacher["model_config"])
    config["learned_joint"] = student["refiner_config"]
    for old_stage in ["neighborhood", "crossview", "surface"]:
        config.pop(old_stage, None)
    config["joint_refinement"] = True
    config["trainable_modules"] = ["refiner"]
    model = RGBTGaussianNetwork(**config)
    state = {
        k: v
        for k, v in teacher["state_dict"].items()
        if k.startswith("reconstruction.")
    }
    state.update({"refiner." + k: v for k, v in student["state_dict"].items()})
    model.load_state_dict(state, strict=True)
    for name, value in model.state_dict().items():
        assert torch.equal(value, state[name]), name
    save_checkpoint(
        path,
        model,
        step=student["step"],
        training={
            "teacher": args.teacher,
            "refiner_checkpoint": args.refiner,
            "validation_required_before_deployment": True,
        },
    )
    path.with_suffix(".config.json").write_text(json.dumps(config, indent=2) + "\n")
    print("EXPORTED", path, "parameters", sum(p.numel() for p in model.parameters()))


if __name__ == "__main__":
    main()
