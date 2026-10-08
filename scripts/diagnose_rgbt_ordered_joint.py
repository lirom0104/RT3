#!/usr/bin/env python3
"""Validation-only error attribution; hybrid teacher/student paths are not candidates."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import cv2
import numpy as np
import torch
from cache_rgbt_joint import move

from depth_anything_3.rgbt.engine import batch_to, psnr, ssim
from depth_anything_3.rgbt.legacy.joint_refinement import JointRefiner
from depth_anything_3.rgbt.learned_joint import LearnedJointRefiner
from depth_anything_3.rgbt.ordered_joint import apply_phase, prepare_phase
from depth_anything_3.rgbt.renderer import render_rgbt


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refiner", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument(
        "--teacher", default="outputs/rgbt/joint_training_20260920/before/model.pt"
    )
    parser.add_argument(
        "--cache", default="outputs/rgbt/joint_training_20260920/cache_ordered_val"
    )
    args = parser.parse_args()
    torch.set_num_threads(4)
    cv2.setNumThreads(2)
    torch.cuda.set_device(args.device)
    c = torch.load(args.teacher, map_location="cpu", weights_only=False, mmap=True)
    config = c["model_config"]
    teacher = JointRefiner(
        config["neighborhood"], config["crossview"], config["surface"]
    )
    teacher.load_state_dict(
        {
            k.removeprefix("refiner."): v
            for k, v in c["state_dict"].items()
            if k.startswith("refiner.")
        }
    )
    teacher = teacher.to(args.device).eval()
    del c
    c = torch.load(args.refiner, map_location="cpu", weights_only=False)
    model = LearnedJointRefiner(**c["refiner_config"]).to(args.device).eval()
    model.load_state_dict(c["state_dict"])
    assert model.config["ordered_updates"]
    rows = []
    for path in sorted(Path(args.cache).glob("*.pt")):
        data = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        assert data["split"] == "val"
        b = batch_to(data["item"], args.device)
        initial = move(data["initial"], args.device)
        reference_render = move(data["teacher_render"], args.device)
        v = initial.depth.shape[1]
        rgb, thermal, E, K, valid = [
            b[key][:, :v] for key in ["rgb", "thermal", "E", "K", "valid"]
        ]
        mask = b["valid"][:, v:]
        reference = {}
        for modality in ["rgb", "thermal"]:
            reference[modality + "_psnr"] = psnr(
                reference_render[modality].clamp(0, 1), b[modality][:, v:], mask
            )
            reference[modality + "_ssim"] = float(
                ssim(
                    reference_render[modality].clamp(0, 1) * mask,
                    b[modality][:, v:] * mask,
                )
            )
        for replaced in [-1, 0, 1, 2]:
            output = initial
            with torch.autocast("cuda", dtype=torch.bfloat16):
                for phase, original in enumerate(
                    [teacher.local, teacher.crossview, teacher.surface]
                ):
                    if phase == replaced:
                        output = original(
                            rgb, thermal, E, K, initial=output, source_valid=valid
                        )
                    else:
                        p = prepare_phase(model, phase, rgb, thermal, output, valid)
                        output = apply_phase(model, rgb, thermal, E, K, valid, p)
            rendered = render_rgbt(
                output, b["E"][:, v:], b["K"][:, v:], *rgb.shape[-2:]
            )
            delta = {}
            for modality in ["rgb", "thermal"]:
                delta[modality + "_psnr"] = (
                    psnr(rendered[modality].clamp(0, 1), b[modality][:, v:], mask)
                    - reference[modality + "_psnr"]
                )
                delta[modality + "_ssim"] = (
                    float(
                        ssim(
                            rendered[modality].clamp(0, 1) * mask,
                            b[modality][:, v:] * mask,
                        )
                    )
                    - reference[modality + "_ssim"]
                )
            rows.append(
                {
                    "scene": b["scene"],
                    "views": v,
                    "target": b["id"],
                    "teacher_phase": replaced,
                    "delta": delta,
                }
            )
        print("DIAGNOSED", b["scene"], v, flush=True)
    summary = {}
    for phase in [-1, 0, 1, 2]:
        selected = [row for row in rows if row["teacher_phase"] == phase]
        summary[str(phase)] = {
            key: float(np.mean([row["delta"][key] for row in selected]))
            for key in selected[0]["delta"]
        }
    result = {
        "purpose": "Error attribution only; hybrids retain teacher modules and must never be deployed as a merged candidate",
        "config": vars(args),
        "summary": summary,
        "rows": rows,
    }
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print("SUMMARY", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
