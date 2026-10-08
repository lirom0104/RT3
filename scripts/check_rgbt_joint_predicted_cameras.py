#!/usr/bin/env python3
"""Check shared refinement with DA3 cameras using observed source-view diagnostics.

These are source reconstruction diagnostics, not novel-view target accuracy.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import cv2
import torch
from ablate_rgbt import cases

from depth_anything_3.rgbt.checkpoint import load_rgbt_model
from depth_anything_3.rgbt.engine import batch_to, psnr, ssim
from depth_anything_3.rgbt.learned_joint import LearnedJointRefiner
from depth_anything_3.rgbt.renderer import render_rgbt


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refiner", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--teacher", default="outputs/rgbt/joint_training_20260920/before/model.pt"
    )
    args = parser.parse_args()
    torch.set_num_threads(4)
    cv2.setNumThreads(2)
    torch.cuda.set_device(args.device)
    teacher, checkpoint = load_rgbt_model(args.teacher, args.device)
    del checkpoint
    checkpoint = torch.load(args.refiner, map_location="cpu", weights_only=False)
    student = LearnedJointRefiner(**checkpoint["refiner_config"]).to(args.device).eval()
    student.load_state_dict(checkpoint["state_dict"])
    rows = []
    roots = [
        "outputs/rgbt/data_336",
        "outputs/rgbt/datasets/external_336",
        "outputs/rgbt/datasets/heldout_336",
        "outputs/rgbt/datasets/plant_336",
    ]
    for _, split, views, dataset, index, scene in cases(
        roots, 1, [3, 6], ["train", "val", "test"]
    ):
        batch = batch_to(dataset[index], args.device)
        rgb, thermal, valid = [
            batch[key][:, :views] for key in ["rgb", "thermal", "valid"]
        ]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            initial = teacher.reconstruction(rgb, thermal)
            original = teacher.refiner(
                rgb, thermal, initial=initial, source_valid=valid
            )
            candidate = student(rgb, thermal, initial=initial, source_valid=valid)
        row = {"scene": scene, "split": split, "views": views}
        for label, output in [("teacher", original), ("student", candidate)]:
            tensors = [
                output.gaussians.means,
                output.gaussians.scales,
                output.gaussians.rotations,
                output.gaussians.harmonics,
                output.gaussians.opacities,
                output.thermal_colors,
            ]
            assert all(torch.isfinite(tensor).all() for tensor in tensors)
            rendered = render_rgbt(
                output, initial.extrinsics, initial.intrinsics, *rgb.shape[-2:]
            )
            scores = {}
            for modality, target in [("rgb", rgb), ("thermal", thermal)]:
                prediction = rendered[modality].clamp(0, 1)
                scores[modality + "_psnr"] = psnr(prediction, target, valid)
                scores[modality + "_ssim"] = float(
                    ssim(prediction * valid, target * valid)
                )
            row[label] = scores
        row["delta"] = {
            key: row["student"][key] - row["teacher"][key] for key in row["teacher"]
        }
        rows.append(row)
        result = {
            "protocol": "DA3-predicted cameras; source-view reconstruction only; no novel-view quality claim",
            "config": vars(args),
            "cases": len(rows),
            "scenes": len({r["scene"] for r in rows}),
            "rows": rows,
        }
        Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
        print("CHECK", len(rows), scene, views, json.dumps(row["delta"]), flush=True)
        del initial, original, candidate, rendered, tensors, output
    print("COMPLETE", len(rows), flush=True)


if __name__ == "__main__":
    main()
