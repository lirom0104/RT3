#!/usr/bin/env python3
"""Cache teacher-forced transitions for one recurrent shared refiner."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import cv2
import torch
import torch.nn.functional as F
from cache_rgbt_joint import move

from depth_anything_3.rgbt.engine import batch_to
from depth_anything_3.rgbt.legacy.joint_refinement import JointRefiner
from depth_anything_3.rgbt.learned_joint import LearnedJointRefiner
from depth_anything_3.rgbt.ordered_joint import prepare_phase
from depth_anything_3.rgbt.renderer import render_rgbt


def pack(output):
    values = {
        k: getattr(output.gaussians, k)
        for k in ["means", "scales", "rotations", "opacities"]
    }
    values.update(
        dc=output.gaussians.harmonics[..., 0],
        thermal=output.thermal_colors,
        depth=output.depth,
        fused_rendering=output.fused_rendering,
    )
    return move(values, "cpu")


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--teacher", default="outputs/rgbt/joint_training_20260920/before/model.pt"
    )
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    torch.set_num_threads(4)
    cv2.setNumThreads(2)
    torch.cuda.set_device(args.device)
    checkpoint = torch.load(
        args.teacher, map_location="cpu", weights_only=False, mmap=True
    )
    config = checkpoint["model_config"]
    teacher = JointRefiner(
        config["neighborhood"], config["crossview"], config["surface"]
    )
    teacher.load_state_dict(
        {
            k.removeprefix("refiner."): v
            for k, v in checkpoint["state_dict"].items()
            if k.startswith("refiner.")
        }
    )
    teacher = teacher.to(args.device).eval()
    del checkpoint
    student = LearnedJointRefiner(ordered_updates=True).to(args.device).eval()
    dest = Path(args.output)
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "config.json").write_text(json.dumps(vars(args), indent=2) + "\n")
    paths = sorted(Path(args.input).glob("*.pt"))
    for number, path in enumerate(paths):
        if args.limit and number >= args.limit:
            break
        output_path = dest / path.name
        if output_path.exists():
            continue
        data = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        b = batch_to(data["item"], args.device)
        initial = move(data["initial"], args.device)
        v = initial.depth.shape[1]
        rgb, thermal, E, K, valid = [
            b[key][:, :v] for key in ["rgb", "thermal", "E", "K", "valid"]
        ]
        previous = initial
        phases = []
        for phase, module in enumerate(
            [teacher.local, teacher.crossview, teacher.surface]
        ):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                prepared = prepare_phase(student, phase, rgb, thermal, previous, valid)
            captured = {}

            def capture(name, captured=captured):
                def hook(_module, _inputs, output):
                    captured[name] = output.float()

                return hook

            handles = [
                (module.head if phase < 2 else module.appearance).register_forward_hook(
                    capture("head")
                )
            ]
            if phase == 2:
                handles.append(
                    module.depth_head.register_forward_hook(capture("depth"))
                )
            with torch.autocast("cuda", dtype=torch.bfloat16):
                target = module(
                    rgb, thermal, E, K, initial=previous, source_valid=valid
                )
            for handle in handles:
                handle.remove()
            delta = captured["head"].tanh()
            weights = delta.new_zeros(15)
            if phase < 2:
                weights[[3, 4, 5, 10, 14]] = 1
                if phase == 1:
                    weights[:11] = 1
            else:
                joint = delta.new_zeros(v, 15, *rgb.shape[-2:])
                joint[:, 3:6], joint[:, 10:15] = delta[:, :3], delta[:, 3:]
                joint[:, 2:3] = F.interpolate(
                    captured["depth"],
                    rgb.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                ).tanh()
                delta = joint
                weights[[2, 3, 4, 5, 10, 11, 12, 13, 14]] = 1
            delta = delta.reshape(1, v, 15, *rgb.shape[-2:]).permute(0, 1, 3, 4, 2)
            render = render_rgbt(target, b["E"][:, v:], b["K"][:, v:], *rgb.shape[-2:])
            compact = {
                k: (pack(value) if k in ["base", "reference"] else move(value, "cpu"))
                for k, value in prepared.items()
            }
            phases.append(
                {
                    "prepared": compact,
                    "teacher": pack(target),
                    "teacher_render": move(render, "cpu"),
                    "target_delta": delta.cpu().bfloat16(),
                    "delta_weight": weights.cpu(),
                }
            )
            previous = target
        payload = {
            "format": "da3-rgbt-ordered-joint-v1",
            "split": data["split"],
            "item": data["item"],
            "initial": data["initial"],
            "phases": phases,
            "teacher": data["teacher"],
            "teacher_render": data["teacher_render"],
            "source_cache": str(path.resolve()),
            "teacher_weights": args.teacher,
        }
        torch.save(payload, output_path.with_suffix(".tmp"))
        output_path.with_suffix(".tmp").replace(output_path)
        print("CACHED", number + 1, len(paths), data["item"]["scene"], v, flush=True)
    print("COMPLETE", flush=True)


if __name__ == "__main__":
    main()
