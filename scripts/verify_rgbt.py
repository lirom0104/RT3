#!/usr/bin/env python3
"""Verify strict unified loading, cached/full inference and optional reference outputs."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import cv2
import torch

from depth_anything_3.rgbt.checkpoint import load_rgbt_model
from depth_anything_3.rgbt.data import SceneDataset
from depth_anything_3.rgbt.engine import batch_to
from depth_anything_3.rgbt.renderer import render_rgbt
from depth_anything_3.utils.geometry import affine_inverse


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights", default="checkpoints/RGBT/model.pt")
    p.add_argument("--data", default="outputs/rgbt/data_336")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output", default="outputs/rgbt/verification.json")
    p.add_argument(
        "--reference-dir", help="Optional tensors captured before model consolidation"
    )
    a = p.parse_args()
    torch.set_num_threads(4)
    cv2.setNumThreads(4)
    torch.cuda.set_device(a.device)
    model, checkpoint = load_rgbt_model(a.weights, a.device)
    assert len(checkpoint["state_dict"]) == len(model.state_dict())
    report = {
        "weights": str(Path(a.weights).resolve()),
        "strict_load": True,
        "parameters": {
            name: sum(p.numel() for p in module.parameters())
            for name, module in model.named_children()
        },
        "trainable_parameters": sum(
            p.numel() for p in model.parameters() if p.requires_grad
        ),
        "cases": {},
    }
    del checkpoint
    for views, known in [(3, True), (6, True), (3, False)]:
        tag = f"{views}_" + ("known" if known else "predicted")
        reference = None
        if a.reference_dir:
            reference = torch.load(
                Path(a.reference_dir) / (tag + ".pt"),
                map_location="cpu",
                weights_only=False,
            )
            item = reference["item"]
        else:
            item = SceneDataset(
                a.data, "val", views=views, sampling="nearest" if views == 3 else "wide"
            )[0]
        b = batch_to(item, a.device)
        args = [
            b["rgb"][:, :views],
            b["thermal"][:, :views],
            b["E"][:, :views] if known else None,
            b["K"][:, :views] if known else None,
        ]
        valid = b["valid"][:, :views]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            prior = model.compute_prior(*args, source_valid=valid)
            output = model(*args, initial=prior, source_valid=valid)
            direct = model(*args, source_valid=valid)
        errors = {}
        for name in ["means", "scales", "rotations", "harmonics", "opacities"]:
            value = getattr(output.gaussians, name)
            assert torch.isfinite(value).all(), name
            torch.testing.assert_close(
                value, getattr(direct.gaussians, name), atol=3e-5, rtol=3e-5
            )
            if reference is not None:
                expected = getattr(reference["output"].gaussians, name).to(a.device)
                torch.testing.assert_close(value, expected, atol=3e-5, rtol=3e-5)
                torch.testing.assert_close(
                    getattr(prior.gaussians, name),
                    getattr(reference["initial"].gaussians, name).to(a.device),
                    atol=3e-5,
                    rtol=3e-5,
                )
                errors[name] = float((value - expected).abs().max())
        torch.testing.assert_close(
            output.thermal_colors, direct.thermal_colors, atol=3e-5, rtol=3e-5
        )
        assert (output.gaussians.scales > 0).all()
        assert output.gaussians.means.shape == prior.gaussians.means.shape
        cameras = affine_inverse(prior.extrinsics)[..., :3, 3][:, :, None, None]
        rays = [
            (o.gaussians.means.reshape(1, views, *o.depth.shape[-2:], 3) - cameras)
            for o in [prior, output]
        ]
        ray_error = float(
            (
                torch.nn.functional.normalize(rays[0], dim=-1)
                - torch.nn.functional.normalize(rays[1], dim=-1)
            )
            .abs()
            .max()
        )
        if reference is not None:
            reference_rays = [
                o.gaussians.means.to(a.device).reshape_as(rays[0]) - cameras
                for o in [reference["initial"], reference["output"]]
            ]
            reference_ray_error = float(
                (
                    torch.nn.functional.normalize(reference_rays[0], dim=-1)
                    - torch.nn.functional.normalize(reference_rays[1], dim=-1)
                )
                .abs()
                .max()
            )
            assert ray_error <= reference_ray_error + 1e-6
        render = render_rgbt(
            output,
            b["E"][:, views:] if known else output.extrinsics[:, :1],
            b["K"][:, views:] if known else output.intrinsics[:, :1],
            *output.depth.shape[-2:],
        )
        assert all(torch.isfinite(value).all() for value in render.values())
        if reference is not None:
            torch.testing.assert_close(
                output.thermal_colors.cpu(),
                reference["output"].thermal_colors,
                atol=3e-5,
                rtol=3e-5,
            )
            errors["thermal_colors"] = float(
                (output.thermal_colors.cpu() - reference["output"].thermal_colors)
                .abs()
                .max()
            )
            for name, value in render.items():
                torch.testing.assert_close(
                    value.cpu(), reference["render"][name], atol=3e-5, rtol=3e-5
                )
                errors["render_" + name] = float(
                    (value.cpu() - reference["render"][name]).abs().max()
                )
        report["cases"][tag] = {
            "finite": True,
            "cached_matches_full": True,
            "gaussians": output.gaussians.means.shape[1],
            "source_ray_max_error": ray_error,
            "reference_max_absolute_errors": errors,
        }
        print(tag, json.dumps(report["cases"][tag]), flush=True)
    model.train()
    assert (model.refiner if model.learned_joint else model.surface_refiner).training
    assert all(
        not module.training and all(not p.requires_grad for p in module.parameters())
        for module in model.frozen_stages
    )
    report["frozen_stages_verified"] = True
    path = Path(a.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    print("VERIFIED", path, flush=True)


if __name__ == "__main__":
    main()
