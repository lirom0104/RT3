#!/usr/bin/env python3
"""Audit full checkpoint inference counts, cached agreement and scene exports."""

import argparse
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from cache_rgbt_joint import move
from PIL import Image

from depth_anything_3.rgbt.checkpoint import load_rgbt_model
from depth_anything_3.rgbt.engine import batch_to
from depth_anything_3.rgbt.export import export_scene
from depth_anything_3.rgbt.renderer import render_rgbt


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights", default="checkpoints/RGBT_single_forward/model.pt")
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    a = p.parse_args()
    torch.set_num_threads(4)
    torch.cuda.set_device(a.device)
    model, _ = load_rgbt_model(a.weights, a.device)
    counts = {}
    handles = []
    for name, module in [
        ("backbone", model.reconstruction.da3.backbone),
        ("gs_head", model.reconstruction.da3.gs_head),
        ("refiner", model.refiner),
    ]:

        def hook(_m, _i, _o, key=name):
            counts[key] = counts.get(key, 0) + 1

        handles.append(module.register_forward_hook(hook))
    rows = []
    for name in ["RGBT-Scenes_Building_img_025_3", "RGBT-Scenes_Parterre_flir5623_6"]:
        data = torch.load(
            a.cache / (name + ".pt"), map_location="cpu", weights_only=False
        )
        b = batch_to(data["item"], a.device)
        v = data["initial"].depth.shape[1]
        args = [b[k][:, :v] for k in ["rgb", "thermal", "E", "K"]]
        counts.clear()
        with (
            patch(
                "depth_anything_3.rgbt.gaussian_update.render_rgbt",
                side_effect=AssertionError("Unexpected inference rendering"),
            ),
            torch.autocast("cuda", dtype=torch.bfloat16),
        ):
            output = model(*args, source_valid=b["valid"][:, :v])
        assert counts == {"backbone": 1, "gs_head": 1, "refiner": 1}, counts
        observed = dict(counts)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            cached = model.refiner(
                *args,
                initial=move(data["initial"], a.device),
                source_valid=b["valid"][:, :v],
            )
        differences = {}
        for key in ["means", "scales", "rotations", "harmonics", "opacities"]:
            x, y = getattr(output.gaussians, key), getattr(cached.gaussians, key)
            torch.testing.assert_close(x, y, atol=3e-5, rtol=1e-5)
            differences[key] = float((x - y).abs().max())
        torch.testing.assert_close(
            output.thermal_colors, cached.thermal_colors, atol=3e-5, rtol=1e-5
        )
        folder = a.output / name
        folder.mkdir(parents=True, exist_ok=True)
        export_scene(output, folder, b["rgb"].shape[-2:], frames=6)
        from plyfile import PlyData

        for modality in ["rgb", "thermal"]:
            vertices = PlyData.read(folder / (modality + ".ply"))["vertex"]
            assert len(vertices) == output.gaussians.means.shape[1]
            xyz = np.stack([vertices[k] for k in ["x", "y", "z"]], -1)
            np.testing.assert_array_equal(xyz, output.gaussians.means[0].cpu().numpy())
        from depth_anything_3.rgbt import renderer

        with patch.object(
            renderer, "rasterization", wraps=renderer.rasterization
        ) as raster:
            r = render_rgbt(output, b["E"][:, v:], b["K"][:, v:], *b["rgb"].shape[-2:])
        assert raster.call_count == 1, raster.call_count
        np.savez_compressed(
            folder / "target.npz", **{k: t.cpu().numpy() for k, t in r.items()}
        )
        for mod in ["rgb", "thermal", "alpha"]:
            x = r[mod][0, 0].clamp(0, 1).permute(1, 2, 0).cpu().numpy()
            if x.shape[-1] == 1:
                x = x[..., 0]
            Image.fromarray((x * 255 + 0.5).astype("uint8")).save(
                folder / (mod + ".png")
            )
        row = {
            "id": data["item"]["id"],
            "views": v,
            "context_ids": data["item"]["context_ids"],
            "network_calls": observed,
            "target_rasterizations": raster.call_count,
            "cached_parameter_max_error": differences,
            "gaussians": output.gaussians.means.shape[1],
            "image_compositing": False,
            "inference_optimization_steps": 0,
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
    for handle in handles:
        handle.remove()
    (a.output / "audit.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
