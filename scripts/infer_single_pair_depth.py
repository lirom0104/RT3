#!/usr/bin/env python3
"""Infer one shared depth map from one registered RGB/thermal pair per scene."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageOps

from depth_anything_3.rgbt.checkpoint import load_rgbt_model
from depth_anything_3.rgbt.data import grayscale
from depth_anything_3.rgbt.geometry import gaussian_depth
from depth_anything_3.rgbt.renderer import render_rgbt


SCENES = (
    ("RGBT-Scenes", "Building"),
    ("RGBT-Scenes", "LandScape"),
    ("RGBT-Scenes", "RotaryKiln"),
    ("ThermoScenes", "buildingA_spring"),
    ("ThermoScenes", "buildingA_winter"),
    ("ThermoScenes", "exhibition_building"),
    ("ThermoScenes", "trees"),
)


def select_pair(scene_dir):
    rgb = {p.stem: p for p in (scene_dir / "rgb").iterdir() if p.is_file()}
    thermal = {p.stem: p for p in (scene_dir / "thermal").iterdir() if p.is_file()}
    common = sorted(rgb.keys() & thermal.keys())
    if not common:
        raise ValueError(f"No paired images in {scene_dir}")
    stem = common[len(common) // 2]
    return stem, rgb[stem], thermal[stem]


def preprocess(rgb_path, thermal_path, long_side, device):
    with Image.open(rgb_path) as source:
        rgb = np.asarray(source.convert("RGB"), dtype=np.float32) / 255
    thermal = grayscale(thermal_path)
    h, w = rgb.shape[:2]
    if thermal.shape != (h, w):
        raise ValueError(f"RGB and thermal grids differ: {rgb_path}, {thermal_path}")
    hw = [max(28, round(x * long_side / max(h, w) / 28) * 28) for x in (h, w)]
    rgb_tensor = F.interpolate(
        torch.from_numpy(rgb).permute(2, 0, 1)[None],
        hw,
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )[None].to(device)
    thermal_tensor = F.interpolate(
        torch.from_numpy(thermal)[None, None],
        hw,
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )[None].to(device)
    valid = torch.ones_like(thermal_tensor)
    return rgb, thermal, rgb_tensor, thermal_tensor, valid


def preview(depth):
    valid = np.isfinite(depth) & (depth > 0)
    if not valid.any():
        raise ValueError("Depth contains no positive finite pixels")
    lo, hi = np.percentile(depth[valid], [2, 98])
    hi = max(hi, lo + 1e-6)
    image = np.zeros(depth.shape, dtype=np.uint8)
    image[valid] = (
        np.clip((depth[valid] - lo) / (hi - lo), 0, 1) * 255 + 0.5
    ).astype(np.uint8)
    return Image.fromarray(image), [float(lo), float(hi)]


def overview(output_dir, rows):
    cell_w, cell_h, header = 320, 240, 44
    sheet = Image.new("RGB", (3 * cell_w, len(rows) * (cell_h + header)), "#171717")
    draw = ImageDraw.Draw(sheet)
    for row_index, row in enumerate(rows):
        scene_dir = output_dir / row["dataset"] / row["scene"]
        top = row_index * (cell_h + header)
        for col, (name, label) in enumerate(
            (("rgb_input.png", "RGB input"), ("thermal_input.png", "thermal input"), ("depth.png", "shared refined depth"))
        ):
            with Image.open(scene_dir / name) as image:
                thumb = ImageOps.contain(image.convert("RGB"), (cell_w, cell_h), Image.Resampling.LANCZOS)
            sheet.paste(thumb, (col * cell_w + (cell_w - thumb.width) // 2, top + header + (cell_h - thumb.height) // 2))
            draw.text((col * cell_w + 8, top + 5), f"{row['dataset']}/{row['scene']}", fill="white")
            draw.text((col * cell_w + 8, top + 22), label, fill="white")
    sheet.save(output_dir / "overview.jpg", quality=90)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT.parent / "data")
    parser.add_argument("--weights", type=Path, default=ROOT / "checkpoints/RGBT/model.pt")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/rgbt/single_pair_depth")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--long-side", type=int, default=336)
    args = parser.parse_args()

    pairs = [(dataset, scene, *select_pair(args.data_dir / dataset / scene)) for dataset, scene in SCENES]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model, checkpoint = load_rgbt_model(args.weights, args.device)
    config = checkpoint["model_config"]
    del checkpoint
    rows = []
    for dataset, scene, stem, rgb_path, thermal_path in pairs:
        rgb, thermal, rgb_tensor, thermal_tensor, valid = preprocess(
            rgb_path, thermal_path, args.long_side, args.device
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            result = model(rgb_tensor, thermal_tensor, source_valid=valid)
            rendered = render_rgbt(
                result, result.extrinsics, result.intrinsics,
                *result.depth.shape[-2:],
            )
        refined_native = gaussian_depth(result)[0, 0].float().cpu().numpy()
        backbone_native = result.depth[0, 0].float().cpu().numpy()
        rendered_native = rendered["depth"][0, 0].float().cpu().numpy()
        alpha_native = rendered["alpha"][0, 0, 0].float().cpu().numpy()
        if not np.isfinite(refined_native).all() or (refined_native <= 0).any():
            raise ValueError(f"Invalid refined depth for {dataset}/{scene}/{stem}")
        h, w = rgb.shape[:2]
        full_size = (w, h)
        refined = cv2.resize(refined_native, full_size, interpolation=cv2.INTER_LINEAR)
        rendered_full = cv2.resize(rendered_native, full_size, interpolation=cv2.INTER_LINEAR)
        alpha_full = cv2.resize(alpha_native, full_size, interpolation=cv2.INTER_LINEAR)
        depth_image, display_range = preview(refined)
        scene_dir = args.output_dir / dataset / scene
        scene_dir.mkdir(parents=True, exist_ok=True)
        Image.fromarray((rgb * 255 + 0.5).astype(np.uint8)).save(scene_dir / "rgb_input.png")
        Image.fromarray((thermal.clip(0, 1) * 255 + 0.5).astype(np.uint8)).save(scene_dir / "thermal_input.png")
        depth_image.save(scene_dir / "depth.png")
        np.save(scene_dir / "depth.npy", refined.astype(np.float32))
        np.savez_compressed(
            scene_dir / "native_outputs.npz",
            refined_gaussian_depth=refined_native.astype(np.float32),
            backbone_depth=backbone_native.astype(np.float32),
            rendered_depth=rendered_native.astype(np.float32),
            rendered_alpha=alpha_native.astype(np.float32),
        )
        row = {
            "dataset": dataset,
            "scene": scene,
            "frame": stem,
            "rgb_source": str(rgb_path),
            "thermal_source": str(thermal_path),
            "original_size_hw": [h, w],
            "model_size_hw": list(refined_native.shape),
            "depth_min_max": [float(refined.min()), float(refined.max())],
            "rendered_alpha_min_max": [float(alpha_full.min()), float(alpha_full.max())],
            "preview_percentile_2_98": display_range,
        }
        rows.append(row)
        print(f"{dataset}/{scene} {stem}: {refined_native.shape} -> {refined.shape}", flush=True)
    (args.output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "model": str(args.weights.resolve()),
                "model_architecture": config["architecture"],
                "thermal_adapter_layers": config["reconstruction"]["adapter_layers"],
                "input": "One registered RGB plus thermal pair per scene, no supplied cameras; thermal converted with rgbt.data.grayscale.",
                "depth": "Final shared Gaussian-center camera-z depth. RGB and thermal use the same geometry. Relative scale, not meters.",
                "native_outputs": "refined_gaussian_depth is the final center depth; backbone_depth is the DA3 depth field; rendered_depth/alpha are same-view rasterization.",
                "resize": "Bilinear from model grid to original image grid.",
                "preview": "Per-image 2nd-98th percentile stretch; black near, white far.",
                "scenes": rows,
            },
            ensure_ascii=False,
            indent=2,
        ) + "\n"
    )
    overview(args.output_dir, rows)


if __name__ == "__main__":
    main()
