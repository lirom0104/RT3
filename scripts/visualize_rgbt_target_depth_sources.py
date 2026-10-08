#!/usr/bin/env python3
"""Render a target depth map from all or each individual RGB-T source view."""

import argparse
from dataclasses import replace
from pathlib import Path

import matplotlib
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from depth_anything_3.rgbt.checkpoint import load_rgbt_model
from depth_anything_3.rgbt.data import SceneDataset
from depth_anything_3.rgbt.engine import batch_to
from depth_anything_3.rgbt.renderer import render_rgbt


ROOT = Path(__file__).resolve().parents[1]
DEPTH_TEST = ROOT / "outputs/rgbt/depth_test_20260922"
DEFAULT_OUTPUT = (
    ROOT
    / "outputs/rgbt/depth_test_20260922/RGBT-Scenes/Building/nearest3/shared_diagnostics"
    / "img_025_target_depth_by_source.png"
)
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def depth_image(depth, valid, limits):
    lo, hi = limits
    value = np.log(np.maximum(depth, lo) / lo) / np.log(hi / lo)
    rgb = (matplotlib.colormaps["magma"](value.clip(0, 1))[..., :3] * 255 + 0.5).astype(np.uint8)
    y, x = np.indices(depth.shape)
    checker = np.where((x // 8 + y // 8) % 2, 105, 135).astype(np.uint8)
    bad = ~valid | ~np.isfinite(depth) | (depth <= 0)
    rgb[bad] = checker[bad, None]
    return Image.fromarray(rgb)


def single_source(output, index, pixels):
    selection = slice(index * pixels, (index + 1) * pixels)
    gs = replace(
        output.gaussians,
        **{
            name: getattr(output.gaussians, name)[:, selection]
            for name in ("means", "scales", "rotations", "harmonics", "opacities")
        },
    )
    thermal_harmonics = (
        output.thermal_harmonics[:, selection]
        if output.thermal_harmonics is not None
        else None
    )
    return replace(
        output,
        gaussians=gs,
        thermal_colors=output.thermal_colors[:, selection],
        thermal_harmonics=thermal_harmonics,
    )


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    item_id = "RGBT-Scenes_Building_img_025"
    dataset = SceneDataset(
        ROOT / "outputs/rgbt/data_336",
        "test",
        views=3,
        sampling="nearest",
        targets_per_scene=100000,
    )
    item = next(dataset[i] for i in range(len(dataset)) if dataset.scenes[dataset.items[i][0]]["id"] == "RGBT-Scenes/Building" and dataset.scenes[dataset.items[i][0]]["frames"][dataset.items[i][1]]["id"] == item_id)
    expected_sources = [f"RGBT-Scenes_Building_img_{n:03d}" for n in (26, 24, 27)]
    if item["context_ids"] != expected_sources:
        raise RuntimeError(f"Source selection changed: {item['context_ids']}")
    model, _ = load_rgbt_model(ROOT / "checkpoints/RGBT/model.pt", args.device)
    batch = batch_to(item, args.device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = model(
            *[batch[key][:, :3] for key in ("rgb", "thermal", "E", "K")],
            source_valid=batch["valid"][:, :3],
        )
        height, width = output.depth.shape[-2:]
        renders = [
            render_rgbt(output, batch["E"][:, 3:], batch["K"][:, 3:], height, width)
        ]
        for index in range(3):
            isolated = single_source(output, index, height * width)
            renders.append(
                render_rgbt(isolated, batch["E"][:, 3:], batch["K"][:, 3:], height, width)
            )
    raw_dir = DEPTH_TEST / "RGBT-Scenes/Building/nearest3/raw" / item_id
    with np.load(raw_dir / "target_depth.npz") as old:
        reference = old["colmap_reference"].astype(np.float32)
        valid = old["valid"].astype(bool)
        limits = tuple(float(v) for v in old["color_limits"])
        scale = float(old["scale_to_colmap"])
        prior_combined = old["shared_depth"]
    depths = np.stack([r["depth"][0, 0].float().cpu().numpy() * scale for r in renders])
    alphas = np.stack([r["alpha"][0, 0, 0].float().cpu().numpy() for r in renders])
    difference = float(np.max(np.abs(depths[0] - prior_combined)))
    if difference > 0.01:
        raise RuntimeError(f"Combined target depth does not reproduce prior output: {difference}")
    images = [depth_image(reference, reference > 0, limits)]
    images.extend(depth_image(z, valid & (a >= 0.01), limits) for z, a in zip(depths, alphas))
    labels = (
        "COLMAP MVS reference",
        "All 3 sources",
        "Only source 026",
        "Only source 024",
        "Only source 027",
    )
    gap, margin, label_height, header, footer = 8, 8, 31, 55, 72
    canvas = Image.new(
        "RGB",
        (3 * width + 4 * gap, header + 2 * (height + label_height + gap) + footer),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    title_font = ImageFont.truetype(FONT, 17)
    label_font = ImageFont.truetype(FONT, 13)
    small_font = ImageFont.truetype(FONT, 11)
    draw.text((margin, 8), "Building img_025: target-view depth by source", font=title_font, fill="#171717")
    draw.text((margin, 34), "Same scene-wide log scale; gray checkerboard = invalid / low opacity", font=small_font, fill="#555555")
    for index, (label, image) in enumerate(zip(labels, images)):
        x = margin + (index % 3) * (width + gap)
        y = header + (index // 3) * (height + label_height + gap)
        draw.text((x, y), label, font=label_font, fill="#171717")
        canvas.paste(image, (x, y + label_height))
    draw.text(
        (margin + 2 * (width + gap), header + height + label_height + gap + 12),
        "Target img_025 was not a model input.\nCOLMAP is a reconstructed reference, not sensor ground truth.",
        font=small_font,
        fill="#555555",
        spacing=9,
    )
    bar_y = canvas.height - footer + 4
    ramp = np.geomspace(*limits, canvas.width - 2 * margin).astype(np.float32)[None].repeat(12, axis=0)
    canvas.paste(depth_image(ramp, np.ones_like(ramp, dtype=bool), limits), (margin, bar_y))
    draw.text((margin, bar_y + 17), f"Near {limits[0]:.3g}", font=small_font, fill="#444444")
    draw.text((canvas.width - 75, bar_y + 17), f"Far {limits[1]:.3g}", font=small_font, fill="#444444")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output)
    np.savez_compressed(
        args.output.with_suffix(".npz"),
        colmap_reference=reference,
        combined_depth=depths[0],
        source_depths=depths[1:],
        combined_alpha=alphas[0],
        source_alphas=alphas[1:],
        source_ids=np.array(item["context_ids"]),
        valid=valid,
        color_limits=np.array(limits),
        scale_to_colmap=np.float32(scale),
    )
    print(args.output)
    print(f"Maximum difference from saved combined depth: {difference:.6g}")


if __name__ == "__main__":
    main()
