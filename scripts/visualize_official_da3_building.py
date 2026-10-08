#!/usr/bin/env python3
"""Compare untouched official DA3 inference with SharedRefiner source depth."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import matplotlib
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


PROJECT = Path(__file__).resolve().parents[1]
OFFICIAL = Path("/home/lf/feedforward/DA3base")
WEIGHTS = OFFICIAL / "checkpoints/DA3-GIANT-1.1"
EXISTING = PROJECT / "outputs/rgbt/depth_test_20260922"
DATA = PROJECT / "outputs/rgbt/data_336"
DEFAULT_OUTPUT = PROJECT / "outputs/rgbt/official_da3_building_nearest3_20260923"
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def colored_depth(depth, valid, limits):
    lo, hi = limits
    value = np.log(np.maximum(depth, lo) / lo) / np.log(hi / lo)
    rgb = (matplotlib.colormaps["magma"](value.clip(0, 1))[..., :3] * 255 + 0.5).astype(np.uint8)
    yy, xx = np.indices(depth.shape)
    checker = np.where((xx // 8 + yy // 8) % 2, 105, 135).astype(np.uint8)
    bad = ~valid | ~np.isfinite(depth) | (depth <= 0)
    rgb[bad] = checker[bad, None]
    return Image.fromarray(rgb)


def comparison(path, pairs, limits, subtitle):
    w, h = pairs[0][1].size
    margin, gap, row_gap = 16, 12, 42
    header, footer = 103, 61
    width = 2 * w + 2 * margin + gap
    height = header + len(pairs) * (h + row_gap) + footer
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = ImageFont.truetype(FONT, 18)
    label_font = ImageFont.truetype(FONT, 14)
    small_font = ImageFont.truetype(FONT, 11)
    draw.text((margin, 12), "Official DA3-GIANT-1.1 / SharedRefiner", font=title_font, fill="#171717")
    draw.text((margin, 41), subtitle, font=small_font, fill="#555555")
    draw.text((margin, 72), "Official DA3 API", font=label_font, fill="#171717")
    draw.text((margin + w + gap, 72), "DA3 + SharedRefiner", font=label_font, fill="#171717")
    for index, (caption, official, shared) in enumerate(pairs):
        y = header + index * (h + row_gap)
        canvas.paste(official, (margin, y))
        canvas.paste(shared, (margin + w + gap, y))
        draw.text((margin, y + h + 7), caption, font=small_font, fill="#444444")
    bar_y = height - footer + 1
    ramp = np.geomspace(*limits, width - 2 * margin).astype(np.float32)[None].repeat(12, axis=0)
    canvas.paste(colored_depth(ramp, np.ones_like(ramp, dtype=bool), limits), (margin, bar_y))
    draw.text((margin, bar_y + 18), f"Near {limits[0]:.3g}", font=small_font, fill="#444444")
    draw.text((width - margin - 74, bar_y + 18), f"Far {limits[1]:.3g}", font=small_font, fill="#444444")
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def difference_overview(path, rows, limits):
    w, h = rows[0][1].size
    margin, gap, row_gap = 16, 12, 43
    header, footer = 105, 80
    width = 3 * w + 2 * margin + 2 * gap
    height = header + len(rows) * (h + row_gap) + footer
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = ImageFont.truetype(FONT, 18)
    label_font = ImageFont.truetype(FONT, 14)
    small_font = ImageFont.truetype(FONT, 11)
    draw.text((margin, 12), "Official DA3 vs SharedRefiner: source depth difference", font=title_font, fill="#171717")
    draw.text((margin, 41), "Difference = (Shared - Official) / Official; blue = Shared nearer", font=small_font, fill="#555555")
    for column, title in enumerate(("Official DA3 API", "DA3 + SharedRefiner", "Signed difference")):
        draw.text((margin + column * (w + gap), 73), title, font=label_font, fill="#171717")
    for index, (caption, official, shared, difference, median) in enumerate(rows):
        y = header + index * (h + row_gap)
        for column, image in enumerate((official, shared, difference)):
            canvas.paste(image, (margin + column * (w + gap), y))
        draw.text((margin, y + h + 7), f"{caption}    Median change: {median:+.1%}", font=small_font, fill="#444444")
    y = height - footer + 4
    ramp = np.geomspace(*limits, 2 * w + gap).astype(np.float32)[None].repeat(12, axis=0)
    canvas.paste(colored_depth(ramp, np.ones_like(ramp, dtype=bool), limits), (margin, y))
    draw.text((margin, y + 17), f"Depth: {limits[0]:.3g} to {limits[1]:.3g}", font=small_font, fill="#444444")
    dbar = (matplotlib.colormaps["coolwarm"](np.linspace(0, 1, w))[None, :, :3] * 255 + 0.5).astype(np.uint8)
    dbar = np.repeat(dbar, 12, axis=0)
    canvas.paste(Image.fromarray(dbar), (margin + 2 * (w + gap), y))
    draw.text((margin + 2 * (w + gap), y + 17), "-35% (nearer)  /  0%  /  +35% (farther)", font=small_font, fill="#444444")
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def colored_difference(official, shared, valid):
    signed = np.divide(shared - official, official, out=np.zeros_like(official), where=official > 0)
    colors = (matplotlib.colormaps["coolwarm"]((signed.clip(-0.35, 0.35) / 0.35 + 1) / 2)[..., :3] * 255 + 0.5).astype(np.uint8)
    yy, xx = np.indices(official.shape)
    checker = np.where((xx // 8 + yy // 8) % 2, 105, 135).astype(np.uint8)
    bad = ~valid | ~np.isfinite(official) | ~np.isfinite(shared) | (official <= 0) | (shared <= 0)
    colors[bad] = checker[bad, None]
    good = ~bad
    return Image.fromarray(colors), float(np.median(signed[good]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    sys.path.insert(0, str(OFFICIAL / "src"))
    from depth_anything_3.api import DepthAnything3

    api_file = Path(sys.modules["depth_anything_3.api"].__file__).resolve()
    if api_file != (OFFICIAL / "src/depth_anything_3/api.py").resolve():
        raise RuntimeError(f"Unexpected DA3 API: {api_file}")
    rows = [json.loads(line) for line in (EXISTING / "rows_RGBT-Scenes.jsonl").read_text().splitlines()]
    rows = [row for row in rows if row["scene"] == "RGBT-Scenes/Building" and row["protocol"] == "nearest3"]
    preview_ids = {rows[int(i)]["id"] for i in np.linspace(0, len(rows) - 1, 3).round()}
    frames = {f["id"]: f for scene in json.loads((DATA / "manifest.json").read_text())["scenes"] for f in scene["frames"]}

    model = DepthAnything3.from_pretrained(str(WEIGHTS)).to(args.device).eval()
    overview_pairs = []
    difference_pairs = []
    records = []
    for row in rows:
        images, extrinsics, intrinsics, masks = [], [], [], []
        for source_id in row["context_ids"]:
            with np.load(frames[source_id]["cache"]) as cache:
                rgb = (np.moveaxis(cache["rgb"], 0, -1).clip(0, 1) * 255 + 0.5).astype(np.uint8)
                images.append(rgb)
                extrinsics.append(cache["E"].copy())
                intrinsics.append(cache["K"].copy())
                masks.append(cache["valid"][0] > 0.5)
        prediction = model.inference(
            images,
            extrinsics=np.stack(extrinsics),
            intrinsics=np.stack(intrinsics),
            process_res=336,
            process_res_method="upper_bound_resize",
            align_to_input_ext_scale=True,
            infer_gs=False,
        )
        official_depth = np.asarray(prediction.depth, dtype=np.float32)
        if official_depth.shape != (3, *masks[0].shape):
            raise RuntimeError(f"Unexpected official depth shape: {official_depth.shape}")
        with np.load(EXISTING / row["raw"] / "source_depths.npz") as prior:
            shared_depth = prior["shared_depth_field"].astype(np.float32)
            limits = tuple(float(x) for x in prior["color_limits"])
        raw_path = args.output / "raw" / f"{row['id']}.npz"
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            raw_path,
            official_depth=official_depth,
            shared_depth=shared_depth,
            valid=np.stack(masks),
            source_ids=np.array(row["context_ids"]),
            extrinsics=np.stack(extrinsics),
            intrinsics=np.stack(intrinsics),
            color_limits=np.array(limits),
        )
        group_pairs = []
        for i, source_id in enumerate(row["context_ids"]):
            official_img = colored_depth(official_depth[i], masks[i], limits)
            shared_img = colored_depth(shared_depth[i], masks[i], limits)
            caption = f"Source: {source_id.split('_')[-1]}    Target: {row['id'].split('_')[-1]}"
            group_pairs.append((caption, official_img, shared_img))
            if i == 0 and row["id"] in preview_ids:
                overview_pairs.append((caption, official_img, shared_img))
                diff_img, median = colored_difference(official_depth[i], shared_depth[i], masks[i])
                difference_pairs.append((caption, official_img, shared_img, diff_img, median))
        comparison(
            args.output / "groups" / f"{row['id']}.png",
            group_pairs,
            limits,
            "Three nearest RGB source views; known cameras; shared Building color scale",
        )
        records.append({"target_id": row["id"], "source_ids": row["context_ids"], "raw": str(raw_path.relative_to(args.output))})
        print("inferred", row["id"], flush=True)
    comparison(
        args.output / "official_vs_shared_overview.png",
        overview_pairs,
        limits,
        "First source of three evenly spaced targets; known cameras; shared Building color scale",
    )
    difference_overview(args.output / "depth_difference_overview.png", difference_pairs, limits)
    metadata = {
        "official_api": str(api_file),
        "official_repo_commit": subprocess.check_output(["git", "-C", str(OFFICIAL), "rev-parse", "HEAD"], text=True).strip(),
        "official_weights": str(WEIGHTS / "model.safetensors"),
        "model": "DA3-GIANT-1.1",
        "official_inputs": "Same cached undistorted RGB source frames and known COLMAP E/K as SharedRefiner; no thermal input",
        "official_inference": "DepthAnything3.inference; 336 upper_bound_resize; align_to_input_ext_scale=True; infer_gs=False",
        "comparison": "SharedRefiner source depth field from depth_test_20260922, restored to COLMAP scene units",
        "color_scale": "magma log scale, same 0.487..9.032 Building scene range for every displayed depth map",
        "records": records,
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(args.output / "official_vs_shared_overview.png", flush=True)


if __name__ == "__main__":
    main()
