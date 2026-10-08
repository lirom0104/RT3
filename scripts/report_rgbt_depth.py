#!/usr/bin/env python3
"""Add clearly labeled independent-range structure previews to depth exports."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from visualize_rgbt_depth import depth_image, font, gray_image


def relative_panel(path, row, kind, rgb, maps):
    w, h = rgb.size
    gap, header, label, bar, footer = 8, 72, 44, 34, 66
    canvas = Image.new(
        "RGB",
        (3 * w + 4 * gap, header + 2 * (label + h + bar + gap) + footer),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (gap, 8),
        f"{row['scene']} | {row['protocol']} | {kind}",
        font=font(17),
        fill="#111111",
    )
    draw.text(
        (gap, 36),
        "INDEPENDENT COLOR SCALES: compare structure, not absolute depth",
        font=font(14),
        fill="#9b341a",
    )
    view_id = row["context_ids"][0] if kind == "SOURCE 1" else row["id"]
    draw.text((gap, 55), f"View: {view_id}", font=font(11), fill="#555555")
    cells = [
        (
            "Input RGB" if kind == "SOURCE 1" else "Held-out target RGB",
            "Same test-scene view",
            rgb,
            None,
        )
    ]
    for title, detail, array, mask, is_depth in maps:
        if is_depth:
            good = mask & np.isfinite(array) & (array > 0)
            values = array[good]
            lo, hi = np.percentile(values, [2, 98]) if values.size else (1, 2)
            limits = (float(max(lo, 1e-5)), float(max(hi, lo * 1.001)))
            cells.append((title, detail, depth_image(array, good, limits), limits))
        else:
            cells.append((title, detail, gray_image(array), None))
    for i, (title, detail, image, limits) in enumerate(cells):
        x, y = gap + (i % 3) * (w + gap), header + (i // 3) * (label + h + bar + gap)
        draw.text((x, y), title, font=font(14), fill="#111111")
        draw.text((x, y + 22), detail, font=font(11), fill="#555555")
        canvas.paste(image, (x, y + label))
        if limits:
            ramp = np.geomspace(*limits, w).astype(np.float32)[None].repeat(8, axis=0)
            canvas.paste(
                depth_image(ramp, np.ones_like(ramp, dtype=bool), limits),
                (x, y + label + h + 3),
            )
            draw.text(
                (x, y + label + h + 13),
                f"Near {limits[0]:.3g}  /  Far {limits[1]:.3g} (COLMAP units)",
                font=font(10),
                fill="#555555",
            )
    for i, text in enumerate(
        [
            "Each depth tile uses its own 2-98% log range. Equal colors do NOT mean equal depth.",
            "COLMAP MVS is a reconstructed reference, not sensor GT. Gray checkerboard = invalid.",
            "DA3 = original weights + RGB-T wrapper. Raw values and shared-scale panels are saved separately.",
        ]
    ):
        draw.text(
            (gap, canvas.height - footer + 7 + i * 18),
            text,
            font=font(11),
            fill="#555555",
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def overview(root, paths, name):
    selected = np.linspace(0, len(paths) - 1, min(3, len(paths))).round().astype(int)
    images = [Image.open(root / paths[i]).convert("RGB") for i in selected]
    canvas = Image.new(
        "RGB", (images[0].width, sum(im.height for im in images)), "white"
    )
    y = 0
    for im in images:
        canvas.paste(im, (0, y))
        y += im.height
        im.close()
    canvas.save(root / "overviews" / name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/rgbt/depth_test_20260922")
    )
    parser.add_argument("--data", type=Path, default=Path("outputs/rgbt/data_336"))
    parser.add_argument(
        "--comparison", type=Path, default=Path("outputs/rgbt/test_comparison_20260922")
    )
    args = parser.parse_args()
    scenes = json.loads((args.data / "manifest.json").read_text())["scenes"]
    frames = {f["id"]: f for s in scenes for f in s["frames"]}
    rows, stats, preview_rows = [], [], []
    for dataset in ["RGBT-Scenes", "ThermoScenes"]:
        metadata = json.loads((args.output / f"metadata_{dataset}.json").read_text())
        assert metadata["status"] == "complete"
        rows.extend(
            map(
                json.loads,
                (args.output / f"rows_{dataset}.jsonl").read_text().splitlines(),
            )
        )
    for row in rows:
        base = args.output / row["scene"] / row["protocol"]
        raw = args.output / row["raw"]
        target = dict(np.load(raw / "target_depth.npz"))
        source = dict(np.load(raw / "source_depths.npz"))
        valid = target["valid"]
        threshold = float(target["alpha_threshold"])
        target_maps = [
            (
                "DA3 rendered depth",
                "Expected camera-z",
                target["da3_depth"],
                valid & (target["da3_alpha"] >= threshold),
                True,
            ),
            (
                "Shared rendered depth",
                "Expected camera-z",
                target["shared_depth"],
                valid & (target["shared_alpha"] >= threshold),
                True,
            ),
            (
                "COLMAP MVS reference",
                "Reconstructed depth, not GT",
                target["colmap_reference"],
                target["colmap_reference"] > 0,
                True,
            ),
            ("DA3 opacity", "0 black / 1 white", target["da3_alpha"], valid, False),
            (
                "Shared opacity",
                "0 black / 1 white",
                target["shared_alpha"],
                valid,
                False,
            ),
        ]
        source_valid = source["valid"][0]
        source_maps = [
            (
                "DA3 source depth field",
                "Pipeline field, before rendering",
                source["da3_depth_field"][0],
                source_valid,
                True,
            ),
            (
                "Shared source depth field",
                "Pipeline field, before rendering",
                source["shared_depth_field"][0],
                source_valid,
                True,
            ),
            (
                "COLMAP MVS reference",
                "Reconstructed depth, not GT",
                source["colmap_reference"][0],
                source["colmap_reference"][0] > 0,
                True,
            ),
            (
                "DA3 Gaussian-center Z",
                "Final geometry, source pixel index",
                source["da3_gaussian_center_z"][0],
                source_valid,
                True,
            ),
            (
                "Shared Gaussian-center Z",
                "Final geometry, source pixel index",
                source["shared_gaussian_center_z"][0],
                source_valid,
                True,
            ),
        ]
        paths = {}
        for kind, maps, fid in [
            ("target", target_maps, row["id"]),
            ("source", source_maps, row["context_ids"][0]),
        ]:
            with np.load(frames[fid]["cache"]) as cache:
                rgb = Image.fromarray(
                    (np.moveaxis(cache["rgb"], 0, -1).clip(0, 1) * 255 + 0.5).astype(
                        np.uint8
                    )
                )
            path = base / f"{kind}_relative" / f"{row['id']}.png"
            relative_panel(
                path, row, "SOURCE 1" if kind == "source" else "TARGET", rgb, maps
            )
            paths[kind] = path.relative_to(args.output).as_posix()
            lo, hi = row["color_limits"]
            for title, _detail, values, mask, is_depth in maps:
                if not is_depth:
                    continue
                mask = mask & np.isfinite(values) & (values > 0)
                z = values[mask]
                p2, median, p98 = (
                    np.percentile(z, [2, 50, 98]) if z.size else [float("nan")] * 3
                )
                stats.append(
                    dict(
                        scene=row["scene"],
                        protocol=row["protocol"],
                        target_id=row["id"],
                        view_id=fid,
                        kind=kind,
                        output=title,
                        p02=p2,
                        median=median,
                        p98=p98,
                        valid_fraction=float(mask.mean()),
                        below_shared_range=float((z < lo).mean()) if z.size else 0,
                        above_shared_range=float((z > hi).mean()) if z.size else 0,
                    )
                )
        preview_rows.append(
            dict(
                scene=row["scene"],
                protocol=row["protocol"],
                id=row["id"],
                source_id=row["context_ids"][0],
                **paths,
            )
        )
    for scene in sorted({r["scene"] for r in rows}):
        for protocol in ["nearest3", "wide6"]:
            group = [
                r
                for r in preview_rows
                if r["scene"] == scene and r["protocol"] == protocol
            ]
            for kind in ["target", "source"]:
                overview(
                    args.output,
                    [r[kind] for r in group],
                    f"{scene.replace('/', '_')}_{protocol}_{kind}_relative.png",
                )
    (args.output / "relative_previews.json").write_text(
        json.dumps(preview_rows, indent=2) + "\n"
    )
    with (args.output / "depth_statistics.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(stats[0]))
        writer.writeheader()
        writer.writerows(stats)
    print(
        f"Created {2 * len(rows)} independent-range previews and 16 overviews",
        flush=True,
    )


if __name__ == "__main__":
    main()
