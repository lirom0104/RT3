#!/usr/bin/env python3
"""Test-only source depth fields and held-out target rendered depth, as PNG/NPZ."""

import argparse
import hashlib
import json
import sys
import time
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import cv2
import matplotlib
import numpy as np
import torch
from PIL import Image, ImageDraw

from ablate_rgbt import VARIANTS, configured
from depth_anything_3.rgbt.checkpoint import load_rgbt_model
from depth_anything_3.rgbt.data import SceneDataset, camera_parameters
from depth_anything_3.rgbt.engine import batch_to
from depth_anything_3.rgbt.geometry import gaussian_depth
from depth_anything_3.rgbt.renderer import render_rgbt
from depth_anything_3.utils.read_write_model import read_model
from visualize_rgbt_test import BASELINE_NOTE, bitmap, font


def read_colmap_depth(path):
    with Path(path).open("rb") as stream:
        shape = []
        for _ in range(3):
            token = b""
            while True:
                char = stream.read(1)
                if char == b"&":
                    break
                if not char or len(token) > 20:
                    raise ValueError(f"Invalid COLMAP array header: {path}")
                token += char
            shape.append(int(token))
        w, h, channels = shape
        if channels != 1:
            raise ValueError(f"Expected scalar depth: {path}")
        values = np.fromfile(stream, dtype="<f4")
    if values.size != w * h:
        raise ValueError(f"Truncated COLMAP depth: {path}")
    return values.reshape(w, h, order="F").T.copy()


class DenseReferences:
    def __init__(self, scene, data_root):
        self.scene = scene
        self.root = data_root / scene["id"] / "dense_colmap"
        self.cameras, images, _ = read_model(str(self.root / "sparse"), ext=".bin")
        self.images = {im.name: im for im in images.values()}
        self.frames = {f["id"]: f for f in scene["frames"]}

    @lru_cache(maxsize=300)
    def get(self, frame_id):
        frame = self.frames[frame_id]
        dense_image = self.images[frame["name"]]
        camera = self.cameras[dense_image.camera_id]
        if camera.model != "PINHOLE":
            raise ValueError("Dense COLMAP must use undistorted PINHOLE cameras")
        with np.load(frame["cache"]) as cached:
            target_K = cached["K"]
            target_E = cached["E"]
            valid = cached["valid"][0] > 0.5
        dense_E = np.eye(4)
        dense_E[:3, :3] = dense_image.qvec2rotmat()
        dense_E[:3, 3] = dense_image.tvec
        if not np.allclose(target_E, dense_E, atol=1e-5):
            raise ValueError(f"Dense/sparse camera frames differ for {frame_id}")
        depth_path = self.root / "stereo/depth_maps" / f"{frame['name']}.geometric.bin"
        depth = read_colmap_depth(depth_path)
        dense_K, _ = camera_parameters(camera)
        dense_K[0] *= depth.shape[1] / camera.width
        dense_K[1] *= depth.shape[0] / camera.height
        h, w = valid.shape
        yy, xx = np.indices((h, w), dtype=np.float32)
        # Cached images are already undistorted: match rays using PINHOLE K.
        rays = np.stack([xx, yy, np.ones_like(xx)], axis=-1) @ np.linalg.inv(target_K).T
        uv = rays @ dense_K.T
        mx = (uv[..., 0] / uv[..., 2]).astype(np.float32)
        my = (uv[..., 1] / uv[..., 2]).astype(np.float32)
        aligned = cv2.remap(
            depth, mx, my, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT
        )
        inside = (
            (mx >= 0)
            & (mx <= depth.shape[1] - 1)
            & (my >= 0)
            & (my <= depth.shape[0] - 1)
        )
        good = valid & inside & np.isfinite(aligned) & (aligned > 0)
        return np.where(good, aligned, 0).astype(np.float32)

    def limits(self):
        # One fixed log color range per scene, used for both protocols and models.
        samples = []
        for index in self.scene["target_pool"]:
            z = self.get(self.scene["frames"][index]["id"])
            values = z[z > 0]
            if values.size:
                samples.append(values[:: max(1, values.size // 4000)])
        lo, hi = np.percentile(np.concatenate(samples), [2, 98])
        return float(max(lo, 1e-5)), float(max(hi, lo * 1.001))


def source_scale(item, frames):
    E = torch.stack(
        [SceneDataset.frame(frames[name]["cache"])["E"] for name in item["context_ids"]]
    )
    E = E @ torch.linalg.inv(E[0])
    scale = torch.pdist(torch.linalg.inv(E)[:, :3, 3]).median().clamp_min(1e-4)
    E[:, :3, 3] /= scale
    torch.testing.assert_close(E, item["E"][:-1], atol=1e-5, rtol=1e-5)
    return float(scale)


def depth_image(depth, valid, limits):
    lo, hi = limits
    values = (np.log(np.maximum(depth, lo)) - np.log(lo)) / np.log(hi / lo)
    colors = (
        matplotlib.colormaps["magma"](values.clip(0, 1))[..., :3] * 255 + 0.5
    ).astype(np.uint8)
    yy, xx = np.indices(depth.shape)
    checker = np.where((xx // 8 + yy // 8) % 2, 105, 135).astype(np.uint8)
    valid = valid & np.isfinite(depth) & (depth > 0)
    colors[~valid] = np.repeat(checker[..., None], 3, axis=-1)[~valid]
    return Image.fromarray(colors)


def gray_image(array):
    return Image.fromarray((np.clip(array, 0, 1) * 255 + 0.5).astype(np.uint8)).convert(
        "RGB"
    )


def panel(path, title, subtitle, cells, limits, note):
    w, h = cells[0][2].size
    gap, header, label, footer = 8, 70, 48, 108
    canvas = Image.new(
        "RGB", (3 * w + 4 * gap, header + 2 * (h + label + gap) + footer), "white"
    )
    draw = ImageDraw.Draw(canvas)
    draw.text((gap, 8), title, font=font(18), fill="#151515")
    draw.text((gap, 37), subtitle, font=font(12), fill="#444444")
    for index, (name, detail, image) in enumerate(cells):
        x = gap + (index % 3) * (w + gap)
        y = header + (index // 3) * (h + label + gap)
        draw.text((x, y), name, font=font(14), fill="#151515")
        draw.text((x, y + 23), detail, font=font(11), fill="#555555")
        canvas.paste(image.convert("RGB"), (x, y + label))
    y = canvas.height - footer + 5
    bar_w = canvas.width - 2 * gap
    ramp = np.geomspace(*limits, bar_w).astype(np.float32)[None].repeat(12, axis=0)
    canvas.paste(depth_image(ramp, np.ones_like(ramp, dtype=bool), limits), (gap, y))
    draw.text((gap, y + 16), f"Near {limits[0]:.3g}", font=font(12), fill="#333333")
    draw.text(
        (canvas.width - 110, y + 16),
        f"Far {limits[1]:.3g}",
        font=font(12),
        fill="#333333",
    )
    for offset, text in [
        (
            35,
            "Same log-depth color scale across this scene; COLMAP units, not calibrated meters.",
        ),
        (
            52,
            "COLMAP MVS = reconstructed reference, not sensor ground truth. Checkerboard = invalid.",
        ),
        (69, note),
        (
            86,
            "DA3 baseline: original weights + RGB-T wrapper, as in the RGB/thermal comparisons.",
        ),
    ]:
        draw.text((gap, y + offset), text, font=font(11), fill="#555555")
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def make_overview(output, scene, protocol, rows, kind):
    indices = np.linspace(0, len(rows) - 1, min(3, len(rows))).round().astype(int)
    selected = [
        rows[i]["target_panel"] if kind == "target" else rows[i]["source_panels"][0]
        for i in indices
    ]
    images = [Image.open(output / p).convert("RGB") for p in selected]
    canvas = Image.new(
        "RGB", (images[0].width, 42 + sum(i.height for i in images)), "#eeeeee"
    )
    ImageDraw.Draw(canvas).text(
        (8, 10),
        f"{scene} | {protocol} | {kind} | evenly spaced examples",
        font=font(16),
        fill="#111111",
    )
    y = 42
    for im in images:
        canvas.paste(im, (0, y))
        y += im.height
        im.close()
    path = output / "overviews" / f"{scene.replace('/', '_')}_{protocol}_{kind}.png"
    path.parent.mkdir(exist_ok=True)
    canvas.save(path)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", required=True, choices=["RGBT-Scenes", "ThermoScenes"]
    )
    parser.add_argument("--weights", default="checkpoints/RGBT/model.pt")
    parser.add_argument("--data", type=Path, default=Path("outputs/rgbt/data_336"))
    parser.add_argument(
        "--dataset-root", type=Path, default=Path("/home/lf/feedforward/data")
    )
    parser.add_argument(
        "--comparison", type=Path, default=Path("outputs/rgbt/test_comparison_20260922")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/rgbt/depth_test_20260922")
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--alpha-threshold", type=float, default=0.01)
    args = parser.parse_args()
    torch.set_num_threads(4)
    cv2.setNumThreads(2)
    torch.cuda.set_device(args.device)
    args.output.mkdir(parents=True, exist_ok=True)
    rowfile = args.output / f"rows_{args.dataset}.jsonl"
    if rowfile.exists():
        parser.error("Use a fresh output directory for this dataset")
    started = time.time()
    manifest = json.loads((args.data / "manifest.json").read_text())
    scenes = {
        s["id"]: s
        for s in manifest["scenes"]
        if s["split"] == "test" and s["domain"] == args.dataset
    }
    frames = {f["id"]: f for s in scenes.values() for f in s["frames"]}
    references = {
        key: DenseReferences(scene, args.dataset_root) for key, scene in scenes.items()
    }
    ranges = {key: reference.limits() for key, reference in references.items()}
    reference_rows = {
        (r["protocol"], r["id"]): r
        for r in map(
            json.loads,
            (args.comparison / f"rows_{args.dataset}.jsonl").read_text().splitlines(),
        )
    }
    model, checkpoint = load_rgbt_model(args.weights, args.device)
    # Exactly the same full checkpoint as the already-verified RGB/thermal run.
    prior_metadata = json.loads(
        (args.comparison / f"metadata_{args.dataset}.json").read_text()
    )
    if Path(prior_metadata["weights"]).resolve() != Path(args.weights).resolve():
        raise ValueError("Checkpoint differs from the existing RGB/thermal comparison")
    from safetensors import safe_open

    with safe_open(
        prior_metadata["official_weights"], framework="pt", device="cpu"
    ) as official:
        for key in official.keys():
            embedded = "reconstruction.da3." + key.removeprefix("model.")
            if not torch.equal(
                checkpoint["state_dict"][embedded], official.get_tensor(key)
            ):
                raise ValueError(f"DA3 foundation differs: {key}")
        verified_tensors = len(official.keys())
    del checkpoint
    rows = []
    max_pixel_difference = 0
    for views in [3, 6]:
        protocol = "nearest3" if views == 3 else "wide6"
        dataset = SceneDataset(
            args.data,
            "test",
            views=views,
            targets_per_scene=100000,
            sampling="nearest" if views == 3 else "wide",
        )
        dataset.items = [
            (si, ti)
            for si, ti in dataset.items
            if dataset.scenes[si]["domain"] == args.dataset
        ]
        for index in range(len(dataset)):
            item = dataset[index]
            previous = reference_rows[(protocol, item["id"])]
            assert (
                item["context_ids"] == previous["context_ids"]
                and item["id"] not in item["context_ids"]
            )
            batch = batch_to(item, args.device)
            scale = source_scale(item, frames)
            h, w = batch["rgb"].shape[-2:]
            inputs = [batch[k][:, :views] for k in ["rgb", "thermal", "E", "K"]]
            outputs = {}
            with (
                configured(model, VARIANTS["da3_rgbt_wrapper"]),
                torch.autocast("cuda", dtype=torch.bfloat16),
            ):
                outputs["da3"] = model.reconstruction(*inputs, teacher=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                outputs["shared"] = model(
                    *inputs, source_valid=batch["valid"][:, :views]
                )
            heads, centers, rendered = {}, {}, {}
            for method, out in outputs.items():
                heads[method] = out.depth[0].float().cpu().numpy() * scale
                centers[method] = gaussian_depth(out)[0].float().cpu().numpy() * scale
                pred = render_rgbt(
                    out, batch["E"][:, views:], batch["K"][:, views:], h, w
                )
                for mod in ["rgb", "thermal"]:
                    expected = np.asarray(
                        Image.open(
                            args.comparison / previous["raw"] / f"{mod}_{method}.png"
                        )
                    )
                    actual = np.asarray(bitmap(pred[mod][0, 0]))
                    difference = int(
                        np.abs(actual.astype(int) - expected.astype(int)).max()
                    )
                    max_pixel_difference = max(max_pixel_difference, difference)
                    if difference:
                        raise ValueError(
                            f"Target render differs from existing comparison: {item['id']}/{method}/{mod}"
                        )
                rendered[method] = {
                    "depth": pred["depth"][0, 0].float().cpu().numpy() * scale,
                    "alpha": pred["alpha"][0, 0, 0].float().cpu().numpy(),
                }
            del outputs, out, pred
            folder = args.output / item["scene"] / protocol
            raw_dir = folder / "raw" / item["id"]
            raw_dir.mkdir(parents=True, exist_ok=True)
            valid = item["valid"][-1, 0].numpy() > 0.5
            dense = references[item["scene"]].get(item["id"])
            limits = ranges[item["scene"]]
            coverage = {}
            depth_panels = {}
            for method in ["da3", "shared"]:
                good = (
                    valid
                    & (rendered[method]["alpha"] >= args.alpha_threshold)
                    & (rendered[method]["depth"] > 0)
                    & np.isfinite(rendered[method]["depth"])
                )
                coverage[method] = float(good.sum() / valid.sum())
                depth_panels[method] = depth_image(
                    rendered[method]["depth"], good, limits
                )
            np.savez_compressed(
                raw_dir / "target_depth.npz",
                da3_depth=rendered["da3"]["depth"],
                shared_depth=rendered["shared"]["depth"],
                da3_alpha=rendered["da3"]["alpha"],
                shared_alpha=rendered["shared"]["alpha"],
                colmap_reference=dense,
                valid=valid,
                scale_to_colmap=np.float32(scale),
                color_limits=np.array(limits),
                alpha_threshold=np.float32(args.alpha_threshold),
            )
            target_path = folder / "target_depth" / f"{item['id']}.png"
            panel(
                target_path,
                f"{item['scene']} | TEST | {views} source views",
                f"Held-out target: {item['id']}",
                [
                    ("Target RGB", "Not a model input", bitmap(item["rgb"][-1])),
                    (
                        "DA3 rendered depth",
                        f"Camera-z | coverage {100 * coverage['da3']:.1f}%",
                        depth_panels["da3"],
                    ),
                    (
                        "Shared rendered depth",
                        f"Camera-z | coverage {100 * coverage['shared']:.1f}%",
                        depth_panels["shared"],
                    ),
                    (
                        "COLMAP MVS reference",
                        f"Not GT | valid {100 * (dense > 0).mean():.1f}%",
                        depth_image(dense, dense > 0, limits),
                    ),
                    (
                        "DA3 opacity / coverage",
                        "Opacity: 0 black / 1 white",
                        gray_image(rendered["da3"]["alpha"]),
                    ),
                    (
                        "Shared opacity / coverage",
                        "Opacity: 0 black / 1 white",
                        gray_image(rendered["shared"]["alpha"]),
                    ),
                ],
                limits,
                f"Target depth = opacity-normalized expected camera-z; alpha < {args.alpha_threshold:g} shown as invalid.",
            )
            source_paths = []
            source_reference = []
            for si, source_id in enumerate(item["context_ids"]):
                source_dense = references[item["scene"]].get(source_id)
                source_reference.append(source_dense)
                source_valid = item["valid"][si, 0].numpy() > 0.5
                source_path = (
                    folder
                    / "source_depth"
                    / item["id"]
                    / f"{si + 1:02d}_{source_id}.png"
                )
                panel(
                    source_path,
                    f"{item['scene']} | TEST SCENE | source {si + 1}/{views}",
                    f"{source_id} | for target {frames[item['id']]['name']}",
                    [
                        (
                            "Input RGB",
                            "Source image provided to both models",
                            bitmap(item["rgb"][si]),
                        ),
                        (
                            "DA3 source depth field",
                            "Pipeline field, before rendering",
                            depth_image(heads["da3"][si], source_valid, limits),
                        ),
                        (
                            "Shared source depth field",
                            "Pipeline field; final geometry below",
                            depth_image(heads["shared"][si], source_valid, limits),
                        ),
                        (
                            "COLMAP MVS reference",
                            f"Not GT | valid {100 * (source_dense > 0).mean():.1f}%",
                            depth_image(source_dense, source_dense > 0, limits),
                        ),
                        (
                            "DA3 Gaussian-center Z",
                            "Final geometry, indexed by source pixel",
                            depth_image(centers["da3"][si], source_valid, limits),
                        ),
                        (
                            "Shared Gaussian-center Z",
                            "Final geometry, indexed by source pixel",
                            depth_image(centers["shared"][si], source_valid, limits),
                        ),
                    ],
                    limits,
                    "Source depth fields and Gaussian-center Z are different outputs; neither is target-view depth.",
                )
                source_paths.append(source_path.relative_to(args.output).as_posix())
            np.savez_compressed(
                raw_dir / "source_depths.npz",
                da3_depth_field=heads["da3"],
                shared_depth_field=heads["shared"],
                da3_gaussian_center_z=centers["da3"],
                shared_gaussian_center_z=centers["shared"],
                colmap_reference=np.stack(source_reference),
                valid=item["valid"][:views, 0].numpy().astype(bool),
                source_ids=np.array(item["context_ids"]),
                scale_to_colmap=np.float32(scale),
                color_limits=np.array(limits),
            )
            row = dict(
                id=item["id"],
                scene=item["scene"],
                split="test",
                protocol=protocol,
                views=views,
                context_ids=item["context_ids"],
                target_panel=target_path.relative_to(args.output).as_posix(),
                source_panels=source_paths,
                raw=raw_dir.relative_to(args.output).as_posix(),
                color_limits=limits,
                scale_to_colmap=scale,
                coverage=coverage,
                colmap_reference_valid_fraction=float((dense > 0).mean()),
            )
            rows.append(row)
            with rowfile.open("a") as stream:
                stream.write(json.dumps(row) + "\n")
            print(
                args.dataset,
                protocol,
                f"{index + 1}/{len(dataset)}",
                item["id"],
                flush=True,
            )
        for scene in scenes:
            group = [
                r for r in rows if r["scene"] == scene and r["protocol"] == protocol
            ]
            for kind in ["target", "source"]:
                make_overview(args.output, scene, protocol, group, kind)
    if {(r["protocol"], r["id"]) for r in rows} != set(reference_rows):
        raise ValueError("Depth target set differs from RGB/thermal comparison")
    metadata = dict(
        status="complete",
        dataset=args.dataset,
        split="test",
        target_panels=len(rows),
        source_panels=sum(len(r["source_panels"]) for r in rows),
        scene_color_limits=ranges,
        color_scale="magma, logarithmic; per-scene COLMAP MVS target depth 2nd/98th percentiles; shared across methods/protocols",
        units="COLMAP reconstruction units, not calibrated meters; source-baseline normalization reversed",
        baseline=BASELINE_NOTE,
        official_tensors_verified_equal=verified_tensors,
        reference="COLMAP geometric MVS, not sensor ground truth; may use all captured images; never model input",
        reference_alignment="Same extrinsics verified; PINHOLE ray remap from dense K to cached undistorted K; nearest interpolation, no hole filling",
        target_depth="gsplat RGB+ED: opacity-normalized expected camera-z",
        source_depth="Pipeline source depth field plus final Gaussian center camera-z indexed by source pixel",
        alpha_threshold=args.alpha_threshold,
        raw_target_rgb_thermal_max_pixel_difference=max_pixel_difference,
        data=str(args.data.resolve()),
        weights=str(Path(args.weights).resolve()),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        elapsed_seconds=time.time() - started,
    )
    (args.output / f"metadata_{args.dataset}.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print("COMPLETE", json.dumps(metadata), flush=True)


if __name__ == "__main__":
    main()
