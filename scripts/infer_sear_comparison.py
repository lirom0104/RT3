#!/usr/bin/env python3
"""Run official DA3 and the best thermal adapter on every SEAR scene.

Output directories contain only colored PLY clouds and depth PNGs. Run records
and resumable progress live separately under artifacts.
"""

import argparse
from datetime import datetime
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
from PIL import Image
import torch

from depth_anything_3.thermal.adapter import load_adapter, load_giant
from depth_anything_3.thermal.preservation import frozen_base_forward
from depth_anything_3.thermal.detail_refinement import refine_rgb_depth
from depth_anything_3.utils.io.input_processor import InputProcessor
from depth_anything_3.utils.logger import logger


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def discover_scenes(root, pairs):
    """Use paired processed images, without requiring any ground-truth labels."""
    root = Path(root).resolve()
    candidates = {}
    for path in sorted(root.rglob("transforms.json")):
        frames = json.loads(path.read_text()).get("frames", [])
        if not frames or not all("rgb" in frame and "thermal" in frame for frame in frames):
            continue
        folder = path.parent.parent if path.parent.name == "VGGT" else path.parent
        key = str(folder.relative_to(root))
        previous = candidates.get(key)
        if previous is None or path.parent.name == "VGGT":
            candidates[key] = path
    scenes = []
    for key, path in sorted(candidates.items()):
        frames = json.loads(path.read_text())["frames"]
        indices = np.linspace(0, len(frames) - 1, min(pairs, len(frames)), dtype=int).tolist()
        views = []
        for modality in ("rgb", "thermal"):
            for index in indices:
                image = (path.parent / frames[index][modality]["file_path"]).resolve()
                if not image.is_file():
                    raise FileNotFoundError(image)
                views.append({"modality": modality, "pair_index": index, "image": str(image)})
        scenes.append({"scene": key, "transforms": str(path), "total_pairs": len(frames),
                       "selected_pair_indices": indices, "views": views})
    if not scenes:
        raise ValueError(f"No paired scenes found under {root}")
    return scenes


def depth_to_world(depth, extrinsic, intrinsic):
    y, x = np.indices(depth.shape)
    pixels = np.stack((x, y, np.ones_like(x)), axis=-1)
    camera = (pixels @ np.linalg.inv(intrinsic).T) * depth[..., None]
    rotation, translation = extrinsic[:3, :3], extrinsic[:3, 3]
    return (camera - translation) @ rotation


def write_ply(path, points, colors):
    """Write portable binary little-endian XYZ/RGB point clouds."""
    points = np.asarray(points, dtype=np.float32)
    colors = np.asarray(colors)
    if points.ndim != 2 or points.shape[1] != 3 or points.shape != colors.shape or not len(points):
        raise ValueError("PLY needs nonempty matching (N, 3) points and colors")
    if not np.isfinite(points).all() or not np.isfinite(colors).all():
        raise ValueError("Nonfinite PLY points/colors")
    colors = np.rint(np.clip(colors, 0, 1) * 255).astype(np.uint8)
    packed = np.empty(len(points), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                           ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    for index, name in enumerate(("x", "y", "z")):
        packed[name] = points[:, index]
    for index, name in enumerate(("red", "green", "blue")):
        packed[name] = colors[:, index]
    header = ("ply\nformat binary_little_endian 1.0\n"
              f"element vertex {len(points)}\n"
              "property float x\nproperty float y\nproperty float z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
    with Path(path).open("wb") as stream:
        stream.write(header.encode("ascii"))
        packed.tofile(stream)


def export_clouds(folder, arrays, images, views, seed, max_points):
    clouds = {}
    for modality in ("rgb", "thermal"):
        points, colors = [], []
        for i, view in enumerate(views):
            if view["modality"] != modality:
                continue
            depth, confidence = arrays["depth"][i], arrays["depth_conf"][i]
            world = depth_to_world(depth, arrays["extrinsics"][i], arrays["intrinsics"][i])
            valid = np.isfinite(world).all(axis=-1) & (depth > 0)
            valid &= confidence >= np.percentile(confidence, 50)
            valid &= depth <= np.percentile(depth, 99)
            border = max(1, int(min(depth.shape) * .02))
            valid[:border] = valid[-border:] = False
            valid[:, :border] = valid[:, -border:] = False
            selected = np.flatnonzero(valid)
            if not len(selected):
                continue
            rng = np.random.default_rng(seed + i)
            selected = rng.choice(selected, min(len(selected), max_points), replace=False)
            points.append(world.reshape(-1, 3)[selected])
            colors.append(images[i].reshape(-1, 3)[selected])
        if not points:
            raise ValueError(f"No valid points for {modality}")
        clouds[modality] = (np.concatenate(points), np.concatenate(colors))
        write_ply(folder / f"{modality}.ply", *clouds[modality])
    write_ply(folder / "combined.ply", np.concatenate([v[0] for v in clouds.values()]),
              np.concatenate([v[1] for v in clouds.values()]))
    return {key: len(value[0]) for key, value in clouds.items()}


def export_depths(folder, depths, scene, mode):
    low, high = np.percentile(depths, [2, 98])
    if high <= low:
        high = low + max(1e-6, abs(float(low)) * 1e-6)
    norm = Normalize(low, high, clip=True)
    cmap = matplotlib.colormaps["turbo"]
    for i, view in enumerate(scene["views"]):
        directory = folder / "depth" / view["modality"]
        directory.mkdir(parents=True, exist_ok=True)
        name = f"{view['pair_index']:05d}_{Path(view['image']).stem}.png"
        colored = np.rint(cmap(norm(depths[i]))[..., :3] * 255).astype(np.uint8)
        Image.fromarray(colored).save(directory / name)
    pairs = len(scene["selected_pair_indices"])
    indices = np.linspace(0, pairs - 1, min(4, pairs), dtype=int)
    figure, axes = plt.subplots(len(indices), 2, figsize=(10, 3.6 * len(indices)), squeeze=False)
    for row, index in enumerate(indices):
        for column, offset in enumerate((0, pairs)):
            axes[row, column].imshow(depths[index + offset], cmap=cmap, norm=norm)
            axes[row, column].set_axis_off()
            modality = "RGB" if column == 0 else "Thermal"
            axes[row, column].set_title(f"{modality} | pair {scene['selected_pair_indices'][index]}")
    figure.suptitle(f"{scene['scene']} | {mode} | {pairs} RGB/thermal pairs")
    figure.subplots_adjust(left=.02, right=.98, top=.95, bottom=.06, wspace=.03, hspace=.12)
    cax = figure.add_axes([.25, .025, .5, .012])
    figure.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal",
                   label="Relative depth (one shared color scale within this scene/run)")
    figure.savefig(folder / "depth.png", dpi=120)
    plt.close(figure)
    return [float(low), float(high)]


def fingerprints(folder):
    return [{"path": str(path.relative_to(folder)), "bytes": path.stat().st_size,
             "sha256": file_hash(path)} for path in sorted(folder.rglob("*")) if path.is_file()]


def verify_completed(folder, record):
    actual = {str(p.relative_to(folder)) for p in folder.rglob("*") if p.is_file()}
    expected = {p["path"] for p in record["files"]}
    if actual != expected:
        raise ValueError(f"Completed output file list changed: {folder}")
    for entry in record["files"]:
        path = folder / entry["path"]
        if path.stat().st_size != entry["bytes"] or file_hash(path) != entry["sha256"]:
            raise ValueError(f"Completed output changed: {path}")


def infer_scene(model, scene, mode, args):
    ref_view_strategy = getattr(args, "ref_view_strategy", "saddle_balanced")
    if ref_view_strategy == "first" and scene["views"][0]["modality"] != "rgb":
        raise ValueError("The first reference view must be an RGB image")
    torch.manual_seed(args.seed)
    normalized, _, _ = InputProcessor()([v["image"] for v in scene["views"]],
                                      process_res=args.process_res, sequential=True)
    if normalized.ndim == 5:
        normalized = normalized[0]
    input_hash = hashlib.sha256(normalized.numpy().tobytes()).hexdigest()
    mean = normalized.new_tensor([.485, .456, .406])[None, :, None, None]
    std = normalized.new_tensor([.229, .224, .225])[None, :, None, None]
    images = (normalized * std + mean).clamp(0, 1).permute(0, 2, 3, 1).numpy()
    inputs = normalized[None].cuda()
    thermal_mask = None
    if mode == "new":
        thermal_mask = torch.tensor([[v["modality"] == "thermal" for v in scene["views"]]],
                                    dtype=torch.bool, device="cuda")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.monotonic()
    teacher_arrays = None
    if mode == "new" and getattr(args, "rgb_detail_sigma", 0) > 0:
        with frozen_base_forward(model) as teacher, torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            original = teacher(inputs, thermal_mask=None, ref_view_strategy=ref_view_strategy,
                               infer_gs=False, use_ray_pose=False)
            teacher_arrays = {key: original[key][0].float().cpu().numpy() for key in ("depth", "depth_conf")}
            del original
        torch.manual_seed(args.seed)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        raw = model(inputs, thermal_mask=thermal_mask, ref_view_strategy=ref_view_strategy,
                    infer_gs=False, use_ray_pose=False)
    torch.cuda.synchronize()
    seconds = time.monotonic() - started
    peak = torch.cuda.max_memory_allocated() / 1024**3
    arrays = {key: raw[key][0].float().cpu().numpy() for key in
              ("depth", "depth_conf", "extrinsics", "intrinsics")}
    refinement = {}
    if teacher_arrays is not None:
        thermal = np.array([v["modality"] == "thermal" for v in scene["views"]], dtype=bool)
        arrays["depth"], refinement = refine_rgb_depth(teacher_arrays["depth"], arrays["depth"], thermal, args.rgb_detail_sigma)
        arrays["depth_conf"][~thermal] = np.minimum(arrays["depth_conf"][~thermal], teacher_arrays["depth_conf"][~thermal])
    for key, value in arrays.items():
        if not np.isfinite(value).all():
            raise ValueError(f"Nonfinite {key}: {scene['scene']}")
    if not (arrays["depth"] > 0).all():
        raise ValueError(f"Nonpositive depth: {scene['scene']}")
    if not (arrays["intrinsics"][:, [0, 1], [0, 1]] > 0).all():
        raise ValueError(f"Invalid intrinsics: {scene['scene']}")
    if arrays["depth"].shape != (len(scene["views"]), *normalized.shape[-2:]):
        raise ValueError("Unexpected depth dimensions")
    del raw, inputs
    torch.cuda.empty_cache()
    return arrays, images, {"input_sha256": input_hash, "shape": list(normalized.shape),
                            "forward_seconds": seconds, "peak_memory_gib": peak,
                            **({"rgb_detail_refinement": refinement} if refinement else {})}


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("/home/lf/feedforward/data/SEAR"))
    parser.add_argument("--base-checkpoint", type=Path,
                        default=Path("/home/lf/feedforward/base/depth-anything-3/checkpoints/DA3-GIANT-1.1"))
    parser.add_argument("--adapter", type=Path, default=ROOT / "checkpoints/thermal_giant_sear/adapter_best.pt")
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs/comparison")
    parser.add_argument("--record-dir", type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=12)
    parser.add_argument("--process-res", type=int, default=518)
    parser.add_argument("--max-points-per-view", type=int, default=15000)
    parser.add_argument("--seed", type=int, default=20261009)
    parser.add_argument("--ref-view-strategy", default="saddle_balanced",
                        choices=("first", "middle", "saddle_balanced", "saddle_sim_range"),
                        help="Reference view strategy; first uses the first selected RGB image")
    parser.add_argument("--save-predictions", action="store_true",
                        help="Save raw predictions under record-dir for subsequent analysis")
    parser.add_argument("--scene", action="append", help="Restrict to an exact dataset/scene key (for checks)")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--rgb-detail-sigma", type=float, default=0,
                        help="Optional official RGB detail fusion; Gaussian sigma in processed pixels (0 disables)")
    args = parser.parse_args()
    if args.pairs < 1 or args.process_res < 14 or args.max_points_per_view < 1 or args.rgb_detail_sigma < 0 or not np.isfinite(args.rgb_detail_sigma):
        parser.error("pairs, process-res and max-points-per-view must be positive")
    return args


def main():
    args = arguments()
    scenes = discover_scenes(args.data_root, args.pairs)
    if args.scene:
        wanted = set(args.scene)
        scenes = [scene for scene in scenes if scene["scene"] in wanted]
        if {scene["scene"] for scene in scenes} != wanted:
            raise ValueError("Unknown --scene")
    if args.dry_run:
        print(json.dumps({"scenes": len(scenes), "pairs_per_scene": args.pairs,
                          "total_selected_views": sum(len(s["views"]) for s in scenes),
                          "scene_keys": [s["scene"] for s in scenes]}, indent=2, ensure_ascii=False))
        return
    if not torch.cuda.is_available():
        raise RuntimeError("DA3 Giant requires CUDA")
    args.output_root = args.output_root.resolve()
    args.record_dir = args.record_dir.resolve()
    if args.record_dir.is_relative_to(args.output_root):
        raise ValueError("Run records must stay outside the requested output directory")
    progress_path = args.record_dir / "progress.json"
    if progress_path.exists() and not args.resume:
        raise FileExistsError(f"Use --resume for existing run records: {args.record_dir}")
    config = {"data_root": str(args.data_root.resolve()), "base_checkpoint": str(args.base_checkpoint.resolve()),
              "adapter": str(args.adapter.resolve()), "adapter_sha256": file_hash(args.adapter),
              "output_root": str(args.output_root), "pairs": args.pairs, "process_res": args.process_res,
              "max_points_per_view": args.max_points_per_view, "seed": args.seed, "scenes": scenes}
    if args.ref_view_strategy != "saddle_balanced":
        config["ref_view_strategy"] = args.ref_view_strategy
    if args.save_predictions:
        config["save_predictions"] = True
    if args.rgb_detail_sigma > 0:
        config["rgb_detail_sigma"] = args.rgb_detail_sigma
        config["depth_method"] = "Adapted coarse log-depth correction plus frozen official RGB detail; thermal unchanged"
    if args.resume:
        if json.loads((args.record_dir / "run_config.json").read_text()) != config:
            raise ValueError("Resume inputs/settings differ from the recorded run")
        progress = json.loads(progress_path.read_text())
    else:
        progress = {"status": "running", "completed": {}, "failed": {}}
    for mode in ("official", "new"):
        for scene in scenes:
            key = f"{mode}/{scene['scene']}"
            destination = args.output_root / key
            if key in progress["completed"]:
                verify_completed(destination, progress["completed"][key])
            elif destination.exists():
                raise FileExistsError(f"Refusing to overwrite existing scene results: {destination}")
    args.record_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(args.record_dir / "run_config.json", config)
    atomic_json(progress_path, progress)
    torch.set_num_threads(8)
    logger.level = 1

    def record(event, **values):
        entry = {"time": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(), "event": event, **values}
        with (args.record_dir / "run.log").open("a") as stream:
            stream.write(json.dumps(entry, ensure_ascii=False) + "\n")
        print(json.dumps(entry, ensure_ascii=False), flush=True)

    record("start", scenes=len(scenes), pairs=args.pairs, gpu=torch.cuda.get_device_name(0),
           physical_gpu=os.environ.get("CUDA_VISIBLE_DEVICES"))
    model = load_giant(args.base_checkpoint).cuda().eval()
    record("official_weights_loaded", strict=True)
    for mode in ("official", "new"):
        if mode == "new":
            adapter = torch.load(args.adapter, map_location="cpu", weights_only=True)
            if Path(adapter["base_checkpoint"]).resolve() != args.base_checkpoint.resolve():
                raise ValueError("Adapter was trained with a different base checkpoint")
            load_adapter(model, adapter)
            model.eval()
            record("adapter_loaded", epoch=adapter["epoch"], step=adapter["step"], tensors=len(adapter["adapter"]))
            del adapter
        for index, scene in enumerate(scenes, 1):
            key = f"{mode}/{scene['scene']}"
            if key in progress["completed"]:
                continue
            staging = args.record_dir / "staging" / key
            if staging.exists():
                shutil.rmtree(staging)
            staging.mkdir(parents=True)
            destination = args.output_root / key
            started = time.monotonic()
            try:
                arrays, images, statistics = infer_scene(model, scene, mode, args)
                if args.ref_view_strategy == "first":
                    statistics["reference_view"] = {"index": 0, **scene["views"][0]}
                if mode == "new":
                    baseline = progress["completed"].get(f"official/{scene['scene']}")
                    if baseline is None or baseline["input_sha256"] != statistics["input_sha256"]:
                        raise ValueError("Official and new inputs differ or official inference failed")
                if args.save_predictions:
                    prediction_path = args.record_dir / "predictions" / key / "predictions.npz"
                    prediction_path.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(prediction_path, **arrays)
                cloud_counts = export_clouds(staging, arrays, images, scene["views"], args.seed, args.max_points_per_view)
                color_range = export_depths(staging, arrays["depth"], scene, mode)
                files = fingerprints(staging)
                if len(files) != 4 + len(scene["views"]) or any(Path(f["path"]).suffix not in {".png", ".ply"} for f in files):
                    raise ValueError("Unexpected output file list")
                destination.parent.mkdir(parents=True, exist_ok=True)
                staging.rename(destination)
                result = {**statistics, "cloud_counts": cloud_counts, "depth_color_range": color_range,
                          "files": files, "total_seconds": time.monotonic() - started}
                progress["completed"][key] = result
                progress["failed"].pop(key, None)
                atomic_json(progress_path, progress)
                record("scene_complete", mode=mode, scene=scene["scene"], index=index, total=len(scenes),
                       files=len(files), points=cloud_counts, **statistics, total_seconds=result["total_seconds"])
                del arrays, images
            except Exception as error:
                progress["failed"][key] = {"error": str(error), "traceback": traceback.format_exc()}
                atomic_json(progress_path, progress)
                record("scene_failed", mode=mode, scene=scene["scene"], error=str(error), traceback=traceback.format_exc())
            gc.collect()
            torch.cuda.empty_cache()
    expected = 2 * len(scenes)
    progress["status"] = "complete" if len(progress["completed"]) == expected and not progress["failed"] else "incomplete"
    atomic_json(progress_path, progress)
    record("finished", status=progress["status"], completed=len(progress["completed"]), expected=expected,
           failed=list(progress["failed"]))
    if progress["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
