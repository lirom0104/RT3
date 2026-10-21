#!/usr/bin/env python3
"""Replay recorded official paired, RGB-only and thermal-only DA3 inference."""

import argparse
from datetime import datetime
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import traceback
from types import SimpleNamespace
from zoneinfo import ZoneInfo

from infer_sear_comparison import (
    ROOT, Image, Normalize, atomic_json, depth_to_world, export_clouds,
    export_depths, file_hash, fingerprints, infer_scene, load_giant,
    logger, matplotlib, np, torch, verify_completed,
)
from infer_sear_official_single_modality import (
    PLY_DTYPE, export_single, processed_images, select_scene,
)

MODES = {"official": None, "official_rgb_only": "rgb", "official_thermal_only": "thermal"}
CODE = [
    "scripts/rerun_official_modalities.py", "scripts/infer_sear_comparison.py",
    "scripts/infer_sear_official_single_modality.py",
    "src/depth_anything_3/thermal/adapter.py", "src/depth_anything_3/model/da3.py",
    "src/depth_anything_3/model/dinov2/vision_transformer.py",
    "src/depth_anything_3/model/dinov2/layers/attention.py",
    "src/depth_anything_3/model/dinov2/layers/block.py",
    "src/depth_anything_3/model/dinov2/layers/swiglu_ffn.py",
    "src/depth_anything_3/utils/io/input_processor.py",
]


def scene_for(original, modality):
    if modality:
        return select_scene(original, modality)
    return {**original, "views": [{**v, "source_view_index": i}
                                 for i, v in enumerate(original["views"])]}


def output_folder(config, progress, mode):
    root = config["output_root"] if mode in progress.get("published_modes", []) else config["staging_output_root"]
    return Path(root) / mode


def read_ply(path):
    with path.open("rb") as stream:
        vertices = None
        while True:
            line = stream.readline()
            if line.startswith(b"element vertex "):
                vertices = int(line.split()[-1])
            if line == b"end_header\n":
                break
            if not line:
                raise ValueError(f"Invalid PLY header: {path}")
        cloud = np.fromfile(stream, dtype=PLY_DTYPE)
    if len(cloud) != vertices:
        raise ValueError(f"Invalid PLY size: {path}")
    return cloud


def validate(config, progress, record_dir, event):
    checked_files, checked_points = 0, 0
    for mode, modality in MODES.items():
        for original in config["scenes"]:
            scene = scene_for(original, modality)
            key = f"{mode}/{scene['scene']}"
            result = progress["completed"][key]
            folder = output_folder(config, progress, mode) / scene["scene"]
            verify_completed(folder, result)
            raw = record_dir / "predictions" / key
            for name, digest in result["raw_sha256"].items():
                if file_hash(raw / name) != digest:
                    raise ValueError(f"Saved prediction/index changed: {key}/{name}")
            normalized = processed_images(scene, config["process_res"])
            if hashlib.sha256(normalized.numpy().tobytes()).hexdigest() != result["input_sha256"]:
                raise ValueError(f"Processed input changed: {key}")
            if result["observed_forward_shapes"] != [[1, *list(normalized.shape)]]:
                raise ValueError(f"Unexpected model call: {key}")
            mean = normalized.new_tensor([.485, .456, .406])[None, :, None, None]
            std = normalized.new_tensor([.229, .224, .225])[None, :, None, None]
            images = (normalized * std + mean).clamp(0, 1).permute(0, 2, 3, 1).numpy()
            clouds = {}
            with np.load(raw / "predictions.npz") as arrays:
                if arrays["depth"].shape != (len(scene["views"]), *normalized.shape[-2:]):
                    raise ValueError(f"Unexpected prediction shape: {key}")
                norm = Normalize(*result["depth_color_range"], clip=True)
                cmap = matplotlib.colormaps["turbo"]
                for i, view in enumerate(scene["views"]):
                    depth, confidence = arrays["depth"][i], arrays["depth_conf"][i]
                    world = depth_to_world(depth, arrays["extrinsics"][i], arrays["intrinsics"][i])
                    valid = np.isfinite(world).all(axis=-1) & (depth > 0)
                    valid &= confidence >= np.percentile(confidence, 50)
                    valid &= depth <= np.percentile(depth, 99)
                    border = max(1, int(min(depth.shape) * .02))
                    valid[:border] = valid[-border:] = False
                    valid[:, :border] = valid[:, -border:] = False
                    candidates = np.flatnonzero(valid)
                    rng = np.random.default_rng(config["seed"] + view["source_view_index"])
                    selected = rng.choice(candidates, min(len(candidates), config["max_points_per_view"]), replace=False)
                    chunks = clouds.setdefault(view["modality"], [[], []])
                    chunks[0].append(world.reshape(-1, 3)[selected].astype(np.float32))
                    chunks[1].append(np.rint(images[i].reshape(-1, 3)[selected] * 255).astype(np.uint8))
                    name = f"{view['pair_index']:05d}_{Path(view['image']).stem}.png"
                    expected = np.rint(cmap(norm(depth))[..., :3] * 255).astype(np.uint8)
                    with Image.open(folder / "depth" / view["modality"] / name) as image:
                        np.testing.assert_array_equal(np.asarray(image), expected)
            expected_clouds = {m: (np.concatenate(xyz), np.concatenate(colors))
                               for m, (xyz, colors) in clouds.items()}
            if modality is None:
                expected_clouds["combined"] = tuple(np.concatenate([expected_clouds[m][i] for m in ("rgb", "thermal")])
                                                    for i in (0, 1))
            for name, (xyz, colors) in expected_clouds.items():
                cloud = read_ply(folder / f"{name}.ply")
                np.testing.assert_array_equal(np.column_stack([cloud[n] for n in ("x", "y", "z")]), xyz)
                np.testing.assert_array_equal(np.column_stack([cloud[n] for n in ("red", "green", "blue")]), colors)
                checked_points += len(cloud)
            expected_files = {f"{name}.ply" for name in expected_clouds} | {"depth.png"} | {
                f"depth/{v['modality']}/{v['pair_index']:05d}_{Path(v['image']).stem}.png" for v in scene["views"]}
            if {f["path"] for f in result["files"]} != expected_files:
                raise ValueError(f"Unexpected output files: {key}")
            checked_files += len(result["files"])
        event("mode_validated", mode=mode, scenes=len(config["scenes"]))
    for entry in config["input_files"]:
        if file_hash(entry["image"]) != entry["sha256"]:
            raise ValueError(f"Source image changed: {entry['image']}")
    return {"status": "passed", "scenes_per_mode": len(config["scenes"]),
            "runs": len(progress["completed"]), "output_files": checked_files,
            "checked_ply_points": checked_points,
            "input_views_per_forward": {m: 12 if v else 24 for m, v in MODES.items()},
            "checks": ["recorded paired inputs and exact single-modality subsets",
                       "one independent official model forward for each scene and mode",
                       "all PLY coordinates/colors match saved predictions and source pixels",
                       "all per-view depth PNG pixels match saved predictions",
                       "all output and prediction hashes; source images unchanged"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-config", type=Path, default=ROOT / "artifacts/comparison_all_scenes_20261010/run_config.json")
    parser.add_argument("--record-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs/depth_pointcloud")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    record_dir = args.record_dir.resolve()
    record_dir.mkdir(parents=True, exist_ok=True)
    config_path, progress_path = record_dir / "run_config.json", record_dir / "progress.json"
    if config_path.exists():
        if not (args.resume or args.verify_only):
            raise FileExistsError("Use --resume or --verify-only")
        config = json.loads(config_path.read_text())
        progress = json.loads(progress_path.read_text())
        if config["source_config"] != str(args.source_config.resolve()) or config["output_root"] != str(args.output_root.resolve()):
            raise ValueError("Resume paths differ from recorded paths")
        for name, digest in config["code_sha256"].items():
            if file_hash(ROOT / name) != digest:
                raise ValueError(f"Code changed: {name}")
    else:
        if args.verify_only:
            raise FileNotFoundError(config_path)
        source_path = args.source_config.resolve()
        source = json.loads(source_path.read_text())
        old_progress = json.loads((source_path.parent / "progress.json").read_text())
        if source["pairs"] != 12 or old_progress["status"] != "complete":
            raise ValueError("Expected a complete twelve-pair reference")
        for scene in source["scenes"]:
            for modality in ("rgb", "thermal"):
                select_scene(scene, modality)
        config = {k: source[k] for k in ("data_root", "base_checkpoint", "pairs", "process_res", "max_points_per_view", "seed", "scenes")}
        config.update({"source_config": str(source_path), "source_config_sha256": file_hash(source_path),
                       "output_root": str(args.output_root.resolve()), "staging_output_root": str(record_dir / "results"),
                       "modes": MODES, "ref_view_strategy": "saddle_balanced", "thermal_adapter": False,
                       "infer_gs": False, "source_cameras_provided": False,
                       "base_weights_sha256": file_hash(Path(source["base_checkpoint"]) / "model.safetensors"),
                       "reference_input_sha256": {s["scene"]: old_progress["completed"]["official/" + s["scene"]]["input_sha256"] for s in source["scenes"]},
                       "input_files": [{"image": v["image"], "sha256": file_hash(v["image"])} for s in source["scenes"] for v in s["views"]],
                       "code_sha256": {name: file_hash(ROOT / name) for name in CODE}})
        atomic_json(config_path, config)
        for name in CODE:
            destination = record_dir / "code" / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, destination)
        progress = {"status": "running", "completed": {}, "failed": {}, "published_modes": []}
        atomic_json(progress_path, progress)
    if file_hash(args.source_config) != config["source_config_sha256"]:
        raise ValueError("Reference configuration changed")
    torch.set_num_threads(8)
    logger.level = 1

    def event(name, **values):
        entry = {"time": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(), "event": name, **values}
        with (record_dir / "run.log").open("a") as stream:
            stream.write(json.dumps(entry, ensure_ascii=False) + "\n")
        print(json.dumps(entry, ensure_ascii=False), flush=True)

    if not args.verify_only:
        if file_hash(Path(config["base_checkpoint"]) / "model.safetensors") != config["base_weights_sha256"]:
            raise ValueError("Model weights changed")
        event("start", scenes=len(config["scenes"]), runs=3 * len(config["scenes"]),
              physical_gpu=os.environ.get("CUDA_VISIBLE_DEVICES"), gpu=torch.cuda.get_device_name(0))
        model = load_giant(config["base_checkpoint"]).cuda().eval()
        event("official_weights_loaded", strict=True, thermal_adapter=False)
        active = {}

        def inspect_forward(module, inputs, kwargs):
            shape = list(inputs[0].shape)
            if shape != active["expected"] or kwargs.get("thermal_mask") is not None:
                raise ValueError("Wrong number of model inputs or unexpected thermal adapter")
            active["shapes"].append(shape)

        hook = model.register_forward_pre_hook(inspect_forward, with_kwargs=True)
        options = SimpleNamespace(**config, rgb_detail_sigma=0)
        for original in config["scenes"]:
            normalized = processed_images(original, config["process_res"])
            if hashlib.sha256(normalized.numpy().tobytes()).hexdigest() != config["reference_input_sha256"][original["scene"]]:
                raise ValueError(f"Original processed input changed: {original['scene']}")
            for mode, modality in MODES.items():
                scene = scene_for(original, modality)
                key = f"{mode}/{scene['scene']}"
                destination = output_folder(config, progress, mode) / scene["scene"]
                if key in progress["completed"]:
                    verify_completed(destination, progress["completed"][key])
                    continue
                if destination.exists():
                    raise FileExistsError(destination)
                stage = record_dir / "staging" / key
                if stage.exists():
                    shutil.rmtree(stage)
                stage.mkdir(parents=True)
                raw = record_dir / "predictions" / key
                raw.mkdir(parents=True, exist_ok=True)
                started = time.monotonic()
                try:
                    expected = normalized[[v["source_view_index"] for v in scene["views"]]]
                    active.update(expected=[1, *list(expected.shape)], shapes=[])
                    arrays, images, statistics = infer_scene(model, scene, "official", options)
                    if statistics["input_sha256"] != hashlib.sha256(expected.numpy().tobytes()).hexdigest() or len(active["shapes"]) != 1:
                        raise ValueError(f"Input subset or forward count mismatch: {key}")
                    np.savez_compressed(raw / "predictions.npz", **arrays)
                    if modality:
                        details = export_single(stage, arrays, images, scene, config["seed"], config["max_points_per_view"], raw)
                    else:
                        details = {"cloud_counts": export_clouds(stage, arrays, images, scene["views"], config["seed"], config["max_points_per_view"]),
                                   "depth_color_range": export_depths(stage, arrays["depth"], scene, mode)}
                    result = {**statistics, **details, "observed_forward_shapes": list(active["shapes"]),
                              "raw_sha256": {p.name: file_hash(p) for p in raw.glob("*.npz")},
                              "files": fingerprints(stage), "total_seconds": time.monotonic() - started}
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    stage.rename(destination)
                    progress["completed"][key] = result
                    progress["failed"].pop(key, None)
                    atomic_json(progress_path, progress)
                    event("scene_complete", mode=mode, scene=scene["scene"], completed=len(progress["completed"]),
                          expected=3 * len(config["scenes"]), **statistics, total_seconds=result["total_seconds"])
                    del arrays, images
                except Exception:
                    progress["status"] = "incomplete"
                    progress["failed"][key] = traceback.format_exc()
                    atomic_json(progress_path, progress)
                    event("scene_failed", key=key, traceback=traceback.format_exc())
                    raise
                gc.collect()
                torch.cuda.empty_cache()
            del normalized
        hook.remove()
        del model
        gc.collect()
        torch.cuda.empty_cache()
    event("validation_start", runs=len(progress["completed"]))
    validation = validate(config, progress, record_dir, event)
    atomic_json(record_dir / "validation.json", validation)
    if not args.verify_only:
        for mode, modality in MODES.items():
            if mode in progress["published_modes"]:
                continue
            stage = Path(config["staging_output_root"]) / mode
            description = "12 张 RGB + 12 张红外联合输入" if modality is None else f"仅 12 张 {'RGB' if modality == 'rgb' else '红外'}，独立推理"
            (stage / "README.md").write_text(
                f"# 官方 DA3：{description}\n\n"
                f"官方 DA3-GIANT-1.1；60 场景；沿用原选定的 12 对图像。\n\n"
                f"输入：{description}。BF16、518、saddle_balanced；无适配器，无外部相机或深度标签。\n\n"
                f"每个场景提供 PLY 点云、depth.png 总览、depth/ 下逐帧深度图。\n\n"
                f"各次推理使用独立预测坐标和相对深度尺度。\n\n"
                f"运行记录和原始预测：{record_dir}\n")
            destination = Path(config["output_root"]) / mode
            if destination.exists():
                archive = record_dir / "previous_outputs" / mode
                archive.parent.mkdir(parents=True, exist_ok=True)
                if archive.exists():
                    raise FileExistsError(archive)
                destination.rename(archive)
                event("previous_output_archived", mode=mode, archive=str(archive))
            stage.rename(destination)
            progress["published_modes"].append(mode)
            atomic_json(progress_path, progress)
            event("mode_published", mode=mode, destination=str(destination))
        progress["status"] = "complete"
        atomic_json(progress_path, progress)
    event("complete", **validation)


if __name__ == "__main__":
    main()
