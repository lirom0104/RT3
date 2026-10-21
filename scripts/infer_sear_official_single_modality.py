#!/usr/bin/env python3
"""Add independent official RGB-only and thermal-only runs to a comparison.

Use the recorded paired scene selection and inference/export settings. Keep raw
predictions, export pixel indices, code snapshots and validation under artifacts.
"""

import argparse
from datetime import datetime
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
from types import SimpleNamespace
from zoneinfo import ZoneInfo

from infer_sear_comparison import (
    ROOT, InputProcessor, Image, Normalize, atomic_json, depth_to_world,
    file_hash, fingerprints, infer_scene, load_giant, logger, matplotlib,
    np, plt, torch, verify_completed, write_ply,
)


MODES = {"official_rgb_only": "rgb", "official_thermal_only": "thermal"}
PLY_DTYPE = np.dtype([
    ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
    ("red", "u1"), ("green", "u1"), ("blue", "u1"),
])


def select_scene(scene, modality):
    views = [{**view, "source_view_index": index}
             for index, view in enumerate(scene["views"])
             if view["modality"] == modality]
    if len(views) != 12 or [v["pair_index"] for v in views] != scene["selected_pair_indices"]:
        raise ValueError(f"Expected the original twelve paired views: {scene['scene']}/{modality}")
    return {**scene, "views": views}


def export_single(folder, arrays, images, scene, seed, max_points, record_folder):
    """Keep the paired run's filtering and original per-view sampling seeds."""
    modality = scene["views"][0]["modality"]
    points, colors, selected_pixels = [], [], {}
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
        rng = np.random.default_rng(seed + view["source_view_index"])
        selected = rng.choice(candidates, min(len(candidates), max_points), replace=False)
        selected_pixels[f"view_{i:02d}"] = selected.astype(np.int32)
        points.append(world.reshape(-1, 3)[selected])
        colors.append(images[i].reshape(-1, 3)[selected])
    write_ply(folder / f"{modality}.ply", np.concatenate(points), np.concatenate(colors))
    np.savez_compressed(record_folder / "export_indices.npz", **selected_pixels)

    low, high = np.percentile(arrays["depth"], [2, 98])
    if high <= low:
        high = low + max(1e-6, abs(float(low)) * 1e-6)
    norm, cmap = Normalize(low, high, clip=True), matplotlib.colormaps["turbo"]
    depth_folder = folder / "depth" / modality
    depth_folder.mkdir(parents=True)
    for i, view in enumerate(scene["views"]):
        name = f"{view['pair_index']:05d}_{Path(view['image']).stem}.png"
        colored = np.rint(cmap(norm(arrays["depth"][i]))[..., :3] * 255).astype(np.uint8)
        Image.fromarray(colored).save(depth_folder / name)
    figure, axes = plt.subplots(2, 2, figsize=(10, 7.2), squeeze=False)
    for axis, i in zip(axes.flat, np.linspace(0, 11, 4, dtype=int)):
        axis.imshow(arrays["depth"][i], cmap=cmap, norm=norm)
        axis.set_axis_off()
        axis.set_title(f"{modality.upper()} | pair {scene['views'][i]['pair_index']}")
    figure.suptitle(f"{scene['scene']} | official {modality}-only | 12 images")
    figure.subplots_adjust(left=.02, right=.98, top=.92, bottom=.10, wspace=.03, hspace=.14)
    cax = figure.add_axes([.25, .035, .5, .02])
    figure.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax,
                   orientation="horizontal", label="Relative depth")
    figure.savefig(folder / "depth.png", dpi=120)
    plt.close(figure)
    return {"cloud_counts": {modality: sum(len(x) for x in selected_pixels.values())},
            "depth_color_range": [float(low), float(high)]}


def processed_images(scene, process_res):
    normalized, _, _ = InputProcessor()([v["image"] for v in scene["views"]],
                                      process_res=process_res, sequential=True)
    return normalized[0] if normalized.ndim == 5 else normalized


def verify_run(config, progress, record_dir):
    """Check every output against its saved independent prediction and source."""
    checked_files, checked_points = 0, 0
    for mode, modality in MODES.items():
        for original in config["scenes"]:
            scene = select_scene(original, modality)
            key = f"{mode}/{scene['scene']}"
            record = progress["completed"][key]
            folder, raw = Path(config["output_root"]) / key, record_dir / "predictions" / key
            verify_completed(folder, record)
            expected_files = {f"{modality}.ply", "depth.png"} | {
                f"depth/{modality}/{v['pair_index']:05d}_{Path(v['image']).stem}.png"
                for v in scene["views"]}
            if {f["path"] for f in record["files"]} != expected_files:
                raise ValueError(f"Unexpected output files: {key}")
            for name, digest in record["raw_sha256"].items():
                if file_hash(raw / name) != digest:
                    raise ValueError(f"Raw prediction/index file changed: {key}/{name}")
            normalized = processed_images(scene, config["process_res"])
            if hashlib.sha256(normalized.numpy().tobytes()).hexdigest() != record["input_sha256"]:
                raise ValueError(f"Processed single-modality input changed: {key}")
            mean = normalized.new_tensor([.485, .456, .406])[None, :, None, None]
            std = normalized.new_tensor([.229, .224, .225])[None, :, None, None]
            images = (normalized * std + mean).clamp(0, 1).permute(0, 2, 3, 1).numpy()
            with (folder / f"{modality}.ply").open("rb") as stream:
                while True:
                    line = stream.readline()
                    if line.startswith(b"element vertex "):
                        vertices = int(line.split()[-1])
                    if line == b"end_header\n":
                        break
                    if not line:
                        raise ValueError(f"Invalid PLY header: {key}")
                cloud = np.fromfile(stream, dtype=PLY_DTYPE)
            if len(cloud) != vertices or vertices != record["cloud_counts"][modality]:
                raise ValueError(f"Invalid PLY count: {key}")
            offset = 0
            with np.load(raw / "predictions.npz") as arrays, np.load(raw / "export_indices.npz") as indices:
                if arrays["depth"].shape != (12, *normalized.shape[-2:]):
                    raise ValueError(f"Prediction shape changed: {key}")
                norm = Normalize(*record["depth_color_range"], clip=True)
                cmap = matplotlib.colormaps["turbo"]
                for i, view in enumerate(scene["views"]):
                    selected = indices[f"view_{i:02d}"]
                    world = depth_to_world(arrays["depth"][i], arrays["extrinsics"][i], arrays["intrinsics"][i])
                    actual = cloud[offset:offset + len(selected)]
                    xyz = np.column_stack([actual[n] for n in ("x", "y", "z")])
                    np.testing.assert_array_equal(xyz, world.reshape(-1, 3)[selected].astype(np.float32))
                    rgb = np.column_stack([actual[n] for n in ("red", "green", "blue")])
                    color = np.rint(images[i].reshape(-1, 3)[selected] * 255).astype(np.uint8)
                    np.testing.assert_array_equal(rgb, color)
                    name = f"{view['pair_index']:05d}_{Path(view['image']).stem}.png"
                    expected = np.rint(cmap(norm(arrays["depth"][i]))[..., :3] * 255).astype(np.uint8)
                    with Image.open(folder / "depth" / modality / name) as image:
                        np.testing.assert_array_equal(np.asarray(image), expected)
                    offset += len(selected)
            if offset != len(cloud) or record["observed_forward_shapes"] != [[1, *record["shape"]]]:
                raise ValueError(f"Invalid twelve-image forward/export record: {key}")
            checked_files += len(record["files"])
            checked_points += vertices
    for original in config["input_files"]:
        if file_hash(original["image"]) != original["sha256"]:
            raise ValueError(f"Source image changed: {original['image']}")
    source = json.loads(Path(config["source_config"]).read_text())
    source_progress = json.loads((Path(config["source_config"]).parent / "progress.json").read_text())
    for key, record in source_progress["completed"].items():
        verify_completed(Path(source["output_root"]) / key, record)
    return {"status": "passed", "scenes_per_mode": len(config["scenes"]),
            "runs": len(MODES) * len(config["scenes"]), "output_files": checked_files,
            "point_count": checked_points, "model_input_views_per_forward": 12,
            "checks": ["all point positions/colors match saved independent predictions and source pixels",
                       "all per-view depth PNGs match saved independent predictions",
                       "same paired indices, preprocessing and sampling seeds as original",
                       "exactly one twelve-image model forward per run",
                       "source images and original official/new files unchanged"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-config", type=Path, default=ROOT / "artifacts/comparison_all_scenes_20261010/run_config.json")
    parser.add_argument("--record-dir", type=Path, default=ROOT / "artifacts/comparison_official_single_modality_20261013")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    source_config, record_dir = args.source_config.resolve(), args.record_dir.resolve()
    source = json.loads(source_config.read_text())
    source_progress = json.loads((source_config.parent / "progress.json").read_text())
    if source_progress["status"] != "complete" or source_progress["failed"]:
        raise ValueError("Original comparison must be complete")
    output = Path(source["output_root"])
    scenes = source["scenes"]
    for scene in scenes:
        for modality in MODES.values():
            select_scene(scene, modality)
    if args.dry_run:
        print(json.dumps({"scenes": len(scenes), "runs": len(MODES) * len(scenes),
                          "input_views_per_run": 12, "output_directories": [str(output / m) for m in MODES]}, indent=2))
        return
    if record_dir.is_relative_to(output) or output.is_relative_to(record_dir):
        raise ValueError("Records must stay separate from outputs")
    progress_path = record_dir / "progress.json"
    config_path = record_dir / "run_config.json"
    if progress_path.exists():
        if not args.resume and not args.verify_only:
            raise FileExistsError("Use --resume or --verify-only for existing records")
        config = json.loads(config_path.read_text())
        if config["source_config"] != str(source_config) or config["source_config_sha256"] != file_hash(source_config):
            raise ValueError("Original configuration changed")
        for name, digest in config["code_sha256"].items():
            if file_hash(ROOT / name) != digest:
                raise ValueError(f"Run code changed: {name}")
        progress = json.loads(progress_path.read_text())
    else:
        if args.verify_only:
            raise FileNotFoundError(progress_path)
        for mode in MODES:
            if (output / mode).exists():
                raise FileExistsError(output / mode)
        record_dir.mkdir(parents=True)
        code_paths = ["scripts/infer_sear_official_single_modality.py", "scripts/infer_sear_comparison.py",
                      "src/depth_anything_3/thermal/adapter.py", "src/depth_anything_3/model/da3.py",
                      "src/depth_anything_3/model/dinov2/vision_transformer.py",
                      "src/depth_anything_3/utils/io/input_processor.py"]
        config = {k: source[k] for k in ("data_root", "base_checkpoint", "output_root", "pairs",
                                        "process_res", "max_points_per_view", "seed", "scenes")}
        config.update({"source_config": str(source_config), "source_config_sha256": file_hash(source_config),
                       "base_weights_sha256": file_hash(Path(source["base_checkpoint"]) / "model.safetensors"),
                       "modes": MODES, "ref_view_strategy": "saddle_balanced", "thermal_adapter": False,
                       "infer_gs": False, "source_cameras_provided": False,
                       "input_files": [{**v, "sha256": file_hash(v["image"])} for s in scenes for v in s["views"]],
                       "code_sha256": {name: file_hash(ROOT / name) for name in code_paths}})
        atomic_json(config_path, config)
        for name in code_paths:
            destination = record_dir / "code" / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, destination)
        progress = {"status": "running", "completed": {}, "failed": {}}
        atomic_json(progress_path, progress)
    torch.set_num_threads(8)
    logger.level = 1

    def event(name, **data):
        value = {"time": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(), "event": name, **data}
        with (record_dir / "run.log").open("a") as stream:
            stream.write(json.dumps(value, ensure_ascii=False) + "\n")
        print(json.dumps(value, ensure_ascii=False), flush=True)

    if not args.verify_only:
        if not torch.cuda.is_available():
            raise RuntimeError("Official DA3 Giant inference requires CUDA")
        if file_hash(Path(config["base_checkpoint"]) / "model.safetensors") != config["base_weights_sha256"]:
            raise ValueError("Official model weights changed")
        for key, record in source_progress["completed"].items():
            verify_completed(output / key, record)
        event("start", scenes=len(scenes), runs=2 * len(scenes),
              physical_gpu=os.environ.get("CUDA_VISIBLE_DEVICES"), gpu=torch.cuda.get_device_name(0))
        model = load_giant(config["base_checkpoint"]).cuda().eval()
        active = {}

        def inspect_forward(module, inputs, kwargs):
            shape = list(inputs[0].shape)
            if shape != [1, 12, 3, *active["hw"]] or kwargs.get("thermal_mask") is not None:
                raise ValueError("Expected twelve same-modality images and no adapter")
            active["shapes"].append(shape)

        hook = model.register_forward_pre_hook(inspect_forward, with_kwargs=True)
        infer_args = SimpleNamespace(seed=config["seed"], process_res=config["process_res"],
                                     ref_view_strategy="saddle_balanced", rgb_detail_sigma=0)
        for original in scenes:
            normalized = processed_images(original, config["process_res"])
            baseline = source_progress["completed"]["official/" + original["scene"]]
            if hashlib.sha256(normalized.numpy().tobytes()).hexdigest() != baseline["input_sha256"]:
                raise ValueError(f"Original processed input changed: {original['scene']}")
            for mode, modality in MODES.items():
                scene = select_scene(original, modality)
                key = f"{mode}/{scene['scene']}"
                destination = output / key
                if key in progress["completed"]:
                    verify_completed(destination, progress["completed"][key])
                    continue
                if destination.exists():
                    raise FileExistsError(destination)
                active.update(hw=list(normalized.shape[-2:]), shapes=[])
                expected = normalized[[v["source_view_index"] for v in scene["views"]]]
                expected_hash = hashlib.sha256(expected.numpy().tobytes()).hexdigest()
                arrays, images, statistics = infer_scene(model, scene, "official", infer_args)
                if statistics["input_sha256"] != expected_hash or len(active["shapes"]) != 1:
                    raise ValueError(f"Single-modality preprocessing/forward differs from reference: {key}")
                stage = record_dir / "staging" / key
                if stage.exists():
                    shutil.rmtree(stage)
                stage.mkdir(parents=True)
                raw = record_dir / "predictions" / key
                raw.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(raw / "predictions.npz", **arrays)
                details = export_single(stage, arrays, images, scene, config["seed"], config["max_points_per_view"], raw)
                record = {**statistics, **details, "modality": modality, "input_views": scene["views"],
                          "selected_pair_indices": scene["selected_pair_indices"],
                          "observed_forward_shapes": active["shapes"], "matches_original_input_subset": True,
                          "raw_sha256": {name: file_hash(raw / name) for name in ("predictions.npz", "export_indices.npz")},
                          "files": fingerprints(stage)}
                if len(record["files"]) != 14:
                    raise ValueError(f"Expected one PLY and thirteen PNGs: {key}")
                destination.parent.mkdir(parents=True, exist_ok=True)
                stage.rename(destination)
                progress["completed"][key] = record
                atomic_json(progress_path, progress)
                event("scene_complete", mode=mode, scene=scene["scene"], completed=len(progress["completed"]),
                      points=details["cloud_counts"], shape=statistics["shape"])
                del arrays, images
                gc.collect()
            del normalized
        hook.remove()
        del model
        gc.collect()
        torch.cuda.empty_cache()
    event("validation_start", runs=len(progress["completed"]))
    validation = verify_run(config, progress, record_dir)
    atomic_json(record_dir / "validation.json", validation)
    progress["status"] = "complete"
    atomic_json(progress_path, progress)
    event("complete", **validation)


if __name__ == "__main__":
    main()
