#!/usr/bin/env python3
"""Replay the original paired comparison using both DATA-DEAL best adapters."""

import argparse
from datetime import datetime
import gc
import json
import os
from pathlib import Path
import shutil
import time
import traceback
from types import SimpleNamespace
from zoneinfo import ZoneInfo

from infer_sear_comparison import (
    ROOT, Image, atomic_json, export_clouds, export_depths, file_hash,
    fingerprints, infer_scene, load_adapter, load_giant, logger, np,
    torch, verify_completed,
)


MODELS = {
    "new_data_deal_shared": ("thermal_giant_data_deal", "shared"),
    "new_data_deal_ir_only": ("thermal_giant_data_deal_ir_only", "thermal_only"),
}
PLY_DTYPE = np.dtype([(n, "<f4") for n in ("x", "y", "z")] +
                     [(n, "u1") for n in ("red", "green", "blue")])


def validate_outputs(config, progress, record_dir):
    files, points = 0, 0
    for mode in MODELS:
        for scene in config["scenes"]:
            key = f"{mode}/{scene['scene']}"
            record = progress["completed"][key]
            folder = Path(config["output_root"]) / key
            verify_completed(folder, record)
            expected = {"rgb.ply", "thermal.ply", "combined.ply", "depth.png"} | {
                f"depth/{v['modality']}/{v['pair_index']:05d}_{Path(v['image']).stem}.png"
                for v in scene["views"]}
            if {f["path"] for f in record["files"]} != expected:
                raise ValueError(f"Unexpected file list: {key}")
            for path in folder.rglob("*.png"):
                with Image.open(path) as image:
                    image.load()
                    if path.name != "depth.png" and image.size != tuple(record["shape"][-2:][::-1]):
                        raise ValueError(f"Depth image dimensions: {path}")
            clouds = {}
            for modality in ("rgb", "thermal", "combined"):
                with (folder / f"{modality}.ply").open("rb") as stream:
                    vertices = None
                    while True:
                        line = stream.readline()
                        if line.startswith(b"element vertex "):
                            vertices = int(line.split()[-1])
                        if line == b"end_header\n":
                            break
                        if not line:
                            raise ValueError(f"Invalid PLY header: {key}")
                    cloud = np.fromfile(stream, dtype=PLY_DTYPE)
                expected_count = (sum(record["cloud_counts"].values()) if modality == "combined"
                                  else record["cloud_counts"][modality])
                if vertices != len(cloud) or len(cloud) != expected_count or not vertices:
                    raise ValueError(f"Invalid point count: {key}/{modality}")
                if not all(np.isfinite(cloud[n]).all() for n in ("x", "y", "z")):
                    raise ValueError(f"Nonfinite point: {key}/{modality}")
                clouds[modality] = cloud
                points += len(cloud)
            np.testing.assert_array_equal(clouds["combined"], np.concatenate([clouds["rgb"], clouds["thermal"]]))
            raw = record_dir / "predictions" / key / "predictions.npz"
            if file_hash(raw) != record["predictions_sha256"]:
                raise ValueError(f"Raw predictions changed: {key}")
            files += len(record["files"])
    for entry in config["input_files"]:
        if file_hash(entry["image"]) != entry["sha256"]:
            raise ValueError(f"Input image changed: {entry['image']}")
    verify_completed(Path(config["reference_output"]), {"files": config["reference_files"]})
    return {"status": "passed", "scenes_per_model": len(config["scenes"]),
            "completed_runs": len(progress["completed"]), "output_files": files,
            "point_count_including_combined": points, "reference_outputs_unchanged": True,
            "checks": ["processed input hashes match original new run for every scene",
                       "all output file hashes and exact standard filenames",
                       "PNG decoding and per-view dimensions",
                       "PLY counts, finite coordinates and exact combined concatenation",
                       "raw prediction hashes, source image hashes, original output hashes"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-config", type=Path,
                        default=ROOT / "artifacts/comparison_all_scenes_20261010/run_config.json")
    parser.add_argument("--record-dir", type=Path,
                        default=ROOT / "artifacts/comparison_data_deal_20261018")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    source_path, record_dir = args.source_config.resolve(), args.record_dir.resolve()
    source = json.loads(source_path.read_text())
    source_progress = json.loads((source_path.parent / "progress.json").read_text())
    if source_progress["status"] != "complete" or source_progress["failed"]:
        raise ValueError("Reference comparison is incomplete")
    output = Path(source["output_root"])
    progress_path, config_path = record_dir / "progress.json", record_dir / "run_config.json"
    if record_dir.is_relative_to(output) or output.is_relative_to(record_dir):
        raise ValueError("Records must be outside comparison outputs")
    torch.set_num_threads(8)
    logger.level = 1
    if progress_path.exists():
        if not args.resume and not args.verify_only:
            raise FileExistsError("Use --resume for this existing run")
        config, progress = json.loads(config_path.read_text()), json.loads(progress_path.read_text())
        if config["source_config_sha256"] != file_hash(source_path):
            raise ValueError("Reference configuration changed")
        for name, digest in config["code_sha256"].items():
            if file_hash(ROOT / name) != digest:
                raise ValueError(f"Run code changed: {name}")
    else:
        if args.verify_only:
            raise FileNotFoundError(progress_path)
        for mode in MODELS:
            if (output / mode).exists():
                raise FileExistsError(output / mode)
        record_dir.mkdir(parents=True, exist_ok=True)
        models = {}
        for mode, (experiment, scope) in MODELS.items():
            path = ROOT / "checkpoints" / experiment / "adapter_best.pt"
            state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
            if state["adapter_config"]["lora_scope"] != scope:
                raise ValueError(f"Wrong adapter scope: {mode}")
            if Path(state["base_checkpoint"]).resolve() != Path(source["base_checkpoint"]).resolve():
                raise ValueError(f"Wrong base checkpoint: {mode}")
            models[mode] = {"adapter": str(path), "sha256": file_hash(path),
                            "epoch": state["epoch"], "step": state["step"],
                            "adapter_config": state["adapter_config"]}
            del state
        code_paths = ["scripts/infer_data_deal_comparison.py", "scripts/infer_sear_comparison.py",
                      "src/depth_anything_3/thermal/adapter.py", "src/depth_anything_3/model/da3.py",
                      "src/depth_anything_3/model/dinov2/vision_transformer.py",
                      "src/depth_anything_3/model/dinov2/layers/attention.py",
                      "src/depth_anything_3/model/dinov2/layers/block.py",
                      "src/depth_anything_3/model/dinov2/layers/swiglu_ffn.py",
                      "src/depth_anything_3/utils/io/input_processor.py"]
        config = {k: source[k] for k in ("data_root", "base_checkpoint", "output_root", "pairs",
                                        "process_res", "max_points_per_view", "seed", "scenes")}
        config.update({"source_config": str(source_path), "source_config_sha256": file_hash(source_path),
                       "models": models, "ref_view_strategy": source.get("ref_view_strategy", "saddle_balanced"),
                       "rgb_detail_sigma": 0, "infer_gs": False, "source_cameras_provided": False,
                       "base_weights_sha256": file_hash(Path(source["base_checkpoint"]) / "model.safetensors"),
                       "input_files": [{"image": v["image"], "sha256": file_hash(v["image"])}
                                       for s in source["scenes"] for v in s["views"]],
                       "reference_output": str(output / "new"), "reference_files": fingerprints(output / "new"),
                       "reference_input_sha256": {s["scene"]: source_progress["completed"]["new/" + s["scene"]]["input_sha256"]
                                                  for s in source["scenes"]},
                       "code_sha256": {name: file_hash(ROOT / name) for name in code_paths}})
        atomic_json(config_path, config)
        for name in code_paths:
            dest = record_dir / "code" / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, dest)
        progress = {"status": "running", "completed": {}, "failed": {}}
        atomic_json(progress_path, progress)

    def event(name, **values):
        entry = {"time": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(), "event": name, **values}
        with (record_dir / "run.log").open("a") as stream:
            stream.write(json.dumps(entry, ensure_ascii=False) + "\n")
        print(json.dumps(entry, ensure_ascii=False), flush=True)

    if not args.verify_only:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required")
        if file_hash(Path(config["base_checkpoint"]) / "model.safetensors") != config["base_weights_sha256"]:
            raise ValueError("Base weights changed")
        options = SimpleNamespace(**config)
        event("start", scenes=len(config["scenes"]), models=config["models"],
              physical_gpu=os.environ.get("CUDA_VISIBLE_DEVICES"), gpu=torch.cuda.get_device_name(0))
        for mode in MODELS:
            info = config["models"][mode]
            if file_hash(info["adapter"]) != info["sha256"]:
                raise ValueError(f"Adapter weights changed: {mode}")
            model = load_giant(config["base_checkpoint"]).cuda().eval()
            load_adapter(model, info["adapter"])
            model.eval()
            event("adapter_loaded", mode=mode, epoch=info["epoch"], step=info["step"],
                  lora_scope=model.thermal_adapter_config["lora_scope"])
            for index, scene in enumerate(config["scenes"], 1):
                key = f"{mode}/{scene['scene']}"
                destination = output / key
                if key in progress["completed"]:
                    verify_completed(destination, progress["completed"][key])
                    continue
                if destination.exists():
                    raise FileExistsError(destination)
                staging = record_dir / "staging" / key
                if staging.exists():
                    shutil.rmtree(staging)
                staging.mkdir(parents=True)
                try:
                    started = time.monotonic()
                    arrays, images, statistics = infer_scene(model, scene, "new", options)
                    if statistics["input_sha256"] != config["reference_input_sha256"][scene["scene"]]:
                        raise ValueError("Input pixels differ from reference comparison")
                    raw = record_dir / "predictions" / key / "predictions.npz"
                    raw.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(raw, **arrays)
                    counts = export_clouds(staging, arrays, images, scene["views"], config["seed"], config["max_points_per_view"])
                    color_range = export_depths(staging, arrays["depth"], scene, mode)
                    result = {**statistics, "cloud_counts": counts, "depth_color_range": color_range,
                              "files": fingerprints(staging), "predictions_sha256": file_hash(raw),
                              "total_seconds": time.monotonic() - started}
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    staging.rename(destination)
                    progress["completed"][key] = result
                    progress["failed"].pop(key, None)
                    atomic_json(progress_path, progress)
                    event("scene_complete", mode=mode, scene=scene["scene"], index=index,
                          total=len(config["scenes"]), **statistics, total_seconds=result["total_seconds"])
                    del arrays, images
                except Exception:
                    progress["status"] = "incomplete"
                    progress["failed"][key] = traceback.format_exc()
                    atomic_json(progress_path, progress)
                    event("scene_failed", key=key, traceback=traceback.format_exc())
                    raise
                gc.collect()
                torch.cuda.empty_cache()
            del model
            gc.collect()
            torch.cuda.empty_cache()
    event("validating", completed=len(progress["completed"]))
    validation = validate_outputs(config, progress, record_dir)
    atomic_json(record_dir / "validation.json", validation)
    progress["status"] = "complete"
    atomic_json(progress_path, progress)
    event("finished", **validation)


if __name__ == "__main__":
    main()
