#!/usr/bin/env python3
"""Run a low-light scene with the weights and export procedure of a saved comparison."""
import argparse
from datetime import datetime
import gc
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
from types import SimpleNamespace
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import torch

from depth_anything_3.thermal.adapter import load_adapter, load_giant
from depth_anything_3.utils.logger import logger
from infer_sear_comparison import (
    atomic_json, export_clouds, export_depths, file_hash, fingerprints,
    infer_scene, verify_completed,
)
from refine_all_multiview_detail import initialize_worker, process_scene


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-config", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--record-dir", type=Path, required=True)
    parser.add_argument("--scene", default="ThermalGaussian/Ebike")
    parser.add_argument("--rgb-root", type=Path, help="RGB variant directory; defaults to dataset/rgb_lowlight.")
    parser.add_argument("--lowlight-manifest", type=Path, help="Manifest belonging to the RGB variant.")
    args = parser.parse_args()
    dataset, output, records = (p.resolve() for p in (args.dataset, args.output_root, args.record_dir))
    if output.exists() or records.exists():
        parser.error("Output and record directories must both be new.")
    if output in records.parents or records in output.parents or output == records:
        parser.error("Output and records must be separate.")
    if dataset in output.parents or dataset in records.parents or dataset == output or dataset == records:
        parser.error("Results must stay outside the input dataset.")
    reference = json.loads(args.reference_config.read_text())
    reference_progress = json.loads((args.reference_config.parent / "progress.json").read_text())
    if reference_progress["status"] != "complete":
        raise ValueError("Reference comparison is incomplete.")
    if file_hash(reference["adapter"]) != reference["adapter_sha256"]:
        raise ValueError("Adapter weights differ from the reference comparison.")
    # These are the actual postprocessing implementations used in the reference.
    for name in ("scripts/refine_multiview_detail.py", "scripts/refine_all_multiview_detail.py",
                 "src/depth_anything_3/thermal/multiview_detail.py",
                 "src/depth_anything_3/thermal/detail_refinement.py"):
        if file_hash(ROOT / name) != reference["code_sha256"][name]:
            raise ValueError(f"Reference postprocessing code changed: {name}")
    source_scene = next(scene for scene in reference["scenes"] if scene["scene"] == args.scene)
    rgb_root = (args.rgb_root or dataset / "rgb_lowlight").resolve()
    manifest_path = (args.lowlight_manifest or dataset / "lowlight_processing/manifest.json").resolve()
    manifest = json.loads(manifest_path.read_text())
    by_name = {Path(entry["relative_path"]).name: entry for entry in manifest["images"]}
    if len(by_name) != len(manifest["images"]):
        raise ValueError("Low-light filenames must be unique across splits.")
    views, inputs = [], []
    for view in source_scene["views"]:
        name = Path(view["image"]).name
        if view["modality"] == "rgb":
            entry = by_name[name]
            path = rgb_root / entry["relative_path"]
            if file_hash(path) != entry["output_sha256"]:
                raise ValueError(f"Low-light RGB does not match its manifest: {path}")
        else:
            path = dataset / "VGGT/thermal" / name
            if file_hash(path) != file_hash(view["image"]):
                raise ValueError(f"Thermal input differs from reference: {path}")
        views.append({**view, "image": str(path)})
        inputs.append({**views[-1], "sha256": file_hash(path)})
    scene = {**source_scene, "transforms": str(dataset / "VGGT/transforms.json"), "views": views}
    if file_hash(scene["transforms"]) != file_hash(source_scene["transforms"]):
        raise ValueError("Pair ordering/transforms differ from the reference.")
    if not torch.cuda.is_available():
        raise RuntimeError("DA3 Giant inference requires CUDA.")
    output.mkdir(parents=True)
    records.mkdir(parents=True)
    config = {key: reference[key] for key in
              ("base_checkpoint", "adapter", "adapter_sha256", "pairs", "process_res", "max_points_per_view", "seed")}
    code_paths = list(reference["code_sha256"]) + [
        "scripts/run_ebike_lowlight_joint_detail.py", "src/depth_anything_3/utils/io/input_processor.py",
    ]
    config.update({
        "data_root": str(dataset.parent), "dataset": str(dataset), "output_root": str(output),
        "raw_output_root": str(records / "raw_outputs"), "scenes": [scene],
        "reference_config": str(args.reference_config.resolve()),
        "reference_config_sha256": file_hash(args.reference_config),
        "reference_output_root": reference["output_root"],
        "rgb_root": str(rgb_root), "thermal_root": str(dataset / "VGGT/thermal"),
        "lowlight_manifest": str(manifest_path), "lowlight_manifest_sha256": file_hash(manifest_path),
        "input_files": inputs, "method": reference["method"],
        "ref_view_strategy": "saddle_balanced", "rgb_detail_sigma": 0,
        "code_sha256": {name: file_hash(ROOT / name) for name in code_paths},
        "environment": {"torch": torch.__version__, "numpy": np.__version__,
                        "physical_gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
                        "gpu": torch.cuda.get_device_name(0)},
    })
    config_path = records / "run_config.json"
    atomic_json(config_path, config)
    for name in code_paths:
        destination = records / "code" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, destination)
    progress = {"status": "running", "completed": {}, "cached": {}, "failed": {}}
    progress_path = records / "progress.json"
    atomic_json(progress_path, progress)

    def event(name, **values):
        record = {"time": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(), "event": name, **values}
        with (records / "run.log").open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)

    torch.set_num_threads(8)
    logger.level = 1
    started = time.monotonic()
    try:
        event("start", scene=scene["scene"], pairs=config["pairs"], gpu=config["environment"])
        model = load_giant(config["base_checkpoint"]).cuda().eval()
        event("base_weights_loaded", checkpoint=config["base_checkpoint"])
        infer_args = SimpleNamespace(seed=config["seed"], process_res=config["process_res"],
                                     rgb_detail_sigma=0, ref_view_strategy="saddle_balanced")
        raw_records = {}
        for mode in ("official", "new"):
            if mode == "new":
                checkpoint = torch.load(config["adapter"], map_location="cpu", weights_only=True)
                if Path(checkpoint["base_checkpoint"]).resolve() != Path(config["base_checkpoint"]).resolve():
                    raise ValueError("Adapter base checkpoint mismatch.")
                load_adapter(model, checkpoint)
                model.eval()
                event("adapter_loaded", path=config["adapter"], epoch=checkpoint["epoch"], step=checkpoint["step"])
                del checkpoint
            arrays, images, statistics = infer_scene(model, scene, mode, infer_args)
            key = mode + "/" + scene["scene"]
            expected_shape = reference_progress["completed"]["official/" + scene["scene"]]["shape"]
            if statistics["shape"] != expected_shape:
                raise ValueError("Processed image shape differs from reference.")
            if mode == "new" and statistics["input_sha256"] != raw_records["official"]["input_sha256"]:
                raise ValueError("Official/new inputs differ.")
            cache = records / "cache" / key / "predictions.npz"
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cache, **arrays)
            raw_output = Path(config["raw_output_root"]) / key
            raw_output.mkdir(parents=True)
            counts = export_clouds(raw_output, arrays, images, views, config["seed"], config["max_points_per_view"])
            color_range = export_depths(raw_output, arrays["depth"], scene, mode)
            record = {**statistics, "cloud_counts": counts, "depth_color_range": color_range,
                      "files": fingerprints(raw_output)}
            verify_completed(raw_output, record)
            raw_records[mode] = record
            progress["completed"][key] = record
            progress["cached"][key] = {"path": str(cache), "sha256": file_hash(cache), **statistics}
            atomic_json(progress_path, progress)
            event("inference_complete", mode=mode, **statistics, points=counts)
            del arrays, images
            gc.collect()
        del model
        gc.collect()
        torch.cuda.empty_cache()
        event("joint_detail_start")
        task = {
            "scene": scene, "base_config": {**config, "output_root": config["raw_output_root"]},
            "config_path": str(config_path),
            "base": progress["cached"]["new/" + scene["scene"]]["path"],
            "teacher": progress["cached"]["official/" + scene["scene"]]["path"],
            "stage_output": str(records / "staging/new" / scene["scene"] / "output"),
            "stage_record": str(records / "staging/new" / scene["scene"] / "record"),
            "log": str(records / "logs" / (scene["scene"] + ".log")), "source_records": raw_records,
        }
        initialize_worker()
        result = process_scene(task)
        if result["record"]["parameters"] != reference_progress["completed"]["new/" + scene["scene"]]["parameters"]:
            raise ValueError("Joint detail parameters differ from reference.")
        official = output / "official" / scene["scene"]
        official.parent.mkdir(parents=True)
        shutil.copytree(Path(config["raw_output_root"]) / "official" / scene["scene"], official)
        final = output / "new" / scene["scene"]
        final.parent.mkdir(parents=True)
        Path(task["stage_output"]).rename(final)
        final_record = records / "scenes/new" / scene["scene"]
        final_record.parent.mkdir(parents=True)
        Path(task["stage_record"]).rename(final_record)
        progress["completed"]["new/" + scene["scene"]] = {
            **result["record"], "validation": result["audit"], "record_dir": str(final_record),
        }
        for mode in ("official", "new"):
            key = mode + "/" + scene["scene"]
            verify_completed(output / key, progress["completed"][key])
            expected = {item["path"] for item in reference_progress["completed"][key]["files"]}
            actual = {item["path"] for item in progress["completed"][key]["files"]}
            if actual != expected:
                raise ValueError(f"Output filenames differ from reference: {mode}")
        for entry in config["input_files"]:
            if file_hash(entry["image"]) != entry["sha256"]:
                raise ValueError("An input file changed during inference.")
        validation = {
            "status": "passed", "output_root": str(output), "scene": scene["scene"],
            "files": sum(len(r["files"]) for r in progress["completed"].values()),
            "png_files": 50, "ply_files": 6,
            "same_weights_and_joint_detail_parameters": True, "same_pair_indices_and_export_filenames": True,
            "official_new_same_lowlight_inputs": True, "thermal_inputs_identical_to_reference": True,
            "input_files_unchanged": True, "joint_detail": result["audit"],
        }
        atomic_json(records / "output_validation.json", validation)
        progress["status"] = "complete"
        atomic_json(progress_path, progress)
        event("finished", status="complete", files=validation["files"],
              output=str(output), elapsed_seconds=time.monotonic() - started)
    except Exception:
        progress["status"] = "failed"
        progress["failed"][scene["scene"]] = {"traceback": traceback.format_exc()}
        atomic_json(progress_path, progress)
        event("failed", traceback=traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
