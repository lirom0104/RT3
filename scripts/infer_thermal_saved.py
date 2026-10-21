#!/usr/bin/env python3
"""Run a Giant adapter on exact saved comparison inputs, preserving their order."""

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from PIL import Image
import torch

from depth_anything_3.thermal.adapter import load_adapter, load_giant
from depth_anything_3.utils.io.input_processor import InputProcessor


def read_case(folder):
    meta = json.loads((folder / "selection.json").read_text())
    with np.load(folder / "predictions.npz") as data:
        arrays = {key: data[key] for key in data.files}
    count = len(meta["selected_pair_indices"])
    assert arrays["images"].shape[:2] == (2 * count, 3)
    np.testing.assert_array_equal(arrays["source_pair_indices"], meta["selected_pair_indices"])
    np.testing.assert_array_equal(arrays["thermal_mask"], [False] * count + [True] * count)
    digest = hashlib.sha256(arrays["images"].tobytes()).hexdigest()
    assert digest == meta["input_rgb_sha256"]
    for i, view in enumerate(meta["selected_views"]):
        source = np.asarray(Image.open(view["source"]).convert("RGB"), dtype=np.float32) / 255
        np.testing.assert_array_equal(source.transpose(2, 0, 1), arrays["images"][i])
    return meta, arrays


def predict(model, meta, arrays, adapted):
    torch.manual_seed(meta["seed"])
    images = InputProcessor.NORMALIZE(torch.from_numpy(arrays["images"]))[None].cuda()
    mask = torch.from_numpy(arrays["thermal_mask"])[None].cuda() if adapted else None
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        raw = model(images, thermal_mask=mask, infer_gs=False, use_ray_pose=False,
                    ref_view_strategy=meta["ref_view_strategy"])
    torch.cuda.synchronize()
    timing = {"inference_seconds": time.perf_counter() - start,
              "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
              "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30}
    mapping = {"depths": "depth", "depth_confidence": "depth_conf",
               "extrinsics_world2cam": "extrinsics", "intrinsics": "intrinsics"}
    result = {name: raw[key][0].float().cpu().numpy() for name, key in mapping.items()}
    for name, value in result.items():
        assert value.shape == arrays[name].shape, (name, value.shape, arrays[name].shape)
        assert np.isfinite(value).all(), name
    assert (result["depths"] > 0).all()
    assert (result["intrinsics"][:, [0, 1], [0, 1]] > 0).all()
    for name in ("images", "thermal_mask", "source_pair_indices"):
        result[name] = arrays[name]
    del raw, images, mask
    return result, timing


def export_case(folder, arrays, metadata):
    folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(folder / "predictions.npz", **arrays)
    with np.load(folder / "predictions.npz") as saved:
        for name, value in arrays.items():
            np.testing.assert_array_equal(saved[name], value)
    pred = folder / "predictions"
    for name in ("images", "depths", "confidence"):
        (pred / name).mkdir(parents=True, exist_ok=True)
    h, w = arrays["images"].shape[-2:]
    frames = []
    for i, view in enumerate(metadata["selected_views"]):
        stem = f"{i:06d}_{view['modality']}_{Path(view['source']).stem}"
        image_file, depth_file, conf_file = (f"images/{stem}.png", f"depths/{stem}.npy",
                                           f"confidence/{stem}.npy")
        Image.fromarray((arrays["images"][i].transpose(1, 2, 0) * 255).astype(np.uint8)).save(pred / image_file)
        np.save(pred / depth_file, arrays["depths"][i])
        np.save(pred / conf_file, arrays["depth_confidence"][i])
        k = arrays["intrinsics"][i]
        frames.append({view["modality"]: {
            "transform_matrix": arrays["extrinsics_world2cam"][i].tolist(),
            "fl_x": float(k[0, 0]), "fl_y": float(k[1, 1]),
            "cx": float(k[0, 2]), "cy": float(k[1, 2]), "w": w, "h": h,
            "file_path": image_file, "depth_file_path": depth_file,
            "confidence_file_path": conf_file, "source": view["source"],
            "source_pair_index": view["frame_index"]}})
    (pred / "transforms.json").write_text(json.dumps({
        "type": metadata["method"], "transform_matrix_convention": "world_to_camera_opencv",
        "frames": frames}, indent=2))
    metadata.update(status="complete", output_round_trip_verified=True,
                    depth_range=[float(arrays["depths"].min()), float(arrays["depths"].max())])
    (folder / "selection.json").write_text(json.dumps(metadata, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--baseline-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--verify-baseline", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(8)
    args.output_root.mkdir(parents=True, exist_ok=True)
    for folder in args.baseline_dirs:
        if (args.output_root / folder.name).resolve() == folder.resolve():
            raise ValueError("Output must not overwrite the archived baseline")
        if (args.output_root / folder.name / "predictions.npz").exists():
            raise FileExistsError(args.output_root / folder.name)
    state = torch.load(args.adapter, map_location="cpu", weights_only=True)
    checkpoint_hash = hashlib.sha256(args.adapter.read_bytes()).hexdigest()
    print(f"Loading Giant; adapter epoch={state['epoch']} step={state['step']}; GPU={os.environ.get('CUDA_VISIBLE_DEVICES')}", flush=True)
    model = load_giant(state["base_checkpoint"]).cuda().eval()
    checks = {}
    if args.verify_baseline:
        for folder in args.baseline_dirs:
            meta, old = read_case(folder)
            fresh, _ = predict(model, meta, old, adapted=False)
            check = {}
            for key in ("depths", "depth_confidence", "extrinsics_world2cam", "intrinsics"):
                delta = np.abs(fresh[key] - old[key])
                check[key] = {"max_abs_difference": float(delta.max()),
                              "mean_abs_difference": float(delta.mean()),
                              "exactly_equal": bool(np.array_equal(fresh[key], old[key]))}
            checks[folder.name] = check
            print(f"Archived baseline replay {folder.name}: {json.dumps(check)}", flush=True)
            del old, fresh
        (args.output_root / "baseline_replay.json").write_text(json.dumps(checks, indent=2))
    load_adapter(model, state)
    model.eval()
    for folder in args.baseline_dirs:
        meta, old = read_case(folder)
        arrays, timing = predict(model, meta, old, adapted=True)
        metadata = {key: meta[key] for key in (
            "seed", "scene", "total_pairs", "selected_pair_indices", "selected_views",
            "input_rgb_sha256", "ref_view_strategy")}
        metadata.update({
            "method": f"DA3-GIANT-1.1 + thermal adapter (best epoch {state['epoch']})",
            "model_name": "da3-giant", "weight_directory": state["base_checkpoint"],
            "adapter": str(args.adapter.resolve()), "adapter_sha256": checkpoint_hash,
            "adapter_epoch": state["epoch"], "adapter_step": state["step"],
            "adapter_config": state["adapter_config"], "adapter_tensor_count": len(state["adapter"]),
            "strict_weight_loading": True, "baseline_predictions": str((folder / "predictions.npz").resolve()),
            "input_shape": list(arrays["images"].shape),
            "input_preprocessing": "Exact archived [0,1] float32 pixels; ImageNet normalization once",
            "input_pixels_verified_identical_to_baseline_and_source": True,
            "modality_mask_passed_to_model": True, "ground_truth_inputs": [],
            "infer_gs": False, "use_ray_pose": False, "autocast_dtype": "bfloat16", "is_metric": 0,
            "torch_version": str(torch.__version__), "cuda_version": torch.version.cuda,
            "physical_gpu": "CUDA_VISIBLE_DEVICES=" + os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "gpu_name": torch.cuda.get_device_name(0),
            "timing": "One synchronized forward, excludes loading and export; after baseline replay if enabled; not a performance benchmark",
            "baseline_replay": checks.get(folder.name), **timing})
        export_case(args.output_root / folder.name, arrays, metadata)
        print(f"Saved {args.output_root / folder.name}: {json.dumps(timing)}", flush=True)
        del arrays, old
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
