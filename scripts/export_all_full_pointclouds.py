#!/usr/bin/env python3
"""Add full-pixel PLYs to every published depth_pointcloud scene from its cache.

No model inference, confidence/depth/border/support masks, sampling, or deduplication.
Each processed prediction pixel produces one vertex. Existing files are verified,
never overwritten. Source records and validation live under artifacts.
"""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import sys
import time
import zipfile
from zoneinfo import ZoneInfo

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/depth_pointcloud"
DTYPE = np.dtype([(name, "<f4") for name in ("x", "y", "z")]
                 + [(name, "u1") for name in ("red", "green", "blue")])
GROUPS = {
    "official": ("official_three_modalities_20261019", "predictions/official", "official"),
    "official_rgb_only": ("official_three_modalities_20261019", "predictions/official_rgb_only", "official_rgb_only"),
    "official_thermal_only": ("official_three_modalities_20261019", "predictions/official_thermal_only", "official_thermal_only"),
    "sear_shared": ("comparison_joint_detail_all_20261012", "cache/new", "new"),
    "sear_ir_only": ("comparison_ir_only_best_all_20261016", "scenes/new", "new"),
    "data_deal_shared": ("comparison_data_deal_20261018", "predictions/new_data_deal_shared", "new_data_deal_shared"),
    "data_deal_ir_only": ("comparison_data_deal_20261018", "predictions/new_data_deal_ir_only", "new_data_deal_ir_only"),
}


def now():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            result.update(block)
    return result.hexdigest()


def atomic_json(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def header(count):
    return ("ply\nformat binary_little_endian 1.0\n"
            f"element vertex {count}\n"
            "property float x\nproperty float y\nproperty float z\n"
            "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n").encode("ascii")


def read_ply(path, count):
    expected = header(count)
    with path.open("rb") as stream:
        if stream.read(len(expected)) != expected:
            raise ValueError(f"Unexpected PLY header: {path}")
    if path.stat().st_size != len(expected) + count * DTYPE.itemsize:
        raise ValueError(f"Unexpected PLY byte count: {path}")
    return np.memmap(path, dtype=DTYPE, mode="r", offset=len(expected), shape=(count,))


def depth_shape(path):
    with zipfile.ZipFile(path) as archive, archive.open("depth.npy") as stream:
        version = np.lib.format.read_magic(stream)
        shape, _, _ = np.lib.format._read_array_header(stream, version)
    return tuple(shape)


def make_plan():
    jobs = []
    records = {}
    for group, (run, subdir, mode) in GROUPS.items():
        run_dir = ROOT / "artifacts" / run
        if run not in records:
            records[run] = (json.loads((run_dir / "run_config.json").read_text()),
                            json.loads((run_dir / "progress.json").read_text()))
        config, progress = records[run]
        for original in config["scenes"]:
            name = original["scene"]
            destination = OUTPUT / group / name
            views = original["views"]
            if group in ("official_rgb_only", "official_thermal_only"):
                modality = "rgb" if group == "official_rgb_only" else "thermal"
                views = [v for v in views if v["modality"] == modality]
            scene = {**original, "views": views}
            source = run_dir / subdir / name / "predictions.npz"
            if not source.is_file() or not (destination / "depth.png").is_file():
                raise FileNotFoundError(f"Missing scene or cache: {group}/{name}: {source}")
            for view in views:
                if not Path(view["image"]).is_file():
                    raise FileNotFoundError(view["image"])
            result = progress["completed"][f"{mode}/{name}"]
            source_hash = result.get("predictions_sha256") or result.get("raw_sha256", {}).get("predictions.npz")
            if group == "sear_shared":
                cached = progress["cached"][f"new/{name}"]
                source_hash = cached["sha256"]
                # Published sear_shared uses the original depths, not joint-detail depths.
                original_progress = ROOT / "artifacts/comparison_all_scenes_20261010/progress.json"
                result = json.loads(original_progress.read_text())["completed"][f"new/{name}"]
                if cached["input_sha256"] != result["input_sha256"]:
                    raise ValueError(f"Cache/input mismatch: {name}")
            shape = depth_shape(source)
            if len(shape) != 3 or shape[0] != len(views):
                raise ValueError(f"Cache/view shape mismatch: {source}")
            counts = {m: sum(v["modality"] == m for v in views) * int(np.prod(shape[1:]))
                      for m in ("rgb", "thermal") if any(v["modality"] == m for v in views)}
            if len(counts) == 2:
                counts["combined"] = int(np.prod(shape))
            if {p.stem for p in destination.glob("*.ply") if not p.stem.endswith("_full")} != set(counts):
                raise ValueError(f"Unexpected published modality files: {destination}")
            jobs.append({"key": f"{group}/{name}", "group": group, "scene": scene,
                         "destination": str(destination), "predictions": str(source),
                         "expected_prediction_sha256": source_hash,
                         "depth_shape": list(shape), "counts": counts,
                         "process_res": config["process_res"],
                         "input_sha256": result["input_sha256"],
                         "depth_color_range": result["depth_color_range"],
                         "depth_method": "saved joint-detail depth" if group == "sear_ir_only" else "saved model prediction"})
    expected = {f"{group}/{p.parent.relative_to(OUTPUT / group)}"
                for group in GROUPS for p in (OUTPUT / group).rglob("depth.png")}
    if {j["key"] for j in jobs} != expected:
        raise ValueError("Plan does not cover every published scene")
    return jobs


def initialize_worker():
    global torch, InputProcessor, depth_to_world, Image, matplotlib, Normalize
    sys.path.insert(0, str(ROOT / "scripts"))
    from infer_sear_comparison import (torch, InputProcessor, depth_to_world,
                                       Image, matplotlib, Normalize)
    torch.set_num_threads(1)


def export_scene(job):
    started = time.monotonic()
    source, destination = Path(job["predictions"]), Path(job["destination"])
    source_hash = digest(source)
    if job["expected_prediction_sha256"] and source_hash != job["expected_prediction_sha256"]:
        raise ValueError(f"Prediction hash mismatch: {source}")
    with np.load(source, allow_pickle=False) as saved:
        arrays = {key: saved[key] for key in ("depth", "extrinsics", "intrinsics")}
    if list(arrays["depth"].shape) != job["depth_shape"]:
        raise ValueError(f"Prediction shape changed: {source}")
    if not all(np.isfinite(value).all() for value in arrays.values()) or not (arrays["depth"] > 0).all():
        # Fail instead of silently dropping any pixels.
        raise ValueError(f"Nonfinite or nonpositive source geometry: {source}")
    views = job["scene"]["views"]
    normalized, _, _ = InputProcessor()([v["image"] for v in views],
                                      process_res=job["process_res"], sequential=True)
    if normalized.ndim == 5:
        normalized = normalized[0]
    if hashlib.sha256(normalized.numpy().tobytes()).hexdigest() != job["input_sha256"]:
        raise ValueError(f"Input pixels differ from archived inference: {job['key']}")
    mean = normalized.new_tensor([.485, .456, .406])[None, :, None, None]
    std = normalized.new_tensor([.229, .224, .225])[None, :, None, None]
    colors = (normalized * std + mean).clamp(0, 1).permute(0, 2, 3, 1).numpy()
    if colors.shape[:-1] != arrays["depth"].shape:
        raise ValueError(f"Image/depth shape mismatch: {job['key']}")
    norm = Normalize(*job["depth_color_range"], clip=True)
    cmap = matplotlib.colormaps["turbo"]
    for i, view in enumerate(views):
        filename = f"{view['pair_index']:05d}_{Path(view['image']).stem}.png"
        path = destination / "depth" / view["modality"] / filename
        expected = np.rint(cmap(norm(arrays["depth"][i]))[..., :3] * 255).astype(np.uint8)
        with Image.open(path) as picture:
            if not np.array_equal(np.asarray(picture), expected):
                raise ValueError(f"Cached depth does not reproduce published PNG: {path}")

    paths = {m: destination / f"{m}_full.ply" for m in job["counts"]}
    temporary, streams, preexisting = {}, {}, {}
    try:
        for name, path in paths.items():
            preexisting[name] = path.is_file()
            if preexisting[name]:
                read_ply(path, job["counts"][name])
                continue
            stage = destination / f".{path.name}.full_export.tmp"
            if stage.exists():
                raise FileExistsError(f"Inspect incomplete staging file before resume: {stage}")
            temporary[name] = stage
            streams[name] = stage.open("xb")
            streams[name].write(header(job["counts"][name]))
        # All pixels in original row-major order; RGB views precede thermal views.
        for modality in ("rgb", "thermal"):
            if modality not in paths:
                continue
            for i, view in enumerate(views):
                if view["modality"] != modality:
                    continue
                xyz = depth_to_world(arrays["depth"][i], arrays["extrinsics"][i],
                                     arrays["intrinsics"][i]).reshape(-1, 3).astype(np.float32)
                if not np.isfinite(xyz).all():
                    raise ValueError(f"Nonfinite world coordinates: {job['key']}/{i}")
                rgb = np.rint(colors[i].reshape(-1, 3) * 255).astype(np.uint8)
                packed = np.empty(len(xyz), dtype=DTYPE)
                for index, field in enumerate(("x", "y", "z")):
                    packed[field] = xyz[:, index]
                for index, field in enumerate(("red", "green", "blue")):
                    packed[field] = rgb[:, index]
                for name in (modality, "combined"):
                    if name in streams:
                        packed.tofile(streams[name])
        for stream in streams.values():
            stream.close()

        # Read every exported vertex back and compare XYZ and RGB to every source pixel.
        mapped = {name: read_ply(temporary.get(name, path), job["counts"][name])
                  for name, path in paths.items()}
        offsets = Counter()
        for modality in ("rgb", "thermal"):
            if modality not in mapped:
                continue
            for i, view in enumerate(views):
                if view["modality"] != modality:
                    continue
                xyz = depth_to_world(arrays["depth"][i], arrays["extrinsics"][i],
                                     arrays["intrinsics"][i]).reshape(-1, 3).astype(np.float32)
                rgb = np.rint(colors[i].reshape(-1, 3) * 255).astype(np.uint8)
                for name in (modality, "combined"):
                    if name not in mapped:
                        continue
                    chunk = mapped[name][offsets[name]:offsets[name] + len(xyz)]
                    for index, field in enumerate(("x", "y", "z")):
                        if not np.array_equal(chunk[field], xyz[:, index]):
                            raise ValueError(f"PLY XYZ differs: {job['key']}/{name}/{i}/{field}")
                    for index, field in enumerate(("red", "green", "blue")):
                        if not np.array_equal(chunk[field], rgb[:, index]):
                            raise ValueError(f"PLY RGB differs: {job['key']}/{name}/{i}/{field}")
                    offsets[name] += len(xyz)
        if dict(offsets) != job["counts"]:
            raise ValueError(f"Not every pixel exported: {job['key']}")
        mapped.clear()
        # Hard linking commits each complete verified file atomically without replacing a file.
        for name, stage in temporary.items():
            os.link(stage, paths[name])
            stage.unlink()
        files = {path.name: {"vertices": job["counts"][name], "bytes": path.stat().st_size,
                             "sha256": digest(path), "created": not preexisting[name]}
                 for name, path in paths.items()}
        return {"key": job["key"], "predictions": str(source), "prediction_sha256": source_hash,
                "depth_method": job["depth_method"], "depth_shape": job["depth_shape"],
                "input_sha256": job["input_sha256"], "files": files,
                "checks": {"all_source_pixels_exported": True, "all_xyz_and_colors_exact": True,
                           "combined_exact_concatenation": "combined" in paths,
                           "all_depth_pngs_match_cached_depth": True,
                           "processed_inputs_match_archived_hash": True},
                "seconds": round(time.monotonic() - started, 3)}
    finally:
        for stream in streams.values():
            stream.close()
        for stage in temporary.values():
            if stage.exists():
                stage.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record-dir", type=Path, default=ROOT / "artifacts/full_pixel_pointclouds_20261019")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("workers must be positive")
    jobs = make_plan()
    missing_bytes = sum(len(header(count)) + count * DTYPE.itemsize
                        for j in jobs for name, count in j["counts"].items()
                        if not (Path(j["destination"]) / f"{name}_full.ply").exists())
    summary = {"scene_runs": len(jobs), "groups": dict(Counter(j["group"] for j in jobs)),
               "full_ply_files": sum(len(j["counts"]) for j in jobs),
               "new_files": sum(not (Path(j["destination"]) / f"{name}_full.ply").exists()
                                for j in jobs for name in j["counts"]),
               "additional_bytes": missing_bytes, "free_bytes": shutil.disk_usage(OUTPUT).free}
    print(json.dumps(summary), flush=True)
    if args.plan_only:
        return
    if summary["free_bytes"] < missing_bytes + 1024**3:
        raise OSError("Insufficient free disk space for full exports")
    record = args.record_dir.resolve()
    record.mkdir(parents=True, exist_ok=True)
    progress_path = record / "progress.json"
    if progress_path.exists():
        if not args.resume:
            raise FileExistsError("Run record exists; use --resume")
        progress = json.loads(progress_path.read_text())
        for job in jobs:
            result = progress["completed"].get(job["key"])
            if result:
                if digest(job["predictions"]) != result["prediction_sha256"]:
                    raise ValueError(f"Source changed after export: {job['key']}")
                for name, info in result["files"].items():
                    if digest(Path(job["destination"]) / name) != info["sha256"]:
                        raise ValueError(f"Output changed after export: {job['key']}/{name}")
    else:
        baseline = {str(p.relative_to(OUTPUT)): {"bytes": p.stat().st_size, "sha256": digest(p)}
                    for p in sorted(OUTPUT.rglob("*")) if p.is_file()}
        atomic_json(record / "original_files.json", baseline)
        atomic_json(record / "plan.json", {"created_at": now(), **summary, "jobs": jobs,
                    "script_sha256": digest(Path(__file__)),
                    "filtering": False, "sampling": False, "deduplication": False,
                    "point_order": "RGB views then thermal views; row-major pixels in each view",
                    "resolution": "every pixel of the saved processed prediction (process_res=518)",
                    "model_inference_repeated": False})
        shutil.copy2(__file__, record / Path(__file__).name)
        progress = {"status": "running", "started_at": now(), "completed": {}, "failed": {}}
    atomic_json(progress_path, progress)
    pending = [j for j in jobs if j["key"] not in progress["completed"]]
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn"),
                             initializer=initialize_worker) as pool:
        futures = {pool.submit(export_scene, job): job for job in pending}
        for future in as_completed(futures):
            job = futures[future]
            try:
                result = future.result()
                progress["completed"][job["key"]] = result
                progress["failed"].pop(job["key"], None)
                event = {"time": now(), "completed": len(progress["completed"]),
                         "total": len(jobs), "key": job["key"], "seconds": result["seconds"]}
            except Exception as error:
                progress["failed"][job["key"]] = repr(error)
                event = {"time": now(), "failed": job["key"], "error": repr(error)}
            atomic_json(progress_path, progress)
            with (record / "run.log").open("a") as stream:
                stream.write(json.dumps(event, ensure_ascii=False) + "\n")
            print(json.dumps(event, ensure_ascii=False), flush=True)
    baseline = json.loads((record / "original_files.json").read_text())
    changed = [name for name, info in baseline.items()
               if not (OUTPUT / name).is_file() or digest(OUTPUT / name) != info["sha256"]]
    expected = {str((Path(j["destination"]) / f"{name}_full.ply").relative_to(OUTPUT))
                for j in jobs for name in j["counts"]}
    actual = {str(p.relative_to(OUTPUT)) for p in OUTPUT.rglob("*_full.ply")}
    passed = not changed and not progress["failed"] and len(progress["completed"]) == len(jobs) and actual == expected
    files = [info for result in progress["completed"].values() for info in result["files"].values()]
    validation = {"status": "passed" if passed else "failed", "finished_at": now(),
                  "scene_runs": len(progress["completed"]), "full_ply_files": len(files),
                  "new_files": sum(info["created"] for info in files),
                  "existing_full_files_verified": sum(not info["created"] for info in files),
                  "total_vertices_including_combined": sum(info["vertices"] for info in files),
                  "total_bytes": sum(info["bytes"] for info in files),
                  "original_files_checked": len(baseline), "original_files_changed": changed,
                  "all_expected_full_files_present": actual == expected,
                  "filtering": False, "sampling": False, "deduplication": False,
                  "all_exported_coordinates_and_colors_read_back_exactly": not progress["failed"],
                  "all_source_depth_pngs_and_processed_input_hashes_verified": not progress["failed"],
                  "failed": progress["failed"]}
    atomic_json(record / "validation.json", validation)
    progress["status"] = "complete" if passed else "failed"
    progress["finished_at"] = now()
    atomic_json(progress_path, progress)
    print(json.dumps(validation, ensure_ascii=False), flush=True)
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
