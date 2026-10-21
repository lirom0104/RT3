#!/usr/bin/env python3
"""Apply the saved joint detail procedure to matched RGB-only and mixed sweeps."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import json
import multiprocessing
from pathlib import Path
import shutil
import sys
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import cv2
import numpy as np
from depth_anything_3.thermal.multiview_detail import transfer_boundaries, reconcile_depths
from infer_sear_comparison import atomic_json, file_hash
from refine_multiview_detail import base_masks, export_masks


def refine_case(task):
    cv2.setNumThreads(1)
    base = dict(np.load(task["base"]))
    teacher = dict(np.load(task["teacher"]))
    assert base["depth"].shape == teacher["depth"].shape
    count = len(base["depth"])
    thermal = np.zeros(count, dtype=bool)
    if task["protocol"] == "rgb_thermal":
        thermal[count // 2:] = True
        paired = (np.arange(count, dtype=np.int32) + count // 2) % count
    else:
        # Paired indices are unused when no thermal views exist.
        paired = np.arange(count, dtype=np.int32)
    edge_depth = transfer_boundaries(base["depth"], teacher["depth"], thermal)
    depth, support = reconcile_depths(edge_depth, base["extrinsics"], base["intrinsics"], thermal, paired)
    valid, thin, recovered = export_masks(base, teacher, depth, support, thermal, paired)
    original_valid = base_masks(base)
    output = Path(task["output"])
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **{**base, "depth": depth}, edge_depth=edge_depth,
                        support=support, export_valid=valid, thin_regions=thin,
                        recovered_thin_valid=recovered)
    with np.load(output) as saved:
        for key in ("extrinsics", "intrinsics", "depth_conf"):
            assert np.array_equal(saved[key], base[key]), key
        assert np.isfinite(saved["depth"]).all() and (saved["depth"] > 0).all()
    return {"case": task["case"], "input_sha256": task["input_sha256"],
            "source_prediction": task["base"], "source_sha256": file_hash(task["base"]),
            "teacher_prediction": task["teacher"], "teacher_sha256": file_hash(task["teacher"]),
            "prediction": str(output), "prediction_sha256": file_hash(output),
            "cameras_intrinsics_confidence_unchanged": True,
            "finite_positive_depths": True,
            "rgb_retained_valid_fraction": float((valid[~thermal] & original_valid[~thermal]).sum()
                                                  / original_valid[~thermal].sum()),
            "recovered_thin_pixels": int(recovered.sum())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--new-run", type=Path, required=True)
    parser.add_argument("--official-run", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    new_run, official_run = args.new_run.resolve(), args.official_run.resolve()
    output = new_run / "postprocessed"
    if output.exists():
        raise FileExistsError(output)
    new_config = json.loads((new_run / "run_config.json").read_text())
    new_summary = json.loads((new_run / "summary.json").read_text())
    official_summary = json.loads((official_run / "summary.json").read_text())
    assert new_summary["status"] == official_summary["status"] == "complete"
    assert new_config["adapter_loaded"]
    reference = json.loads(Path(new_config["reference_config"]).read_text())
    assert new_config["adapter_sha256"] == reference["adapter_sha256"]
    source_files = ["src/depth_anything_3/thermal/multiview_detail.py",
                    "src/depth_anything_3/thermal/detail_refinement.py",
                    "scripts/refine_multiview_detail.py"]
    for name in source_files:
        assert file_hash(ROOT / name) == reference["code_sha256"][name], name
    teachers = {(m["exposure"], m["protocol"]): m for m in official_summary["metrics"]}
    tasks = []
    for entry in new_summary["metrics"]:
        label, protocol = entry["exposure"], entry["protocol"]
        teacher = teachers[label, protocol]
        assert entry["input_sha256"] == teacher["input_sha256"]
        base_path = new_run / "predictions" / label / protocol / "predictions.npz"
        teacher_path = official_run / "predictions" / label / protocol / "predictions.npz"
        assert file_hash(base_path) == entry["prediction_sha256"]
        assert file_hash(teacher_path) == teacher["prediction_sha256"]
        tasks.append({"case": f"{label}/{protocol}", "protocol": protocol,
                      "base": str(base_path), "teacher": str(teacher_path),
                      "output": str(output / "predictions" / label / protocol / "predictions.npz"),
                      "input_sha256": entry["input_sha256"]})
    output.mkdir()
    config = {**new_config, "postprocessing": "Reference boundary transfer and multiview depth consensus",
              "raw_run": str(new_run), "teacher_run": str(official_run),
              "code_sha256": {name: file_hash(ROOT / name) for name in source_files},
              "postprocess_script_sha256": file_hash(Path(__file__)),
              "parameters": {"boundary_radius": 8, "boundary_iterations": 2000,
                             "boundary_sigma": .025, "band_anchor": 1e-5, "initial_sigma": 32.,
                             "thin_surface_anchor": 1., "core_gradient": .01,
                             "consensus_iterations": 5, "log_depth_tolerance": .025,
                             "rgb_neighbors": 4, "paired_thermal_weight": .25}}
    atomic_json(output / "run_config.json", config)
    shutil.copy2(Path(__file__), output / Path(__file__).name)

    def event(name, **values):
        row = {"time": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(), "event": name, **values}
        with (output / "run.log").open("a") as stream:
            stream.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)

    results = []
    event("start", cases=len(tasks), workers=args.workers)
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = [pool.submit(refine_case, task) for task in tasks]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            atomic_json(output / "progress.json", {"status": "running", "completed": results})
            event("case_complete", case=result["case"], retained=result["rgb_retained_valid_fraction"])
    for path, expected in new_config["input_files_sha256"].items():
        assert file_hash(path) == expected
    atomic_json(output / "summary.json", {"status": "complete", "cases": sorted(results, key=lambda x: x["case"]),
                "checks": {"same_inputs_as_official_teachers": True, "same_reference_postprocessing_code": True,
                           "cameras_intrinsics_confidence_unchanged": True, "finite_positive_depths": True,
                           "input_files_unchanged": True}})
    atomic_json(output / "progress.json", {"status": "complete", "completed": results})
    event("finished", cases=len(results), output=str(output))


if __name__ == "__main__":
    main()
