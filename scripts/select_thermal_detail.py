#!/usr/bin/env python3
"""Select with normalized boundary and depth scores plus modality regression guards."""
import argparse
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from infer_sear_comparison import file_hash, atomic_json


def select(rows, boundary_weight=1.0, max_regression=.05):
    baseline = next(r for r in rows if r["event"] == "baseline")

    def scores(row):
        metrics = row["metrics"]
        depth = sum(metrics[m]["abs_rel"] for m in ("rgb", "mixed", "thermal")) / 3
        boundary = sum(metrics[m]["rgb_boundary"] for m in ("rgb", "mixed")) / 2
        return depth, boundary

    base_depth, base_boundary = scores(baseline)
    if base_boundary <= 0:
        raise ValueError("No evaluable baseline RGB boundaries")
    candidates = []
    for row in rows:
        if row["event"] not in ("baseline", "validation"):
            continue
        depth, boundary = scores(row)
        regression = {mode: row["metrics"][mode]["abs_rel"] / baseline["metrics"][mode]["abs_rel"] - 1
                      for mode in ("rgb", "mixed", "thermal")}
        allowed = all(value <= max_regression for value in regression.values())
        score = depth / base_depth + boundary_weight * boundary / base_boundary
        candidates.append({"step": row["step"], "depth": depth, "boundary": boundary,
                           "normalized_score": score, "modality_regression": regression, "allowed": allowed})
    accepted = [row for row in candidates if row["allowed"]]
    selected = min(accepted, key=lambda r: (r["normalized_score"], r["step"]))
    return {"rule": "Equal normalized mean RGB/mixed/thermal pseudo AbsRel and mean RGB/mixed teacher-boundary loss; at most 5% regression per mode. No individual Ebike ROI enters selection.",
            "boundary_weight": boundary_weight, "max_regression": max_regression,
            "selected": selected, "candidates": candidates}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record-dir", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    args = parser.parse_args()
    record = Path(args.record_dir)
    checkpoint_dir = Path(args.checkpoint_dir)
    rows = [json.loads(l) for l in (record / "metrics.jsonl").read_text().splitlines()]
    result = select(rows)
    step = result["selected"]["step"]
    config = json.loads((record / "run_config.json").read_text())
    source = Path(config["initialize_adapter"]) if step == 0 else checkpoint_dir / f"adapter_step_{step:04d}.pt"
    destination = checkpoint_dir / "adapter_selected.pt"
    if destination.exists():
        raise FileExistsError(destination)
    shutil.copy2(source, destination)
    result.update(source=str(source), destination=str(destination), sha256=file_hash(destination))
    atomic_json(record / "selection.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
