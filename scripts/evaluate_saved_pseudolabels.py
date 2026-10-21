#!/usr/bin/env python3
"""Evaluate saved predictions against dataset VGGT pseudo labels (not measured GT)."""

import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch

from depth_anything_3.thermal.data import read_frame
from depth_anything_3.thermal.loss import geometry_loss


def evaluate(folder):
    meta = json.loads((folder / "selection.json").read_text())
    scene = Path(meta["scene"])
    frames = json.loads((scene / "transforms.json").read_text())["frames"]
    with np.load(folder / "predictions.npz") as data:
        arrays = {k: data[k] for k in data.files}
    h, w = arrays["depths"].shape[-2:]
    depths, extrinsics, intrinsics = [], [], []
    for i, view in enumerate(meta["selected_views"]):
        f = frames[view["frame_index"]][view["modality"]]
        assert (scene / f["file_path"]).resolve() == Path(view["source"]).resolve()
        image, depth, ext, intr = read_frame(scene, f, h, w)
        np.testing.assert_array_equal(np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255,
                                      arrays["images"][i])
        depths.append(depth)
        extrinsics.append(ext)
        intrinsics.append(intr)
    target = torch.from_numpy(np.stack(depths))[None]
    valid = torch.isfinite(target) & (target > 1e-6)
    target = torch.where(valid, target, 0)
    gt_scale = target[valid].median()
    target = target / gt_scale
    gt_ext = torch.from_numpy(np.stack(extrinsics))[None]
    gt_ext = gt_ext @ torch.linalg.inv(gt_ext[:, :1])
    gt_ext[..., :3, 3] /= gt_scale
    gt_intr = torch.from_numpy(np.stack(intrinsics))[None]
    prediction = {key: torch.from_numpy(arrays[source])[None] for key, source in (
        ("depth", "depths"), ("extrinsics", "extrinsics_world2cam"), ("intrinsics", "intrinsics"))}
    batch = {"depth": target, "valid": valid, "extrinsics": gt_ext, "intrinsics": gt_intr}
    with torch.inference_mode():
        _, metrics = geometry_loss(prediction, batch)
    shared_scale = float(metrics["scale"])
    predicted_depth = prediction["depth"] * shared_scale
    pred_ext = torch.eye(4).expand(1, len(depths), 4, 4).clone()
    pred_ext[..., :3, :] = prediction["extrinsics"][..., :3, :]
    pred_ext = pred_ext @ torch.linalg.inv(pred_ext[:, :1])
    pred_ext[..., :3, 3] *= shared_scale
    pred_centers = torch.linalg.inv(pred_ext)[0, :, :3, 3]
    gt_centers = torch.linalg.inv(gt_ext)[0, :, :3, 3]
    thermal = torch.from_numpy(arrays["thermal_mask"])
    rgb_centers = gt_centers[~thermal]
    radius = ((rgb_centers - rgb_centers.mean(0)).square().sum(-1).mean()).sqrt()
    assert radius > 1e-8
    center_errors = (pred_centers - gt_centers).norm(dim=-1) / radius
    relative_rotation = pred_ext[0, :, :3, :3] @ gt_ext[0, :, :3, :3].mT
    cosine = ((relative_rotation.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(-1, 1)
    angles = cosine.acos().rad2deg()
    rows = []
    for mode, ids in (("all", torch.arange(len(depths))),
                      ("rgb", (~thermal).nonzero().flatten()),
                      ("thermal", thermal.nonzero().flatten())):
        mask = valid[0, ids]
        p, g = predicted_depth[0, ids][mask], target[0, ids][mask]
        ratio = torch.maximum(p / g, g / p)
        pose_ids = ids[ids != 0]
        rows.append({
            "method": meta["method"], "pairs": len(meta["selected_pair_indices"]), "mode": mode,
            "abs_rel": float(((p - g).abs() / g).mean()),
            "delta1": float((ratio < 1.25).float().mean()),
            "rotation_mean_deg_excluding_reference": float(angles[pose_ids].mean()),
            "camera_center_rmse_over_pseudo_rgb_radius": float(center_errors[ids].square().mean().sqrt()),
            "valid_depth_pixels": int(mask.sum()),
            "shared_depth_scale_to_source": shared_scale * float(gt_scale)})
    per_view = []
    for i, view in enumerate(meta["selected_views"]):
        mask = valid[0, i]
        relative = (predicted_depth[0, i][mask] - target[0, i][mask]).abs() / target[0, i][mask]
        per_view.append({**view, "abs_rel": float(relative.mean()), "rotation_deg": float(angles[i]),
                         "center_error_over_pseudo_rgb_radius": float(center_errors[i])})
    return {"folder": str(folder.resolve()), "scene": str(scene),
            "training_loss_metrics": {k: float(v) for k, v in metrics.items()},
            "rows": rows, "per_view": per_view}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folders", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--training-manifest", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(8)
    manifest = json.loads(args.training_manifest.read_text())
    result = {"methodology": {
        "labels": "VGGT pseudo labels, not independently measured ground truth",
        "depth_alignment": "One median log-depth scale over all valid pixels of both modalities per joint inference; no per-frame/per-modality fitting",
        "pose_alignment": "Both w2c trajectories referenced to first RGB camera; apply the same depth scale to translations; no separate trajectory fitting",
        "depth_pixels": "All source-valid pixels, including border; no prediction confidence filter",
        "rotation": "Relative-to-first-camera geodesic error, mean excluding the reference view",
        "camera_center_error": "RMS center error divided by source pseudo-label RGB trajectory RMS radius",
        "source_transforms_convention": "OpenCV world-to-camera, same interpretation as training",
    }, "cases": []}
    for folder in args.folders:
        case = evaluate(folder)
        case["training_scene_split"] = [key for key in ("train", "eval", "held_out")
            if any(Path(record["transforms"]).parent.resolve() == Path(case["scene"]).resolve()
                   for record in manifest[key])]
        result["cases"].append(case)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "pseudo_label_metrics.json").write_text(json.dumps(result, indent=2))
    rows = [row for case in result["cases"] for row in case["rows"]]
    with (args.output_dir / "pseudo_label_metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
