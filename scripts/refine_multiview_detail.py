#!/usr/bin/env python3
"""Reproducible cached-prediction repair; never overwrite an existing result."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
from depth_anything_3.thermal.multiview_detail import (
    transfer_boundaries, reconcile_depths, thin_surface_regions,
)
from depth_anything_3.utils.io.input_processor import InputProcessor
from infer_sear_comparison import depth_to_world, write_ply, export_depths, fingerprints


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def base_masks(arrays, confidence_ratio=1.):
    masks = []
    for depth, confidence in zip(arrays["depth"], arrays["depth_conf"]):
        valid = np.isfinite(depth) & (depth > 0)
        valid &= confidence >= confidence_ratio * np.percentile(confidence, 50)
        valid &= depth <= np.percentile(depth, 99)
        border = max(1, int(min(depth.shape) * .02))
        valid[:border] = valid[-border:] = False
        valid[:, :border] = valid[:, -border:] = False
        masks.append(valid)
    return np.array(masks)


def export_masks(base, teacher, depth, support, thermal, pairs, *,
                 core_gradient=.01, recover_thin=True):
    """Retain corroborated thin surfaces below the global confidence median.

    Exceptions need two other RGB views, half the original confidence threshold,
    and a small near-surface region. Thermal exceptions additionally require a
    matching reprojection from the paired RGB detail. Depth, borders and cameras
    keep their original validity rules.
    """
    original = base_masks(base)
    valid = original & (support >= 1)
    thin = np.zeros_like(valid)
    recovered = np.zeros_like(valid)
    if recover_thin:
        thin = thin_surface_regions(base["depth"], teacher["depth"], thermal,
                                    core_gradient=core_gradient)
        for j in np.flatnonzero(thermal):
            i = int(pairs[j])
            y, x = np.where(thin[i] & (support[i] >= 2))
            if not len(x):
                continue
            xyz = depth_to_world(depth[i], base["extrinsics"][i], base["intrinsics"][i])[y, x]
            camera = xyz @ base["extrinsics"][j, :3, :3].T + base["extrinsics"][j, :3, 3]
            projected = camera @ base["intrinsics"][j].T
            uv = np.rint(projected[:, :2] / projected[:, 2:3]).astype(np.int32)
            h, w = depth[j].shape
            inside = ((camera[:, 2] > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < w)
                      & (uv[:, 1] >= 0) & (uv[:, 1] < h))
            u, v = uv[inside].T
            matches = np.abs(np.log(depth[j, v, u] / camera[inside, 2])) < .025
            thin[j, v[matches], u[matches]] = True
        recovered = (thin & (support >= 2) & base_masks(base, confidence_ratio=.5) & ~valid)
        valid |= recovered
    return valid, thin, recovered


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--thin-anchor-weight", type=float, default=1.)
    parser.add_argument("--thin-anchor-gradient", type=float, default=.01)
    parser.add_argument("--no-thin-export-recovery", action="store_true")
    args = parser.parse_args()
    if args.output.exists() or args.record.exists():
        raise FileExistsError("Output and record directories must both be new")
    config = json.loads(args.config.read_text())
    scene = next(s for s in config["scenes"] if s["scene"] == args.scene)
    base, teacher = dict(np.load(args.base)), dict(np.load(args.teacher))
    views = scene["views"]
    thermal = np.array([view["modality"] == "thermal" for view in views])
    pairs = np.array([next(j for j, other in enumerate(views)
                          if other["pair_index"] == view["pair_index"]
                          and other["modality"] != view["modality"])
                      for view in views], dtype=np.int32)
    if len(views) != len(base["depth"]):
        raise ValueError("Scene views do not match recorded predictions")
    normalized, _, _ = InputProcessor()([view["image"] for view in views],
                                      process_res=config["process_res"], sequential=True)
    if normalized.ndim == 5:
        normalized = normalized[0]
    input_hash = hashlib.sha256(normalized.numpy().tobytes()).hexdigest()
    progress_path = args.config.parent / "progress.json"
    if progress_path.is_file():
        original = json.loads(progress_path.read_text())["completed"][f"new/{args.scene}"]
        if original["input_sha256"] != input_hash:
            raise ValueError("Input images/preprocessing differ from the base inference")
    mean = normalized.new_tensor([.485, .456, .406])[None, :, None, None]
    std = normalized.new_tensor([.229, .224, .225])[None, :, None, None]
    images = (normalized * std + mean).clamp(0, 1).permute(0, 2, 3, 1).numpy()
    if images.shape[:-1] != base["depth"].shape:
        raise ValueError("Image dimensions do not match cached depths")
    print("Transferring boundaries with base geometry anchors", flush=True)
    edge_depth = transfer_boundaries(base["depth"], teacher["depth"], thermal,
                                     thin_anchor=args.thin_anchor_weight,
                                     core_gradient=args.thin_anchor_gradient)
    print("Reconciling RGB and thermal in fixed base cameras", flush=True)
    depth, support = reconcile_depths(edge_depth, base["extrinsics"], base["intrinsics"], thermal, pairs)
    result = {**base, "depth": depth}
    original_valid = base_masks(base)
    valid, thin_regions, recovered = export_masks(
        base, teacher, depth, support, thermal, pairs,
        core_gradient=args.thin_anchor_gradient, recover_thin=not args.no_thin_export_recovery)
    args.record.mkdir(parents=True)
    args.output.mkdir(parents=True)
    np.savez_compressed(args.record / "predictions.npz", **result, support=support,
                        export_valid=valid, edge_depth=edge_depth,
                        thin_regions=thin_regions, recovered_thin_valid=recovered)
    np.save(args.record / "processed_images.npy", images)
    clouds, retained, selection = {}, {}, {}
    max_points, seed = config["max_points_per_view"], config["seed"]
    for modality in ("rgb", "thermal"):
        points, colors, pixel_indices, view_indices = [], [], [], []
        indices = np.flatnonzero(thermal if modality == "thermal" else ~thermal)
        for i in indices:
            selected = np.flatnonzero(valid[i])
            selected = np.random.default_rng(seed + int(i)).choice(
                selected, min(len(selected), max_points), replace=False)
            world = depth_to_world(depth[i], result["extrinsics"][i], result["intrinsics"][i])
            points.append(world.reshape(-1, 3)[selected])
            colors.append(images[i].reshape(-1, 3)[selected])
            pixel_indices.append(selected)
            view_indices.append(np.full(len(selected), i, dtype=np.int32))
        clouds[modality] = (np.concatenate(points), np.concatenate(colors))
        write_ply(args.output / f"{modality}.ply", *clouds[modality])
        selection[f"{modality}_pixels"] = np.concatenate(pixel_indices)
        selection[f"{modality}_views"] = np.concatenate(view_indices)
        retained[modality] = float((valid[indices] & original_valid[indices]).sum()
                                   / original_valid[indices].sum())
    write_ply(args.output / "combined.ply", np.concatenate([c[0] for c in clouds.values()]),
              np.concatenate([c[1] for c in clouds.values()]))
    np.savez_compressed(args.record / "export_pixels.npz", **selection)
    color_range = export_depths(args.output, depth, scene, "joint multiview detail")
    record = {
        "scene": scene, "base": str(args.base.resolve()), "base_sha256": digest(args.base),
        "teacher": str(args.teacher.resolve()), "teacher_sha256": digest(args.teacher),
        "config": str(args.config.resolve()), "input_sha256": input_hash,
        "implementation_sha256": digest(ROOT / "src/depth_anything_3/thermal/multiview_detail.py"),
        "export_script_sha256": digest(Path(__file__)),
        "parameters": {"boundary_radius": 8, "boundary_iterations": 2000,
                       "boundary_sigma": .025, "band_anchor": 1e-5, "initial_sigma": 32.,
                       "thin_surface_anchor": args.thin_anchor_weight,
                       "thin_surface_base_max_log_gradient": args.thin_anchor_gradient,
                       "thin_surface_ridge_window": 5,
                       "thin_surface_min_log_range": .12,
                       "thin_export_recovery": not args.no_thin_export_recovery,
                       "thin_export_region_expansion_pixels": 3,
                       "thin_export_max_guide_log_difference": .04,
                       "thin_export_confidence_ratio": .5,
                       "thin_export_minimum_other_rgb_support": 2,
                       "consensus_iterations": 5, "log_depth_tolerance": .025,
                       "rgb_neighbors": 4, "paired_thermal_weight": .25,
                       "minimum_rgb_support": 1, "seed": seed, "max_points_per_view": max_points},
        "cameras_and_confidence_exactly_base": all(np.array_equal(result[k], base[k])
                                                   for k in ("extrinsics", "intrinsics", "depth_conf")),
        "retained_valid_pixel_fraction": retained,
        "additional_supported_thin_pixels": {
            m: int(recovered[thermal if m == "thermal" else ~thermal].sum())
            for m in ("rgb", "thermal")},
        "retained_valid_pixels_per_view": [int(v.sum()) for v in valid],
        "cloud_counts": {k: len(c[0]) for k, c in clouds.items()},
        "depth_color_range": color_range, "files": fingerprints(args.output),
        "limits": "Ebike-tested postprocessing, not a depth-ground-truth benchmark. PLY uses the same "
                  "depths as PNG, with explicit confidence/border/multiview masks. Occluded or "
                  "single-view details may be excluded. RGB/thermal consistency does not prove true geometry.",
    }
    (args.record / "run.json").write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(args.output), "retained": retained, "counts": record["cloud_counts"]}), flush=True)


if __name__ == "__main__":
    main()
