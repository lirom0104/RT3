#!/usr/bin/env python3
"""Check detail candidates on fixed original inputs and held-out scenes."""
import argparse
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import infer_sear_comparison as run
from train_thermal_detail import evaluate, atomic_json
from depth_anything_3.thermal.detail_data import DetailSEARDataset
sys.path.insert(0, str(ROOT / "artifacts/ebike_depth_edge_analysis"))
from analyze_depth import LINES, transition


def align(depth, target, valid):
    return float(np.exp(np.median(np.log(target[valid]) - np.log(depth[valid]))))


def geometry_metrics(depth, target, valid):
    scale = align(depth, target, valid)
    result = {"global_scale": scale}
    for name, selected in (("rgb", slice(0, 12)), ("thermal", slice(12, None))):
        d, g, m = depth[selected] * scale, target[selected], valid[selected]
        result[name + "_abs_rel"] = float(np.mean(np.abs(d[m] - g[m]) / g[m]))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--control")
    parser.add_argument("--extra", action="append", default=[], help="name=checkpoint")
    parser.add_argument("--record-dir", default=str(ROOT / "artifacts/thermal_giant_detail_20261012"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--skip-held-out", action="store_true")
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument("--teacher-context", choices=("rgb", "mixed"), default="rgb",
                        help="Use one common teacher context for all evaluated checkpoints")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    run.logger.level = 1
    original = json.loads((ROOT / "artifacts/comparison_sear_deal_20261011/run_config.json").read_text())
    training = SimpleNamespace(**json.loads((Path(args.record_dir) / "run_config.json").read_text()))
    training.teacher_context = args.teacher_context
    manifest = json.loads((Path(args.record_dir) / "data_manifest.json").read_text())
    scenes = {s["scene"].split("/")[-1]: copy.deepcopy(s) for s in original["scenes"]
              if s["scene"] in ("ThermalGaussian/Ebike", "ThermalGaussian/Building")}
    bd = ROOT.parent / "analysis/sear_deal_repair_20261011/building_diagnosis"
    building = dict(np.load(bd / "raw_depths.npz"))
    ebike = dict(np.load(ROOT / "artifacts/ebike_sear_deal_depth_analysis_20261012/original_raw_depths.npz"))
    label_dir = ROOT.parent / "data/SEAR-deal/ThermalGaussian/Ebike/VGGT"
    frames = json.loads((label_dir / "transforms.json").read_text())["frames"]
    ebike_gt = np.stack([np.load(label_dir / frames[i]["rgb"]["depth_file_path"]).squeeze()
                         for i in scenes["Ebike"]["selected_pair_indices"]])
    ebike_valid = np.isfinite(ebike_gt) & (ebike_gt > 1e-6)
    repaired = copy.deepcopy(scenes["Building"])
    label_dir = ROOT.parent / "data/SEAR-deal/ThermalGaussian/Building/VGGT"
    frames = json.loads((label_dir / "transforms.json").read_text())["frames"]
    lookup = {f["source_frame_index"]: f for f in frames}
    for view in repaired["views"]:
        view["image"] = str(label_dir / lookup[view["pair_index"]][view["modality"]]["file_path"])
    official = {"Ebike": ebike["official"], "Building": building["original_official"]}
    expected = {"Ebike": ebike["sear_deal_new"], "Building": building["original_new"]}
    variants = {"before": original["adapter"], "detail": args.checkpoint}
    if args.control:
        variants["control"] = args.control
    for item in args.extra:
        name, path = item.split("=", 1)
        variants[name] = path
    report = {"checkpoints": {}, "scenes": {}, "held_out": {}, "validation": {},
              "common_teacher_context": args.teacher_context,
              "limits": "Ebike labels are VGGT validation labels. Three edge profiles are local sharpness/position checks, not independent boundary GT. Building uses valid COLMAP training pixels on matched undistorted inputs, one scale shared by both modalities. Held-out supervision is also pseudo geometry."}
    depths, images = {}, {}
    model = run.load_giant(original["base_checkpoint"]).cuda().eval()
    for name, path in variants.items():
        digest = run.file_hash(path)
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        run.load_adapter(model, checkpoint)
        model.eval()
        report["checkpoints"][name] = {"path": str(path), "sha256": digest,
                                        "step": checkpoint["step"], "epoch": checkpoint["epoch"]}
        del checkpoint
        for scene_name, scene in scenes.items():
            arrays, im, stats = run.infer_scene(model, scene, "new", SimpleNamespace(**original))
            depth = arrays["depth"]
            if name == "before":
                assert np.array_equal(depth, expected[scene_name])
            row = stats
            if scene_name == "Ebike":
                scale = align(depth[:12], ebike_gt, ebike_valid)
                row["rgb_abs_rel_to_VGGT"] = float(np.mean(np.abs(depth[:12][ebike_valid] * scale - ebike_gt[ebike_valid]) / ebike_gt[ebike_valid]))
                row["edges"] = {}
                for line in LINES:
                    try:
                        edge = transition(depth[9], line)[0]
                        edge["center_shift_from_official_pixels"] = edge["center_50_pixels"] - transition(official["Ebike"][9], line)[0]["center_50_pixels"]
                        row["edges"][line[0]] = edge
                    except (ValueError, IndexError) as error:
                        row["edges"][line[0]] = {"unavailable": str(error)}
            else:
                a, _, repaired_stats = run.infer_scene(model, repaired, "new", SimpleNamespace(**original))
                row["colmap_consistency"] = geometry_metrics(a["depth"], building["gt"], building["valid"])
                row["repaired_input_sha256"] = repaired_stats["input_sha256"]
            depths[scene_name + "_" + name] = depth
            images[scene_name] = im
            np.savez_compressed(output / (scene_name + "_" + name + ".npz"), **arrays)
            report["scenes"].setdefault(scene_name, {})[name] = row
            atomic_json(output / "metrics.json", report)
            print(json.dumps({"variant": name, "scene": scene_name, "metrics": row}), flush=True)
        for split, skip in (("eval", args.skip_validation), ("held_out", args.skip_held_out)):
            if skip:
                continue
            loader = DataLoader(DetailSEARDataset(manifest[split], training.views,
                                    training.height, training.width, False, training.seed),
                                batch_size=1, num_workers=2)
            metrics, rows = evaluate(model, loader, torch.device("cuda"), training)
            report["validation" if split == "eval" else "held_out"][name] = {"metrics": metrics, "scenes": rows}
            atomic_json(output / "metrics.json", report)
            print(json.dumps({"variant": name, "split": split, "metrics": metrics}), flush=True)
        assert run.file_hash(path) == digest
    for scene_name in scenes:
        assert len({r["input_sha256"] for r in report["scenes"][scene_name].values()}) == 1
    panels = ["official", *variants]
    for scene_name, indices in (("Ebike", (0, 9)), ("Building", (7, 19))):
        fig, axes = run.plt.subplots(2, len(panels)+1, figsize=(4*(len(panels)+1), 8))
        reference = official[scene_name][:12]
        lo, hi = np.percentile(reference, [2, 98])
        for row, index in enumerate(indices):
            axes[row, 0].imshow(images[scene_name][index], interpolation="nearest")
            axes[row, 0].set_title("RGB input" if index < 12 else "Thermal input")
            for col, name in enumerate(panels, start=1):
                d = official[scene_name] if name == "official" else depths[scene_name + "_" + name]
                scale = align(d[:12], reference, np.ones_like(reference, dtype=bool))
                axes[row, col].imshow(d[index] * scale, cmap="turbo", vmin=lo, vmax=hi, interpolation="nearest")
                axes[row, col].set_title(name)
            for axis in axes[row]:
                axis.set_axis_off()
        fig.suptitle(scene_name + " | Same inputs and colors | One RGB sequence scale per model | No smoothing")
        fig.tight_layout()
        fig.savefig(output / (scene_name + "_comparison.png"), dpi=150)
        run.plt.close(fig)


if __name__ == "__main__":
    main()
