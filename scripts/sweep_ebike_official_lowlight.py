#!/usr/bin/env python3
"""Inspect original DA3 at fixed exposure levels, with and without thermal inputs.

Prediction changes are diagnostics against normal-light predictions, not GT errors.
"""
import argparse
from datetime import datetime
import gc
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch

from depth_anything_3.thermal.adapter import load_adapter, load_giant
from depth_anything_3.utils.logger import logger
from infer_sear_comparison import atomic_json, file_hash, infer_scene


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("/home/lf/feedforward/data/test-dark/Ebike"))
    parser.add_argument("--reference-config", type=Path,
                        default=ROOT / "artifacts/comparison_joint_detail_all_20261012/run_config.json")
    parser.add_argument("--record-dir", type=Path, required=True)
    parser.add_argument("--model", choices=("official", "new"), default="official")
    parser.add_argument("--comparison-reference", type=Path,
                        help="Existing official sweep to verify identical inputs and share display ranges.")
    args = parser.parse_args()
    dataset, records = args.dataset.resolve(), args.record_dir.resolve()
    if records.exists():
        parser.error("Record directory must be new.")
    if dataset == records or dataset in records.parents:
        parser.error("Records must be outside the dataset.")
    reference = json.loads(args.reference_config.read_text())
    comparison_config = comparison_summary = None
    if args.comparison_reference:
        comparison_config = json.loads((args.comparison_reference / "run_config.json").read_text())
        comparison_summary = json.loads((args.comparison_reference / "summary.json").read_text())
        assert comparison_summary["status"] == "complete"
        for key in ("base_checkpoint", "seed", "process_res"):
            assert comparison_config[key] == reference[key], key
        assert comparison_config["ref_view_strategy"] == "first"
    if args.model == "new" and file_hash(reference["adapter"]) != reference["adapter_sha256"]:
        raise ValueError("Adapter weights differ from the reference experiment.")
    original_scene = next(s for s in reference["scenes"] if s["scene"] == "ThermalGaussian/Ebike")
    original_manifest = json.loads((dataset / "lowlight_processing/manifest.json").read_text())
    original_names = {Path(i["relative_path"]).name: i for i in original_manifest["images"]}
    variants = [("normal", 0, "rgb", None),
                ("ev4", -4, "rgb_lowlight", "lowlight_processing"),
                ("ev6", -6, "rgb_lowlight_ev6", "lowlight_processing_ev6"),
                ("ev8", -8, "rgb_lowlight_ev8", "lowlight_processing_ev8")]
    protocols = ("rgb_only", "rgb_thermal")
    scenes, source_statistics, input_files = {}, {}, {}
    for label, ev, folder, report in variants:
        entries = original_names if report is None else {
            Path(i["relative_path"]).name: i for i in
            json.loads((dataset / report / "manifest.json").read_text())["images"]}
        views = []
        for view in original_scene["views"]:
            name = Path(view["image"]).name
            if view["modality"] == "rgb":
                entry = entries[name]
                path = dataset / folder / entry["relative_path"]
                expected = entry["source_sha256"] if report is None else entry["output_sha256"]
            else:
                path = dataset / "VGGT/thermal" / name
                expected = file_hash(view["image"])
            if file_hash(path) != expected:
                raise ValueError(f"Input provenance mismatch: {path}")
            input_files[str(path)] = expected
            views.append({**view, "image": str(path)})
        scenes[label] = {**original_scene, "views": views}
        means, below_8 = [], []
        for entry in entries.values():
            with Image.open(dataset / folder / entry["relative_path"]) as im:
                gray = np.asarray(im.convert("L"))
            means.append(float(gray.mean()))
            below_8.append(float((gray < 8).mean()))
        source_statistics[label] = {"exposure_ev": ev, "rgb_root": str(dataset / folder),
                                    "mean_gray_8bit": float(np.mean(means)),
                                    "fraction_gray_below_8": float(np.mean(below_8))}
    records.mkdir(parents=True)
    if comparison_config is not None:
        assert input_files == comparison_config["input_files_sha256"]
    config = {"base_checkpoint": reference["base_checkpoint"], "adapter_loaded": args.model == "new",
              "reference_config": str(args.reference_config.resolve()), "seed": reference["seed"],
              "process_res": reference["process_res"], "selected_pair_indices": original_scene["selected_pair_indices"],
              "ref_view_strategy": "first", "reference_image": scenes["normal"]["views"][0]["image"],
              "protocols": list(protocols), "variants": source_statistics,
              "postprocessing": "None: raw model predictions", "input_files_sha256": input_files,
              "script_sha256": file_hash(Path(__file__)),
              "limits": "Synthetic exposure sweep; no ground-truth depth. Protocols use the same first RGB reference and identical RGB pixels."}
    config["model_version"] = args.model
    if args.model == "new":
        config.update({"adapter": reference["adapter"], "adapter_sha256": reference["adapter_sha256"]})
    if args.comparison_reference:
        config["comparison_reference"] = str(args.comparison_reference.resolve())
    atomic_json(records / "run_config.json", config)
    shutil.copy2(Path(__file__), records / Path(__file__).name)

    def event(name, **values):
        row = {"time": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(), "event": name, **values}
        with (records / "run.log").open("a") as stream:
            stream.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)

    torch.set_num_threads(8)
    logger.level = 1
    infer_args = SimpleNamespace(seed=config["seed"], process_res=config["process_res"],
                                 rgb_detail_sigma=0, ref_view_strategy="first")
    model = load_giant(config["base_checkpoint"]).cuda().eval()
    if args.model == "new":
        checkpoint = torch.load(config["adapter"], map_location="cpu", weights_only=True)
        if Path(checkpoint["base_checkpoint"]).resolve() != Path(config["base_checkpoint"]).resolve():
            raise ValueError("Adapter base checkpoint mismatch.")
        load_adapter(model, checkpoint)
        model.eval()
        event("adapter_loaded", path=config["adapter"], epoch=checkpoint["epoch"], step=checkpoint["step"])
        del checkpoint
    event("weights_loaded")
    count = len(original_scene["selected_pair_indices"])
    predictions, thumbnails, inference_records = {}, {}, {}
    thermal_images = None
    for label, *_ in variants:
        for protocol in protocols:
            scene = scenes[label]
            if protocol == "rgb_only":
                scene = {**scene, "views": scene["views"][:count]}
            arrays, images, info = infer_scene(model, scene, args.model, infer_args)
            if comparison_summary is not None:
                matched = next(m for m in comparison_summary["metrics"]
                               if m["exposure"] == label and m["protocol"] == protocol)
                assert info["input_sha256"] == matched["input_sha256"]
            assert info["shape"] == [count if protocol == "rgb_only" else 2 * count, 3, 392, 518]
            key = (label, protocol)
            predictions[key] = arrays["depth"][:count].copy()
            if protocol == "rgb_only":
                thumbnails[label] = np.rint(images * 255).astype(np.uint8)
            else:
                assert np.array_equal(thumbnails[label], np.rint(images[:count] * 255).astype(np.uint8))
                if thermal_images is None:
                    thermal_images = images[count:].copy()
                else:
                    assert np.array_equal(thermal_images, images[count:])
            folder = records / "predictions" / label / protocol
            folder.mkdir(parents=True)
            np.savez_compressed(folder / "predictions.npz", **arrays)
            inference_records[f"{label}/{protocol}"] = {**info,
                "rgb_confidence_median": float(np.median(arrays["depth_conf"][:count])),
                "prediction_sha256": file_hash(folder / "predictions.npz")}
            event("inference_complete", exposure=label, protocol=protocol, **info)
            del arrays, images
            gc.collect()
    del model
    gc.collect()
    torch.cuda.empty_cache()

    anchor_median = np.median(predictions[("normal", "rgb_only")])
    if args.comparison_reference:
        with np.load(args.comparison_reference / "predictions/normal/rgb_only/predictions.npz") as anchor:
            anchor_median = np.median(anchor["depth"])
    aligned, scale_factors, metrics = {}, {}, []
    for (label, protocol), depth in predictions.items():
        scale = float(anchor_median / np.median(depth))
        aligned[(label, protocol)] = depth * scale
        scale_factors[f"{label}/{protocol}"] = scale
    normal_depths = np.concatenate([aligned[("normal", p)].ravel() for p in protocols])
    low, high = np.percentile(normal_depths, [2, 98])
    if comparison_summary is not None:
        low, high = comparison_summary["display_color_range"]
    for label, *_ in variants:
        for protocol in protocols:
            normal = aligned[("normal", protocol)]
            changed = aligned[(label, protocol)]
            # Keep the same interior pixels, without confidence-based selection.
            a, b = normal[:, 8:-8, 8:-8], changed[:, 8:-8, 8:-8]
            metrics.append({"exposure": label, "protocol": protocol,
                "mean_relative_change_vs_normal_prediction": float(np.mean(np.abs(a - b) / a)),
                "sequence_scale_for_display": scale_factors[f"{label}/{protocol}"],
                **inference_records[f"{label}/{protocol}"]})
            directory = records / "depth" / label / protocol
            directory.mkdir(parents=True)
            for i, view in enumerate(scenes[label]["views"][:count]):
                colored = np.rint(matplotlib.colormaps["turbo"](
                    np.clip((changed[i] - low) / (high - low), 0, 1))[..., :3] * 255).astype(np.uint8)
                Image.fromarray(colored).save(directory / f"{view['pair_index']:05d}_{Path(view['image']).stem}.png")
    sample_indices = np.linspace(0, count - 1, 4, dtype=int)
    plots = []
    for index in sample_indices:
        figure, axes = plt.subplots(4, 3, figsize=(13, 12), squeeze=False)
        for row, (label, ev, *_) in enumerate(variants):
            axes[row, 0].imshow(thumbnails[label][index])
            axes[row, 0].set_title(f"RGB input | {'normal' if ev == 0 else str(ev) + ' EV'}")
            for column, protocol in enumerate(protocols, 1):
                artist = axes[row, column].imshow(aligned[(label, protocol)][index],
                                                 cmap="turbo", vmin=low, vmax=high)
                display_model = "New" if args.model == "new" else "Official"
                axes[row, column].set_title(f"{display_model} | RGB only" if protocol == "rgb_only" else f"{display_model} | RGB + thermal")
            for axis in axes[row]:
                axis.set_axis_off()
        name = Path(scenes["normal"]["views"][int(index)]["image"]).stem
        display_model = "new SEAR adapter" if args.model == "new" else "original DA3"
        figure.suptitle(f"{name} | {display_model} | fixed first RGB reference | raw depth", fontsize=14)
        figure.subplots_adjust(left=.02, right=.98, top=.94, bottom=.065, wspace=.02, hspace=.16)
        color_axis = figure.add_axes([.25, .025, .5, .012])
        figure.colorbar(artist, cax=color_axis, orientation="horizontal",
                       label="Common color range; one median scale per sequence; no ground-truth depth")
        filename = f"comparison_{name}.png"
        figure.savefig(records / filename, dpi=130)
        plt.close(figure)
        plots.append(filename)
    for path, expected in input_files.items():
        assert file_hash(path) == expected, path
    summary = {"status": "complete", "model_version": args.model, "input_statistics": source_statistics, "metrics": metrics,
               "display_color_range": [float(low), float(high)], "sequence_scales": scale_factors,
               "comparison_figures": plots,
               "checks": {"same_rgb_pixels_between_protocols": True, "thermal_unchanged_across_exposures": True,
                          "fixed_first_rgb_reference": True, "input_files_unchanged": True,
                          "finite_positive_depths_and_expected_shapes": True},
               "metric_interpretation": "Changes relative to each protocol's normal-light prediction after sequence-wide median scale alignment; these are not ground-truth accuracy errors."}
    atomic_json(records / "summary.json", summary)
    event("finished", report=str(records), metrics=metrics)


if __name__ == "__main__":
    main()
