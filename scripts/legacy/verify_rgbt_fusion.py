#!/usr/bin/env python3
"""Compare shared refinement against a source snapshot, including rendered output."""

import argparse
import importlib
import importlib.util
import json
import sys
import time
from contextlib import ExitStack
from dataclasses import fields
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import cv2
import numpy as np
import torch
from ablate_rgbt import cases

from depth_anything_3.rgbt.checkpoint import load_rgbt_model
from depth_anything_3.rgbt.engine import batch_to, psnr
from depth_anything_3.rgbt.renderer import render_rgbt


def snapshot_package(directory):
    directory = Path(directory).resolve()
    spec = importlib.util.spec_from_file_location(
        "_rgbt_before_fusion",
        directory / "__init__.py",
        submodule_search_locations=[str(directory)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return spec.name


def compared(left, right):
    errors = {}
    for name, expected in left.items():
        actual = right[name]
        assert torch.isfinite(expected).all() and torch.isfinite(actual).all(), name
        torch.testing.assert_close(actual, expected, rtol=3e-5, atol=3e-5, msg=name)
        errors[name] = float((actual.float() - expected.float()).abs().max())
    return errors


def output_tensors(output):
    result = {
        "gaussians." + f.name: getattr(output.gaussians, f.name)
        for f in fields(output.gaussians)
        if isinstance(getattr(output.gaussians, f.name), torch.Tensor)
    }
    result.update(
        {
            f.name: getattr(output, f.name)
            for f in fields(output)
            if isinstance(getattr(output, f.name), torch.Tensor)
        }
    )
    return result


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", default="outputs/rgbt/fusion_study/before/model.pt")
    parser.add_argument(
        "--before-source", default="outputs/rgbt/fusion_study/before/rgbt"
    )
    parser.add_argument("--after", default="checkpoints/RGBT/model.pt")
    parser.add_argument(
        "--data",
        nargs="+",
        default=[
            "outputs/rgbt/data_336",
            "outputs/rgbt/datasets/external_336",
            "outputs/rgbt/datasets/heldout_336",
            "outputs/rgbt/datasets/plant_336",
        ],
    )
    parser.add_argument("--camera", choices=["known", "predicted"], required=True)
    parser.add_argument("--targets", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="DailyStuff only, alternating warmed full forwards",
    )
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()
    torch.set_num_threads(4)
    cv2.setNumThreads(2)
    torch.cuda.set_device(args.device)
    model, checkpoint = load_rgbt_model(args.after, args.device)
    assert model.joint_refinement
    before = torch.load(args.before, mmap=True, map_location="cpu", weights_only=False)
    prefixes = {
        "neighborhood_refiner.": "refiner.local.",
        "crossview_refiner.": "refiner.crossview.",
        "surface_refiner.": "refiner.surface.",
    }
    for key, value in before["state_dict"].items():
        mapped = next(
            (
                new + key[len(old) :]
                for old, new in prefixes.items()
                if key.startswith(old)
            ),
            key,
        )
        assert torch.equal(value, checkpoint["state_dict"][mapped]), key
    assert len(before["state_dict"]) == len(checkpoint["state_dict"])
    package = snapshot_package(args.before_source)
    stages = []
    for module_name, class_name, config_key, prefix in [
        ("refinement", "NeighborhoodRefiner", "neighborhood", "neighborhood_refiner."),
        ("crossview", "CrossViewRefiner", "crossview", "crossview_refiner."),
        ("surface", "SurfaceRefiner", "surface", "surface_refiner."),
    ]:
        cls = getattr(importlib.import_module(package + "." + module_name), class_name)
        stage = cls(**before["model_config"][config_key])
        stage.load_state_dict(
            {
                k[len(prefix) :]: v
                for k, v in before["state_dict"].items()
                if k.startswith(prefix)
            },
            strict=True,
        )
        stages.append(stage.to(args.device).eval())
    old_render = importlib.import_module(package + ".renderer").render_rgbt
    del before, checkpoint

    def forward(inputs, valid, version):
        cv2.setRNGSeed(20260920)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if version == "after":
                return model(*inputs, source_valid=valid)
            result = model.reconstruction(*inputs)
            for stage in stages:
                result = stage(*inputs, initial=result, source_valid=valid)
            return result

    report = {
        "before": args.before,
        "before_source": args.before_source,
        "after": args.after,
        "camera": args.camera,
        "weights_exactly_preserved": True,
        "output_tolerance": {"atol": 3e-5, "rtol": 3e-5},
        "cases": [],
        "runtime": {},
        "note": "Snapshot refinement vs fused refinement; shared unchanged reconstruction. Predicted-camera rendering uses source camera 0, not target accuracy.",
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    for root, split, views, dataset, index, scene in cases(
        args.data, args.targets, [3, 6], ["train", "val", "test"]
    ):
        if args.benchmark and (Path(scene).name != "DailyStuff" or index != 0):
            continue
        item = dataset[index]
        batch = batch_to(item, args.device)
        known = args.camera == "known"
        inputs = [
            batch["rgb"][:, :views],
            batch["thermal"][:, :views],
            batch["E"][:, :views] if known else None,
            batch["K"][:, :views] if known else None,
        ]
        valid = batch["valid"][:, :views]
        original = forward(inputs, valid, "before")
        fused = forward(inputs, valid, "after")
        errors = compared(output_tensors(original), output_tensors(fused))
        target_E = batch["E"][:, views:] if known else original.extrinsics[:, :1]
        target_K = batch["K"][:, views:] if known else original.intrinsics[:, :1]
        shape = original.depth.shape[-2:]
        before_render = old_render(original, target_E, target_K, *shape)
        after_render = render_rgbt(fused, target_E, target_K, *shape)
        errors.update(
            {
                "render." + k: value
                for k, value in compared(before_render, after_render).items()
            }
        )
        metrics = {}
        if known:
            for modality in ("rgb", "thermal"):
                metrics[modality] = {
                    version: psnr(
                        render[modality].clamp(0, 1),
                        batch[modality][:, views:],
                        batch["valid"][:, views:],
                    )
                    for version, render in (
                        ("before", before_render),
                        ("after", after_render),
                    )
                }
        row = {
            "scene": scene,
            "split": split,
            "root": root,
            "views": views,
            "target": item["id"],
            "max_absolute_errors": errors,
            "psnr": metrics,
        }
        report["cases"].append(row)
        print(scene, views, args.camera, "max_error", max(errors.values()), flush=True)
        del original, fused, before_render, after_render
        if args.benchmark:
            times = {"before": [], "after": []}
            memory = {}
            triangulations = {}
            for version in times:
                modules = (
                    [package + ".calibration", package + ".source_anchors"]
                    if version == "before"
                    else ["depth_anything_3.rgbt.source_geometry"]
                )
                with ExitStack() as stack:
                    counters = []
                    for module_name in modules:
                        module = importlib.import_module(module_name)
                        counters.append(
                            stack.enter_context(
                                patch.object(
                                    module,
                                    "triangulate_sources",
                                    wraps=module.triangulate_sources,
                                )
                            )
                        )
                    warm = forward(inputs, valid, version)
                    del warm
                    triangulations[version] = sum(
                        counter.call_count for counter in counters
                    )
            assert triangulations == {"before": 3 if known else 2, "after": 1}
            for repeat in range(args.repeats):
                for version in (
                    ("before", "after") if repeat % 2 == 0 else ("after", "before")
                ):
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    start = time.perf_counter()
                    result = forward(inputs, valid, version)
                    torch.cuda.synchronize()
                    times[version].append(time.perf_counter() - start)
                    memory[version] = max(
                        memory.get(version, 0), torch.cuda.max_memory_allocated()
                    )
                    del result
            report["runtime"][str(views)] = {
                version: {
                    "median_seconds": float(np.median(values)),
                    "seconds": values,
                    "peak_allocated_bytes": memory[version],
                    "source_triangulations": triangulations[version],
                }
                for version, values in times.items()
            }
            print("runtime", views, report["runtime"][str(views)], flush=True)
        report["scenes"] = sorted({row["scene"] for row in report["cases"]})
        report["maximum_absolute_error"] = max(
            max(row["max_absolute_errors"].values()) for row in report["cases"]
        )
        path.write_text(json.dumps(report, indent=2) + "\n")
    assert report["cases"], "No cases selected"
    print(
        "VERIFIED",
        len(report["cases"]),
        "cases in",
        len(report["scenes"]),
        "scenes:",
        path,
        flush=True,
    )


if __name__ == "__main__":
    main()
