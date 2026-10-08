#!/usr/bin/env python3
"""Paired source-only, inference-time ablations of the selected RGB-T checkpoint.

No variant is retrained. Shared prefixes are reused only when their computation
is identical. Targets and sparse depths are passed to metrics, never the model.
Rows are flushed after each variant and can be resumed with the same arguments.
"""

import argparse
import hashlib
import json
import random
import sys
import time
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import cv2
import numpy as np
import torch

from depth_anything_3.rgbt.checkpoint import load_rgbt_model
from depth_anything_3.rgbt.data import SceneDataset
from depth_anything_3.rgbt.engine import batch_to, panel, psnr, ssim
from depth_anything_3.rgbt.geometry import gaussian_depth
from depth_anything_3.rgbt.renderer import render_rgbt
from depth_anything_3.rgbt.source_anchors import source_anchor_field
from depth_anything_3.rgbt.surface_consensus import anchor_consensus, build_consensus

# Changes are made to module attributes, not just a disconnected JSON config.
VARIANTS = {
    "full": {},
    "no_adapters": {"reconstruction.adapters_enabled": False},
    "no_geometry_residual": {"reconstruction.geometry_strength": 0.0},
    "no_rgb_source_blend": {"reconstruction.rgb_source_blend": 0.0},
    "no_neighborhood": {"skip_neighborhood": True},
    "no_crossview": {"skip_crossview": True},
    "no_surface": {"skip_surface": True},
    "reconstruction_only": {
        "skip_neighborhood": True,
        "skip_crossview": True,
        "skip_surface": True,
    },
    "da3_rgbt_wrapper": {
        "teacher": True,
        "reconstruction.geometry_mode": "legacy",
        "reconstruction.rgb_source_blend": 0.0,
        "skip_neighborhood": True,
        "skip_crossview": True,
        "skip_surface": True,
    },
    "anchored_da3": {
        "teacher": True,
        "reconstruction.rgb_source_blend": 0.0,
        "skip_neighborhood": True,
        "skip_crossview": True,
        "skip_surface": True,
    },
    "no_neighborhood_feedback": {"neighborhood_refiner.render_feedback": False},
    "no_neighborhood_guard": {"neighborhood_refiner.source_guard": False},
    "no_crossview_feedback": {"crossview_refiner.render_feedback": False},
    "no_crossview_calibration": {"crossview_refiner.calibration_strength": 0.0},
    "no_crossview_trust": {"crossview_refiner.trust_mode": "none"},
    "no_crossview_thermal_conditioning": {"crossview_refiner.refiner_thermal": False},
    "no_plane_learning": {"surface_refiner.learn_planes": False},
    "no_plane_covariance": {"surface_refiner.plane_cov": False},
    "no_plane_anchoring": {"surface_refiner.anchored": False},
    "no_surface_guard": {"surface_refiner.geometry_guard": False},
    "no_surface_fusion": {
        "surface_refiner.thermal_fusion": 0.0,
        "surface_refiner.rgb_fusion": 0.0,
    },
    "no_surface_appearance": {"surface_refiner.appearance_strength": 0.0},
    "no_appearance_trust": {"surface_refiner.appearance_trust": False},
    "no_dense_depth": {"surface_refiner.dense_depth_limit": 0.0},
    "no_source_anchors": {"surface_refiner.anchor_strength": 0.0},
    "no_rgb_feedback": {"surface_refiner.rgb_feedback_step": 0.0},
    "no_thermal_feedback": {"surface_refiner.thermal_feedback_step": 0.0},
    "no_thermal_consensus": {"surface_refiner.thermal_feedback_consensus": False},
    "no_thermal_blur": {"surface_refiner.thermal_feedback_blur": 0},
}


@contextmanager
def configured(model, changes):
    originals = []
    try:
        for key, value in changes.items():
            if "." not in key:
                continue
            *parents, attr = key.split(".")
            module = model
            for parent in parents:
                module = getattr(module, parent)
            current = getattr(module, attr)
            if isinstance(current, torch.Tensor):
                originals.append((module, attr, current.detach().clone()))
                current.fill_(value)
            else:
                originals.append((module, attr, current))
                setattr(module, attr, value)
        yield
    finally:
        for module, attr, value in originals:
            if isinstance(value, torch.Tensor):
                getattr(module, attr).copy_(value)
            else:
                setattr(module, attr, value)


class PairedForward:
    def __init__(self, model, batch, views):
        self.model = model
        self.args = [batch[k][:, :views] for k in ["rgb", "thermal", "E", "K"]]
        self.valid = batch["valid"][:, :views]
        self.cache = {}
        self.associations = {}
        self.fixed_association = None

    def run(self, changes):
        m = self.model
        parts = []
        initial = None
        with configured(m, changes), torch.autocast("cuda", dtype=torch.bfloat16):
            for name, skip in [
                ("reconstruction", False),
                ("neighborhood_refiner", changes.get("skip_neighborhood", False)),
                ("crossview_refiner", changes.get("skip_crossview", False)),
            ]:
                if skip:
                    parts.append((name, "skip"))
                    continue
                opts = {k: v for k, v in changes.items() if k.startswith(name + ".")}
                if name == "reconstruction" and changes.get("teacher"):
                    opts["teacher"] = True
                parts.append((name, json.dumps(opts, sort_keys=True)))
                key = tuple(parts)
                if key not in self.cache:
                    if name == "reconstruction":
                        output = m.reconstruction(
                            *self.args, teacher=changes.get("teacher", False)
                        )
                    else:
                        output = getattr(m, name)(
                            *self.args, initial=initial, source_valid=self.valid
                        )
                    # Keep standard prefixes; alternate paths are used once and
                    # must not accumulate many full-resolution Gaussian arrays.
                    if (
                        not opts
                        and not any("skip" == p[1] for p in parts)
                        and len(parts) <= 3
                    ):
                        self.cache[key] = output
                    initial = output
                else:
                    initial = self.cache[key]
            if changes.get("skip_surface"):
                return initial
            assoc_key = (tuple(parts), bool(m.surface_refiner.anchored))
            if assoc_key not in self.associations:
                assoc = build_consensus(initial, self.args[0], self.valid)
                if m.surface_refiner.anchored:
                    assoc = anchor_consensus(initial, self.args[0], assoc)
                # Precompute only source observations; independent of learned
                # surface updates and of the evaluation target.
                if m.surface_refiner.anchor_strength:
                    delta, counts = source_anchor_field(
                        initial,
                        self.args[0],
                        self.valid,
                        m.surface_refiner.anchor_radius,
                    )
                    assoc.update(
                        source_anchor_delta=delta,
                        source_anchor_counts=counts,
                        source_anchor_radius=m.surface_refiner.anchor_radius,
                    )
                if all(
                    not json.loads(p[1]) for p in parts if p[1] != "skip"
                ) and not any(p[1] == "skip" for p in parts):
                    self.associations[assoc_key] = assoc
            else:
                assoc = self.associations[assoc_key]
            if not changes:
                self.fixed_association = deepcopy(assoc)
            return m.surface_refiner(
                *self.args,
                initial=initial,
                source_valid=self.valid,
                association=deepcopy(assoc),
            )


def metrics(output, batch, views, perceptual, fixed_association=None):
    h, w = output.depth.shape[-2:]
    pred = render_rgbt(output, batch["E"][:, views:], batch["K"][:, views:], h, w)
    mask = batch["valid"][:, views:]
    result = {}
    for modality in ["rgb", "thermal"]:
        image = pred[modality].clamp(0, 1)
        target = batch[modality][:, views:]
        result[modality + "_psnr"] = psnr(image, target, mask)
        result[modality + "_ssim"] = float(ssim(image * mask, target * mask))
    z, gt = gaussian_depth(output), batch["depth"][:, :views]
    valid_depth = (gt > 0) & torch.isfinite(gt) & torch.isfinite(z) & (z > 0)
    result["depth_points"] = int(valid_depth.sum())
    result["depth_log_error"] = (
        float((z[valid_depth].log() - gt[valid_depth].log()).clamp(-3, 3).abs().mean())
        if result["depth_points"] >= 16
        else None
    )
    result["alpha_mean"] = float((pred["alpha"] * mask).sum() / mask.sum().clamp_min(1))
    if fixed_association is not None:
        points = output.gaussians.means.float().reshape(-1, 3)
        ids = fixed_association["ids"].reshape(-1)
        thicknesses = []
        for index, group in enumerate(fixed_association["groups"]):
            selected = points[ids == index]
            if len(selected) < 2:
                continue
            residual = selected @ fixed_association["normals"][index]
            thicknesses.append(
                float(
                    (residual.quantile(0.95) - residual.quantile(0.05))
                    / group["radius"]
                )
            )
        result["fixed_surface_thickness"] = (
            float(np.mean(thicknesses)) if thicknesses else None
        )
        result["fixed_surface_groups"] = len(thicknesses)
    if perceptual is not None:
        result["rgb_lpips"] = float(
            perceptual(
                (pred["rgb"].clamp(0, 1) * mask).flatten(0, 1),
                (batch["rgb"][:, views:] * mask).flatten(0, 1),
                normalize=True,
            ).mean()
        )
    if not all(np.isfinite(value) for value in result.values() if value is not None):
        raise RuntimeError("Non-finite evaluation metric")
    return result, pred


def cases(roots, targets, views, splits):
    for root in roots:
        for split in splits:
            for v in views:
                ds = SceneDataset(
                    root,
                    split,
                    views=v,
                    sampling="nearest" if v == 3 else "wide",
                    targets_per_scene=targets,
                )
                for i, (si, _) in enumerate(ds.items):
                    yield str(root), split, v, ds, i, ds.scenes[si]["id"]


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--weights", default="outputs/rgbt/ablation_20260920/before/model.pt"
    )
    p.add_argument(
        "--data",
        nargs="+",
        default=[
            "outputs/rgbt/data_336",
            "outputs/rgbt/datasets/external_336",
            "outputs/rgbt/datasets/heldout_336",
            "outputs/rgbt/datasets/plant_336",
        ],
    )
    p.add_argument("--split", nargs="+", default=["train", "val", "test"])
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--targets", type=int, default=3)
    p.add_argument("--views", nargs="+", type=int, default=[3, 6])
    p.add_argument("--variants", nargs="+", default=list(VARIANTS))
    p.add_argument(
        "--variants-file",
        help="JSON mapping extra variant names to attribute overrides",
    )
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--shards", type=int, default=1)
    p.add_argument("--limit", type=int)
    p.add_argument("--lpips", action="store_true")
    p.add_argument("--surface-metrics", action="store_true")
    p.add_argument("--panels", nargs="*", default=[])
    a = p.parse_args()
    torch.set_num_threads(4)
    cv2.setNumThreads(2)
    torch.cuda.set_device(a.device)
    torch.manual_seed(20260920)
    np.random.seed(20260920)
    random.seed(20260920)
    cv2.setRNGSeed(20260920)
    if a.variants_file:
        VARIANTS.update(json.loads(Path(a.variants_file).read_text()))
    variants = {name: VARIANTS[name] for name in a.variants}
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    arguments = vars(a).copy()
    if not a.surface_metrics:
        arguments.pop("surface_metrics")
    protocol = {
        "arguments": arguments,
        "variants": variants,
        "evaluation": "Frozen-weight removal; no retraining. Equal scene weights; paired sources/targets.",
        "baseline": "da3_rgbt_wrapper: frozen original DA3 weights, no thermal adapters/residuals, legacy Gaussian adapter; source camera scale alignment and thermal sampling wrapper. Not unmodified official API.",
        "manifest_hashes": {
            root: hashlib.sha256(
                (Path(root) / "manifest.json").read_bytes()
            ).hexdigest()
            for root in a.data
        },
    }
    protocol_path = out / f"protocol_{a.shard}.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError("Refusing to mix different experiment protocols in one shard")
    protocol_path.write_text(json.dumps(protocol, indent=2) + "\n")
    model, checkpoint = load_rgbt_model(a.weights, a.device)
    del checkpoint
    perceptual = None
    if a.lpips:
        import lpips

        perceptual = lpips.LPIPS(net="alex").to(a.device).eval()
    dest = out / f"rows_{a.shard}.jsonl"
    done = set()
    if dest.exists():
        for line in dest.read_text().splitlines():
            r = json.loads(line)
            done.add((r["data"], r["id"], r["views"], r["variant"]))
    all_cases = [
        c
        for j, c in enumerate(cases(a.data, a.targets, a.views, a.split))
        if j % a.shards == a.shard
    ]
    if a.limit is not None:
        all_cases = all_cases[: a.limit]
    with dest.open("a", buffering=1) as stream:
        for j, (root, split, views, ds, i, scene) in enumerate(all_cases):
            item = ds[i]
            pending = [
                name for name in variants if (root, item["id"], views, name) not in done
            ]
            if not pending:
                continue
            batch = batch_to(item, a.device)
            forward = PairedForward(model, batch, views)
            if a.surface_metrics and "full" not in pending:
                forward.run({})
            started = time.perf_counter()
            for name in pending:
                # Reset randomized correspondence solvers for paired comparison.
                cv2.setRNGSeed(20260920)
                output = forward.run(variants[name])
                values, pred = metrics(
                    output,
                    batch,
                    views,
                    perceptual,
                    forward.fixed_association if a.surface_metrics else None,
                )
                row = dict(
                    data=root,
                    split=split,
                    scene=scene,
                    id=item["id"],
                    views=views,
                    context_ids=item["context_ids"],
                    variant=name,
                    **values,
                )
                stream.write(json.dumps(row, allow_nan=False) + "\n")
                if name in a.panels:
                    folder = out / "panels" / str(views) / name
                    folder.mkdir(parents=True, exist_ok=True)
                    panel(folder / (item["id"] + ".jpg"), batch, pred)
                del output, pred
            del forward, batch
            print(
                json.dumps(
                    {
                        "case": j + 1,
                        "total": len(all_cases),
                        "scene": scene,
                        "views": views,
                        "variants": len(pending),
                        "seconds": round(time.perf_counter() - started, 2),
                    }
                ),
                flush=True,
            )
    print("COMPLETE", dest, flush=True)


if __name__ == "__main__":
    main()
