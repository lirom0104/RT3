#!/usr/bin/env python3
"""Distill a single shared RGB-T refiner with frozen reconstruction and source priors."""

import argparse
import json
import random
import sys
import time
from dataclasses import replace
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import cv2
import numpy as np
import torch
from cache_rgbt_joint import move

from depth_anything_3.rgbt.engine import batch_to, depth_loss, psnr, ssim
from depth_anything_3.rgbt.geometry import gaussian_depth
from depth_anything_3.rgbt.learned_joint import LearnedJointRefiner
from depth_anything_3.rgbt.renderer import render_rgbt
from depth_anything_3.utils.geometry import quat_to_mat


@lru_cache(maxsize=8)
def read(path):
    return torch.load(path, map_location="cpu", weights_only=False, mmap=True)


def prepare(path, device, phase=None):
    data = read(str(path))
    b = batch_to(data["item"], device)
    if data["format"] == "da3-rgbt-ordered-joint-v1":
        base = replace(move(data["initial"], device), fused_rendering=True)
        packed = data["phases"][0 if phase is None else phase]
        prepared = move(packed["prepared"], device)
        for key in ["base", "reference"]:
            if key in prepared:
                state = prepared[key]
                prepared[key] = replace(
                    teacher_output(base, state), depth=state["depth"]
                )
        prepared["phase_training"] = phase is not None
        observations = (
            Path(path).parent.parent / "source_observations" / Path(path).name
        )
        if phase is None and observations.exists():
            prepared["source_observations"] = read(str(observations))
        if phase is not None and "target_delta" in packed:
            prepared["target_delta"] = move(packed["target_delta"], device)
            prepared["delta_weight"] = move(packed["delta_weight"], device)
        target = data if phase is None else packed
        return (
            b,
            prepared,
            move(target["teacher"], device),
            move(target["teacher_render"], device),
        )
    prepared = move(data["prepared"], device)
    if "geometry_context" not in prepared:
        prepared["geometry_context"] = LearnedJointRefiner.geometry_features(
            move(data["initial"], device), prepared
        )
    # features, cameras and depth are the same as the raw initial reconstruction.
    return (
        b,
        prepared,
        move(data["teacher"], device),
        move(data["teacher_render"], device),
    )


def teacher_output(base, teacher):
    sh = base.gaussians.harmonics.clone()
    sh[..., 0] = teacher["dc"]
    gs = replace(
        base.gaussians,
        harmonics=sh,
        **{k: teacher[k] for k in ["means", "scales", "rotations", "opacities"]},
    )
    return replace(base, gaussians=gs, thermal_colors=teacher["thermal"])


def forward(model, b, prepared, proposal=False):
    base = prepared["base"]
    v = base.depth.shape[1]
    return model(
        b["rgb"][:, :v],
        b["thermal"][:, :v],
        b["E"][:, :v],
        b["K"][:, :v],
        initial=base,
        prepared=prepared,
        source_valid=b["valid"][:, :v],
        return_proposal=proposal,
    )


def parameter_loss(output, teacher, base):
    gs = output.gaussians
    z = (
        gaussian_depth(base)
        .detach()
        .clamp_min(1e-4)
        .reshape(base.depth.shape[0], -1, 1)
    )
    return {
        "position": 1e4 * ((gs.means - teacher["means"]) / z).square().mean(),
        "scale": (
            gs.scales.clamp_min(1e-6).log() - teacher["scales"].clamp_min(1e-6).log()
        )
        .square()
        .mean(),
        "rotation": (
            1 - (gs.rotations * teacher["rotations"]).sum(-1).square().clamp(0, 1)
        ).mean(),
        "opacity": (gs.opacities - teacher["opacities"]).square().mean(),
        "color": 20
        * ((gs.harmonics[..., 0] - teacher["dc"]) * 0.28209479177387814)
        .square()
        .mean(),
        "thermal": 20 * (output.thermal_colors - teacher["thermal"]).square().mean(),
    }


def image_mse(image, target, mask):
    return ((image - target).square() * mask).sum() / (
        mask.sum() * image.shape[2]
    ).clamp_min(1)


@torch.no_grad()
def evaluate(model, paths, device, perceptual):
    model.eval()
    model.core.capture_delta = False
    rows = []
    for path in paths:
        b, p, t, tr = prepare(path, device)
        v = p["base"].depth.shape[1]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = forward(model, b, p)
        rendered = render_rgbt(
            output, b["E"][:, v:], b["K"][:, v:], *output.depth.shape[-2:]
        )
        mask = b["valid"][:, v:]
        row = {"scene": b["scene"], "target": b["id"], "views": v}
        for tag, pred, obj in [
            ("teacher", tr, teacher_output(p["base"], t)),
            ("student", rendered, output),
        ]:
            vals = {}
            for m in ["rgb", "thermal"]:
                vals[m + "_psnr"] = psnr(pred[m].clamp(0, 1), b[m][:, v:], mask)
                vals[m + "_ssim"] = float(
                    ssim(pred[m].clamp(0, 1) * mask, b[m][:, v:] * mask)
                )
            vals["lpips"] = float(
                perceptual(
                    (pred["rgb"].clamp(0, 1) * mask).flatten(0, 1),
                    (b["rgb"][:, v:] * mask).flatten(0, 1),
                    normalize=True,
                ).mean()
            )
            vals["depth_log_error"] = float(
                depth_loss(gaussian_depth(obj), b["depth"][:, :v])
            )
            row[tag] = vals
        row["delta"] = {
            k: row["student"][k] - row["teacher"][k] for k in row["teacher"]
        }
        rows.append(row)
    scene_rows = []
    for scene, views in sorted({(r["scene"], r["views"]) for r in rows}):
        selected = [r for r in rows if r["scene"] == scene and r["views"] == views]
        scene_rows.append(
            {
                "scene": scene,
                "views": views,
                **{
                    tag: {
                        k: float(np.mean([r[tag][k] for r in selected]))
                        for k in rows[0][tag]
                    }
                    for tag in ["teacher", "student", "delta"]
                },
            }
        )
    summary = {
        k: float(np.mean([r["delta"][k] for r in scene_rows])) for k in rows[0]["delta"]
    }
    # Worst RGB/T scene matters in candidate selection, not just the global average.
    worst = min(r["delta"][m + "_psnr"] for r in scene_rows for m in ["rgb", "thermal"])
    score = (
        worst
        + 0.25 * (summary["rgb_psnr"] + summary["thermal_psnr"])
        - 5 * summary["lpips"]
    )
    policy = json.loads(
        (ROOT / "outputs/rgbt/joint_training_20260920/quality_policy.json").read_text()
    )
    failures = []
    for label, group, rules in [
        ("scene", scene_rows, policy["required_scene_protocol_deltas"]),
        ("target", rows, policy["required_per_target_deltas"]),
    ]:
        for row in group:
            for rule, limit in rules.items():
                name, bound = rule.rsplit("_", 1)
                d = row["delta"][name]
                if (bound == "min" and d < limit) or (bound == "max" and d > limit):
                    failures.append(
                        {
                            "level": label,
                            "scene": row["scene"],
                            "views": row["views"],
                            "target": row.get("target"),
                            "metric": name,
                            "delta": d,
                            "limit": limit,
                        }
                    )
    return {
        "summary_delta": summary,
        "worst_scene_psnr_delta": worst,
        "score": score,
        "passed": not failures,
        "failures": failures,
        "per_scene": scene_rows,
        "per_target": rows,
    }


def save(path, model, optimizer, step, best, args):
    tmp = path.with_suffix(".tmp")
    torch.save(
        {
            "format": "da3-rgbt-learned-refiner-training",
            "refiner_config": model.config,
            "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "optimizer": optimizer.state_dict(),
            "step": step,
            "best_score": best,
            "args": vars(args),
            "python_rng": random.getstate(),
            "numpy_rng": np.random.get_state(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state(),
        },
        tmp,
    )
    tmp.replace(path)


def configured_args(parser, argv=None):
    """Restore the recipe on resume; explicit command-line options take precedence."""
    probe, _ = parser.parse_known_args(argv)
    defaults = {}
    if probe.config:
        defaults.update(json.loads(Path(probe.config).read_text()))
    if probe.resume:
        checkpoint = torch.load(probe.resume, map_location="cpu", weights_only=False)
        if checkpoint.get("format") != "da3-rgbt-learned-refiner-training":
            raise ValueError("Resume requires a shared-refiner training checkpoint")
        defaults.update(checkpoint["args"])
    # Execution controls belong to this invocation, not the previous run.
    for key in ["output", "device", "resume", "config", "smoke", "evaluate_only"]:
        defaults.pop(key, None)
    unknown = set(defaults) - {action.dest for action in parser._actions}
    if unknown:
        raise ValueError(f"Unknown training configuration keys: {sorted(unknown)}")
    parser.set_defaults(**defaults)
    return parser.parse_args(argv)


def configure_training(model):
    """Keep step zero fixed while training the weights shared by later updates."""
    if not model.config.get("preserve_first_update"):
        return
    model.requires_grad_(False)
    for module in [
        model.core.blocks[1],
        model.core.blocks[3],
        model.core.phase_embedding,
        model.core.ordered_projection,
        model.core.head[-1],
    ]:
        module.requires_grad_(True)
    for name in ["phase_modulation", "shared_residual"]:
        if hasattr(model.core, name):
            getattr(model.core, name).requires_grad_(True)
    for name, parameter in model.named_parameters():
        if name.startswith(("core.phase_embedding.", "core.phase_modulation.")):
            parameter.register_hook(
                lambda gradient: torch.cat(
                    [torch.zeros_like(gradient[:1]), gradient[1:]], 0
                )
            )
        elif name.startswith("core.head.2."):
            mask = torch.ones_like(parameter)
            mask[[3, 4, 5, 10, 14]] = 0
            parameter.register_hook(lambda gradient, mask=mask: gradient * mask)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument(
        "--config", help="JSON training recipe; CLI options override it"
    )
    parser.add_argument(
        "--train-cache", default="outputs/rgbt/joint_training_20260920/cache_train"
    )
    parser.add_argument(
        "--val-cache", default="outputs/rgbt/joint_training_20260920/cache_val"
    )
    parser.add_argument(
        "--teacher", default="outputs/rgbt/joint_training_20260920/before/model.pt"
    )
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--attention-pairs", type=int, default=1)
    parser.add_argument("--channels", type=int, default=96)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--global-context", action="store_true")
    parser.add_argument("--geometry-context", action="store_true")
    parser.add_argument("--ordered-updates", action="store_true")
    parser.add_argument("--phase-modulation", action="store_true")
    parser.add_argument("--preserve-first-update", action="store_true")
    parser.add_argument("--shared-residual", action="store_true")
    parser.add_argument("--rollout-final-probability", type=float, default=0.0)
    parser.add_argument("--quality-preservation-weight", type=float, default=0.0)
    parser.add_argument("--update-distill-weight", type=float, default=0.0)
    parser.add_argument(
        "--phase-loss-weights", type=float, nargs=3, default=[1.0, 1.0, 1.0]
    )
    parser.add_argument(
        "--phase-sampling-weights", type=float, nargs=3, default=[1.0, 1.0, 1.0]
    )
    parser.add_argument("--phase-lr-multiplier", type=float, default=1.0)
    parser.add_argument("--covariance-distill", action="store_true")
    parser.add_argument("--distill-weight", type=float, default=1.0)
    parser.add_argument("--image-weight", type=float, default=1.0)
    parser.add_argument("--teacher-image-weight", type=float, default=4.0)
    parser.add_argument("--no-trust", action="store_true")
    parser.add_argument("--flexible-geometry", action="store_true")
    parser.add_argument("--robust-distill", action="store_true")
    parser.add_argument("--perceptual-weight", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument("--resume")
    parser.add_argument("--init-refiner")
    parser.add_argument(
        "--distill-only",
        action="store_true",
        help="Match teacher outputs; do not optimize ground truth quality beyond teacher",
    )
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = configured_args(parser)
    torch.set_num_threads(4)
    cv2.setNumThreads(2)
    torch.cuda.set_device(args.device)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    out = Path(args.output)
    if out.exists() and (out / "config.json").exists() and not args.resume:
        raise ValueError("Use a fresh training directory")
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(args), indent=2) + "\n")
    train_paths = sorted(Path(args.train_cache).glob("*.pt"))
    val_paths = sorted(Path(args.val_cache).glob("*.pt"))
    if not train_paths or (not val_paths and not args.smoke):
        raise ValueError("Training and validation caches must be complete")
    byscene = {}
    for path in train_paths:
        data = read(str(path))
        assert data["split"] == "train"
        byscene.setdefault(data["item"]["scene"], []).append(path)
    assert len(byscene) == 14, f"Expected 14 training scenes, got {len(byscene)}"
    for path in val_paths:
        data = read(str(path))
        assert data["split"] == "val" and data["item"]["scene"] not in byscene
    if args.resume:
        resume = torch.load(args.resume, map_location="cpu", weights_only=False)
        model = LearnedJointRefiner(**resume["refiner_config"])
        model.load_state_dict(resume["state_dict"], strict=True)
        for key in model.config:
            if hasattr(args, key) and getattr(args, key) != model.config[key]:
                raise ValueError(
                    f"Cannot change architecture option {key} when resuming"
                )
    else:
        model = LearnedJointRefiner(
            channels=args.channels,
            attention_pairs=args.attention_pairs,
            trust=not args.no_trust,
            flexible_geometry=args.flexible_geometry,
            iterations=args.iterations,
            global_context=args.global_context,
            geometry_context=args.geometry_context,
            ordered_updates=args.ordered_updates,
            phase_modulation=args.phase_modulation,
            preserve_first_update=args.preserve_first_update,
            shared_residual=args.shared_residual,
        )
        if args.init_refiner:
            warm = torch.load(args.init_refiner, map_location="cpu", weights_only=False)
            if warm.get("refiner_config", {}).get("channels", 96) != args.channels:
                raise ValueError("Widen checkpoint first with widen_rgbt_joint.py")
            missing, unexpected = model.load_state_dict(
                warm["state_dict"], strict=False
            )
            assert not unexpected
            assert all(
                name.startswith(
                    (
                        "core.global_context.",
                        "core.geometry_context.",
                        "core.phase_embedding.",
                        "core.phase_modulation.",
                        "core.shared_residual.",
                        "core.ordered_projection.",
                    )
                )
                for name in missing
            ), missing
        else:
            from depth_anything_3.rgbt.legacy.crossview import CrossViewRefiner
            from depth_anything_3.rgbt.legacy.refinement import NeighborhoodRefiner

            teacher = torch.load(
                args.teacher, map_location="cpu", weights_only=False, mmap=True
            )
            modules = []
            for cls, key, prefix in [
                (NeighborhoodRefiner, "neighborhood", "refiner.local."),
                (CrossViewRefiner, "crossview", "refiner.crossview."),
            ]:
                module = cls(**teacher["model_config"][key])
                module.load_state_dict(
                    {
                        k[len(prefix) :]: v
                        for k, v in teacher["state_dict"].items()
                        if k.startswith(prefix)
                    },
                    strict=True,
                )
                modules.append(module)
            model.initialize_from_teacher(
                SimpleNamespace(
                    neighborhood_refiner=modules[0], crossview_refiner=modules[1]
                )
            )
            del teacher, modules, module
    model = model.to(args.device)
    configure_training(model)
    if model.config.get("preserve_first_update"):
        print(
            "PRESERVED FIRST UPDATE; trainable",
            sum(p.numel() for p in model.parameters() if p.requires_grad),
            flush=True,
        )
    parameters = model.parameters()
    if args.phase_lr_multiplier != 1:
        assert model.config.get("ordered_updates")
        phase_parameters = list(model.core.phase_embedding.parameters())
        if hasattr(model.core, "phase_modulation"):
            phase_parameters += list(model.core.phase_modulation.parameters())
        phase_ids = {id(p) for p in phase_parameters}
        parameters = [
            {
                "params": [p for p in model.parameters() if id(p) not in phase_ids],
                "lr_scale": 1.0,
            },
            {
                "params": phase_parameters,
                "lr_scale": args.phase_lr_multiplier,
                "lr": args.lr * args.phase_lr_multiplier,
            },
        ]
    optimizer = torch.optim.AdamW(
        parameters, lr=args.lr, weight_decay=0 if args.preserve_first_update else 0.01
    )
    best, step0 = -float("inf"), 0
    if args.resume:
        optimizer.load_state_dict(resume["optimizer"])
        best, step0 = resume["best_score"], resume["step"]
        random.setstate(resume["python_rng"])
        np.random.set_state(resume["numpy_rng"])
        torch.set_rng_state(resume["torch_rng"])
        torch.cuda.set_rng_state(resume["cuda_rng"], args.device)
        del resume
    import lpips

    perceptual = lpips.LPIPS(net="alex").to(args.device).eval().requires_grad_(False)
    print(
        "PARAMETERS",
        sum(p.numel() for p in model.parameters()),
        "TRAIN",
        len(train_paths),
        "VAL",
        len(val_paths),
        flush=True,
    )
    if not args.smoke:
        baseline = evaluate(model, val_paths, args.device, perceptual)
        if not args.resume:
            best = baseline["score"]
            save(out / "step_0.pt", model, optimizer, 0, best, args)
            save(out / "best.pt", model, optimizer, 0, best, args)
        (out / f"val_{step0}.json").write_text(json.dumps(baseline, indent=2) + "\n")
        print(
            "VAL",
            step0,
            json.dumps(baseline["summary_delta"]),
            "worst",
            baseline["worst_scene_psnr_delta"],
            flush=True,
        )
    if args.evaluate_only:
        return
    start = time.time()
    for step in range(step0 + 1, args.steps + 1):
        model.train()
        scene = random.choice(sorted(byscene))
        path = random.choice(byscene[scene])
        phase = None
        if model.config.get("ordered_updates"):
            phase = (
                random.randrange(3)
                if args.phase_sampling_weights == [1.0, 1.0, 1.0]
                else random.choices(range(3), weights=args.phase_sampling_weights, k=1)[
                    0
                ]
            )
        b, p, t, tr = prepare(path, args.device, phase)
        rollout_final = False
        if (
            model.config.get("ordered_updates")
            and args.rollout_final_probability
            and random.random() < args.rollout_final_probability
        ):
            b, p, t, tr = prepare(path, args.device, None)
            rollout_final = True
        v = p["base"].depth.shape[1]
        h, w = p["base"].depth.shape[-2:]
        optimizer.zero_grad(set_to_none=True)
        model.core.capture_delta = bool(
            args.update_distill_weight and not rollout_final
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output, proposal = forward(model, b, p, True)
        terms = parameter_loss(output, t, p["base"])
        pd = parameter_loss(proposal, t, p["base"])
        if args.robust_distill:
            # A few differently associated surfaces must not dominate shared features.
            z = gaussian_depth(p["base"]).detach().clamp_min(1e-4).reshape(1, -1, 1)
            for obj, losses in [(output, terms), (proposal, pd)]:
                error = (obj.gaussians.means - t["means"]) / z
                losses["position"] = 100 * torch.nn.functional.smooth_l1_loss(
                    error, torch.zeros_like(error), beta=0.01
                )
        if args.covariance_distill:

            def covariance(scales, rotations):
                R = quat_to_mat(rotations[..., [1, 2, 3, 0]])
                return (R * scales.square()[..., None, :]) @ R.transpose(-1, -2)

            expected = covariance(t["scales"][:, ::8], t["rotations"][:, ::8])
            normalization = expected.square().sum((-1, -2)).sqrt().clamp_min(1e-10)
            for obj, losses in [(output, terms), (proposal, pd)]:
                actual = covariance(
                    obj.gaussians.scales[:, ::8], obj.gaussians.rotations[:, ::8]
                )
                error = (actual - expected) / normalization[..., None, None]
                losses["scale"] = torch.nn.functional.smooth_l1_loss(
                    error, torch.zeros_like(error), beta=0.1
                )
                losses["rotation"] = losses["scale"] * 0
        # Ordered proposals are supervised by the teacher's actual update logits.
        # Their unguarded parameters must not also be pulled toward guarded output.
        proposal_weight = 0 if model.config.get("ordered_updates") else 0.25
        distill = sum(terms.values()) + proposal_weight * sum(pd.values())
        pred = render_rgbt(output, b["E"][:, v:], b["K"][:, v:], h, w)
        mask = b["valid"][:, v:]
        image = pred["rgb"].new_zeros(())
        imitation = image.clone()
        for mod, weight in [("rgb", 1.0), ("thermal", 4.0)]:
            image += weight * (
                10 * image_mse(pred[mod], b[mod][:, v:], mask)
                + 0.2 * (1 - ssim(pred[mod] * mask, b[mod][:, v:] * mask))
            )
            imitation += weight * 10 * image_mse(pred[mod], tr[mod], mask)
            if args.distill_only:
                imitation += weight * 0.2 * (1 - ssim(pred[mod] * mask, tr[mod] * mask))
        teacher_obj = teacher_output(p["base"], t)
        perceptual_loss = pred["rgb"].new_zeros(())
        if args.perceptual_weight:
            perceptual_loss = perceptual(
                (pred["rgb"].clamp(0, 1) * mask).flatten(0, 1),
                (
                    (tr["rgb"].clamp(0, 1) if args.distill_only else b["rgb"][:, v:])
                    * mask
                ).flatten(0, 1),
                normalize=True,
            ).mean()
        depth = depth_loss(gaussian_depth(output), b["depth"][:, :v])
        depth_reference = depth_loss(
            gaussian_depth(teacher_obj), b["depth"][:, :v]
        ).detach()
        regression = pred["rgb"].new_zeros(())
        for mod, weight in [("rgb", 1.0), ("thermal", 4.0)]:
            current = (pred[mod] - b[mod][:, v:]).square().mean(2, keepdim=True)
            reference = (tr[mod] - b[mod][:, v:]).square().mean(2, keepdim=True)
            regression += (
                weight
                * 10
                * ((current - reference).clamp_min(0) * mask).sum()
                / mask.sum().clamp_min(1)
            )
        loss = (
            args.distill_weight * distill
            + args.image_weight * (image + 0.5 * depth)
            + args.teacher_image_weight * imitation
            + 2 * regression
            + 2 * (depth - depth_reference).clamp_min(0)
            + 0.001 * output.regularization
            + args.perceptual_weight * perceptual_loss
        )
        if args.distill_only:
            loss = (
                args.distill_weight * distill
                + args.teacher_image_weight * imitation
                + args.perceptual_weight * perceptual_loss
            )
        if args.quality_preservation_weight:
            # Ground-truth supervision only penalizes falling below the teacher.
            # There is no reward for exceeding its reconstruction quality.
            preservation = pred["rgb"].new_zeros(())
            for modality in ["rgb", "thermal"]:
                actual = pred[modality].clamp(0, 1)
                reference = tr[modality].clamp(0, 1)
                target = b[modality][:, v:]
                error = image_mse(actual, target, mask)
                reference_error = image_mse(reference, target, mask).detach()
                preservation += (error / reference_error.clamp_min(1e-5) - 1).clamp_min(
                    0
                )
                reference_ssim = ssim(reference * mask, target * mask).detach()
                preservation += 5 * (
                    reference_ssim - ssim(actual * mask, target * mask)
                ).clamp_min(0)
            predicted_lpips = perceptual(
                (pred["rgb"].clamp(0, 1) * mask).flatten(0, 1),
                (b["rgb"][:, v:] * mask).flatten(0, 1),
                normalize=True,
            ).mean()
            with torch.no_grad():
                reference_lpips = perceptual(
                    (tr["rgb"].clamp(0, 1) * mask).flatten(0, 1),
                    (b["rgb"][:, v:] * mask).flatten(0, 1),
                    normalize=True,
                ).mean()
            preservation += 5 * (predicted_lpips - reference_lpips).clamp_min(0)
            preservation += (depth - depth_reference).clamp_min(0)
            loss = loss + args.quality_preservation_weight * preservation
        if args.update_distill_weight and not rollout_final:
            weights = p["delta_weight"]
            update_loss = (
                (model.core.last_delta - p["target_delta"].float()).square() * weights
            ).sum(-1).mean() / weights.sum().clamp_min(1)
            loss = loss + args.update_distill_weight * update_loss
            del model.core.last_delta
        if phase is not None:
            loss = loss * args.phase_loss_weights[phase]
        if not torch.isfinite(loss):
            raise FloatingPointError(terms)
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        if not torch.isfinite(gradient) or gradient == 0:
            raise FloatingPointError(f"Invalid training gradient: {gradient}")
        optimizer.step()
        lr = args.lr * min(1.0, step / 50) * (0.2 + 0.8 * (1 - step / args.steps))
        for group in optimizer.param_groups:
            group["lr"] = lr * group.get("lr_scale", 1.0)
        row = {
            "step": step,
            "scene": scene,
            "views": v,
            "phase": phase,
            "rollout_final": rollout_final,
            "loss": float(loss.detach()),
            "distill": float(distill.detach()),
            "image": float(image.detach()),
            "imitation": float(imitation.detach()),
            "gradient": float(gradient),
            "seconds": time.time() - start,
            "terms": {k: float(val.detach()) for k, val in terms.items()},
        }
        with (out / "train.jsonl").open("a") as f:
            f.write(json.dumps(row) + "\n")
        if step == 1 or step % 25 == 0:
            print("TRAIN", json.dumps(row), flush=True)
        if args.smoke:
            print("SMOKE PASSED", flush=True)
            break
        if step % args.eval_every == 0 or step == args.steps:
            result = evaluate(model, val_paths, args.device, perceptual)
            (out / f"val_{step}.json").write_text(json.dumps(result, indent=2) + "\n")
            improved = result["score"] > best
            best = max(best, result["score"])
            save(out / "last.pt", model, optimizer, step, best, args)
            save(out / f"step_{step}.pt", model, optimizer, step, best, args)
            if improved:
                save(out / "best.pt", model, optimizer, step, best, args)
            print(
                "VAL",
                step,
                json.dumps(result["summary_delta"]),
                "worst",
                result["worst_scene_psnr_delta"],
                "passed",
                result["passed"],
                flush=True,
            )
            if result["passed"]:
                save(out / "passed.pt", model, optimizer, step, best, args)
                print(
                    "VALIDATION GATE PASSED; requires full evaluation before deployment",
                    flush=True,
                )


if __name__ == "__main__":
    main()
