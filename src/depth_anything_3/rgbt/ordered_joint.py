"""Three updates by one shared network, with source constraints between updates.

This module contains no learned parameters or teacher calls. Phase conditioning,
encoding, attention and the update head all belong to the same ``core`` instance.
"""

from dataclasses import replace

import torch
import torch.nn.functional as F

from .calibration import calibrate_output
from .geometry import gaussian_depth
from .source_geometry import SourceGeometry
from .local_trust import spatial_trust
from .renderer import leave_one_out, render_rgbt
from .surface_consensus import anchor_consensus, build_consensus
from .surface_reliability import geometry_feedback, source_color_residual


@torch.no_grad()
def prepare_initial(rgb, initial):
    initial = replace(initial, fused_rendering=True)
    h, w = rgb.shape[-2:]
    return {
        "base": initial,
        "feedback": render_rgbt(initial, initial.extrinsics, initial.intrinsics, h, w),
        "phase": 0,
        "association": {},
    }


def prepare_phase(
    model, phase, rgb, thermal, initial, valid, known_cameras=True, observations=None
):
    """Compute source-only constraints from the current shared-network state."""
    autocast_enabled = torch.is_autocast_enabled(rgb.device.type)
    autocast_dtype = torch.get_autocast_dtype(rgb.device.type)
    with torch.no_grad(), torch.autocast(rgb.device.type, enabled=False):
        if phase == 0:
            return prepare_initial(rgb, initial)
        if phase == 1:
            base = initial
            if known_cameras:
                with torch.autocast(
                    rgb.device.type, enabled=autocast_enabled, dtype=autocast_dtype
                ):
                    calibrated, _ = calibrate_output(
                        initial, rgb, observations=observations
                    )
                base = replace(
                    initial,
                    gaussians=replace(
                        initial.gaussians,
                        means=initial.gaussians.means.lerp(
                            calibrated.gaussians.means, 0.1
                        ),
                        scales=initial.gaussians.scales.lerp(
                            calibrated.gaussians.scales, 0.1
                        ),
                    ),
                    depth=initial.depth.lerp(calibrated.depth, 0.1),
                )
            feedback = leave_one_out(
                base, base.extrinsics, base.intrinsics, *rgb.shape[-2:]
            )
            return {
                "base": base,
                "reference": initial,
                "feedback": feedback,
                "phase": phase,
                "association": {},
            }
        if phase != 2:
            raise ValueError(phase)
        # Keep association arithmetic in the same precision as the reference
        # inference protocol; changing it can change discrete plane membership.
        with torch.autocast(
            rgb.device.type, enabled=autocast_enabled, dtype=autocast_dtype
        ):
            association = anchor_consensus(
                initial,
                rgb,
                build_consensus(initial, rgb, valid),
                observations=observations,
            )
        p = model._prepare(
            rgb,
            thermal,
            initial,
            valid,
            False,
            association=association,
            observations=observations,
        )
        feedback = geometry_feedback(initial, rgb, thermal, valid)
        p["ordered_context"] = torch.cat([p["geometry_context"], feedback], 1)
        support = F.interpolate(feedback[:, 7:8], rgb.shape[-2:], mode="nearest")
        p["depth_mask"] = support.reshape_as(p["association"]["ids"]) * (
            p["association"]["ids"] < 0
        )
        if valid is not None:
            p["depth_mask"] *= valid[:, :, 0] > 0.5
        p["phase"] = phase
        return p


def apply_phase(model, rgb, thermal, E, K, valid, prepared, return_proposal=False):
    phase = prepared["phase"]
    base = replace(prepared["base"], fused_rendering=True)
    core = model.core
    # These are update bounds and analytic rules, not separate parameter sets.
    settings = [
        {
            "depth_limit": 0.0,
            "position_strength": 0.0,
            "scale_limit": 0.8,
            "opacity_limit": 2.0,
            "color_limit": 0.15,
            "rgb_limit": 0.0,
            "min_update_pixel_scale": 0.1,
            "max_update_pixel_scale": 6.0,
            "rotation_update_limit": 0.0,
            "aligned_points": False,
        },
        {
            "depth_limit": 0.05,
            "position_strength": 0.2,
            "scale_limit": 0.8,
            "opacity_limit": 2.0,
            "color_limit": 0.15,
            "rgb_limit": 0.0,
            "min_update_pixel_scale": 0.1,
            "max_update_pixel_scale": 6.0,
            "rotation_update_limit": 0.02,
            "aligned_points": True,
        },
        {
            "depth_limit": 0.02,
            "position_strength": 0.0,
            "scale_limit": 0.325,
            "opacity_limit": 0.75,
            "color_limit": 0.06,
            "rgb_limit": 0.01,
            "min_update_pixel_scale": 0.0,
            "max_update_pixel_scale": float("inf"),
            "rotation_update_limit": 0.0,
            "aligned_points": True,
        },
    ][phase]
    settings["bound_total_depth"] = phase < 2
    missing = object()
    previous = {k: getattr(core, k, missing) for k in settings}
    for key, value in settings.items():
        setattr(core, key, value)
    core._phase_index = phase
    core._ordered_context = prepared.get("ordered_context")
    core._depth_mask = prepared.get("depth_mask")
    core._scale_with_depth = phase == 2
    feedback = prepared["feedback"]
    if phase == 0 and getattr(core, "preserve_first_update", False):
        # The original first step renders feedback under the caller's autocast
        # context. A cached FP32 render can change subsequent BF16 rounding.
        with torch.no_grad():
            feedback = render_rgbt(
                base, base.extrinsics, base.intrinsics, *rgb.shape[-2:]
            )
    try:
        proposal = core(
            rgb,
            thermal,
            E,
            K,
            initial=base,
            source_valid=valid,
            feedback=feedback,
        )
    finally:
        core._phase_index = core._ordered_context = core._depth_mask = None
        core._scale_with_depth = False
        for key, value in previous.items():
            if value is missing:
                delattr(core, key)
            else:
                setattr(core, key, value)
    if phase == 0:
        output = core.guard_update(base, proposal, rgb, valid, thermal)
    elif phase == 1:
        reference = replace(prepared["reference"], fused_rendering=True)
        gs = replace(
            reference.gaussians,
            **{
                name: getattr(reference.gaussians, name).lerp(
                    getattr(proposal.gaussians, name), 0.5
                )
                for name in ["means", "scales", "rotations", "harmonics", "opacities"]
            },
        )
        gs = replace(gs, rotations=F.normalize(gs.rotations, dim=-1))
        blended = replace(
            proposal,
            gaussians=gs,
            thermal_colors=reference.thermal_colors.lerp(proposal.thermal_colors, 0.5),
        )
        output = spatial_trust(
            reference, blended, rgb, thermal, valid, mode="local", detach=True
        )
    else:
        model.last_association = prepared["association"]
        # The original appearance trust rule compares two states with the same
        # learned depth. It must not gate the earlier geometric correction again.
        with torch.autocast(rgb.device.type, enabled=False):
            ratio = (
                gaussian_depth(proposal) / gaussian_depth(base).clamp_min(1e-6)
            ).reshape(1, -1, 1)
            reference = replace(
                base,
                gaussians=replace(
                    base.gaussians,
                    means=proposal.gaussians.means,
                    scales=base.gaussians.scales * ratio,
                ),
            )
        output = spatial_trust(
            reference, proposal, rgb, thermal, valid, mode="balanced", detach=True
        )
        with torch.autocast(rgb.device.type, enabled=False):
            correction = source_color_residual(
                output, rgb, thermal, valid, thermal_consensus=True
            )
        sh = output.gaussians.harmonics.clone()
        sh[..., 0] += correction[..., :3] / 0.28209479177387814
        output = replace(
            output,
            gaussians=replace(output.gaussians, harmonics=sh),
            thermal_colors=(output.thermal_colors + correction[..., 3:]).clamp(0, 1),
        )
    return (output, proposal) if return_proposal else output


def ordered_forward(
    model, rgb, thermal, E, K, initial, valid, prepared, return_proposal
):
    if prepared is not None and prepared.get("phase_training", False):
        return apply_phase(model, rgb, thermal, E, K, valid, prepared, return_proposal)
    output = initial
    observations = None if prepared is None else prepared.get("source_observations")
    if observations is None:
        observations = SourceGeometry(rgb).observations(initial)
    for phase in range(3):
        p = (
            prepared
            if phase == 0 and prepared is not None
            else prepare_phase(
                model,
                phase,
                rgb,
                thermal,
                output,
                valid,
                E is not None,
                observations=observations,
            )
        )
        output, proposal = apply_phase(model, rgb, thermal, E, K, valid, p, True)
    return (output, proposal) if return_proposal else output
