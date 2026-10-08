"""Continuous spatial trust from observed RGB/T residuals, without target access."""

from contextlib import nullcontext
from dataclasses import replace

import torch
import torch.nn.functional as F

from .renderer import render_rgbt


def spatial_trust(
    initial,
    updated,
    rgb,
    thermal,
    valid=None,
    thermal_weight=0.25,
    kernel=21,
    mode="local",
    detach=True,
):
    """Two source renders estimate a local least-squares update coefficient.

    Rendering is nonlinear in geometry; this is a local approximation rather than
    a guarantee for novel views. Coefficients are detached, while interpolated
    Gaussian parameters retain gradients for training through the chosen update.
    """
    b, v, _, h, w = rgb.shape
    if valid is None:
        valid = torch.ones_like(rgb[:, :, :1])
    with (
        torch.no_grad() if detach else nullcontext(),
        torch.autocast(device_type=rgb.device.type, enabled=False),
    ):
        renders = [
            render_rgbt(o, initial.extrinsics, initial.intrinsics, h, w)
            for o in [initial, updated]
        ]
        numerator = rgb.new_zeros(b, v, 1, h, w, dtype=torch.float32)
        denominator = torch.zeros_like(numerator)
        if mode in ["balanced", "balanced_equal"]:
            rgb_error = (
                (
                    (renders[0]["rgb"].clamp(0, 1) - rgb).square().mean(2, keepdim=True)
                    * valid
                )
                .flatten(1)
                .sum(-1)
            )
            t_error = (
                ((renders[0]["thermal"].clamp(0, 1) - thermal).square() * valid)
                .flatten(1)
                .sum(-1)
            )
            normalization = (rgb_error / t_error.clamp_min(1e-5)).clamp(0.1, 100)[
                :, None, None, None, None
            ]
            thermal_weight = normalization * (
                1.0 if mode == "balanced_equal" else thermal_weight
            )
        for mod, target, weight in [
            ("rgb", rgb, 1.0),
            ("thermal", thermal, thermal_weight),
        ]:
            a, c = (r[mod].clamp(0, 1) for r in renders)
            direction = c - a
            numerator += (
                weight * (direction * (target - a)).mean(2, keepdim=True) * valid
            )
            denominator += weight * direction.square().mean(2, keepdim=True) * valid
        num = F.avg_pool2d(numerator.flatten(0, 1), kernel, 1, kernel // 2)
        den = F.avg_pool2d(denominator.flatten(0, 1), kernel, 1, kernel // 2)
        trust = (num / (den + 1e-7)).clamp(0, 1).reshape(b, v, h, w)
        if mode in ["global", "hybrid"]:
            global_trust = (
                numerator.sum((1, 2, 3, 4)) / (denominator.sum((1, 2, 3, 4)) + 1e-7)
            ).clamp(0, 1)[:, None, None, None]
            if mode == "global":
                trust = global_trust.expand_as(trust)
            else:
                trust = 0.75 * global_trust + 0.25 * trust
        trust = F.avg_pool2d(trust.reshape(b * v, 1, h, w), 5, 1, 2).reshape(b, -1)
    a, c = initial.gaussians, updated.gaussians
    gs = replace(
        a,
        means=a.means.lerp(c.means, trust[..., None]),
        scales=a.scales.lerp(c.scales, trust[..., None]),
        rotations=F.normalize(a.rotations.lerp(c.rotations, trust[..., None]), dim=-1),
        harmonics=a.harmonics.lerp(c.harmonics, trust[..., None, None]),
        opacities=a.opacities.lerp(c.opacities, trust),
    )
    return replace(
        updated,
        gaussians=gs,
        thermal_colors=initial.thermal_colors.lerp(
            updated.thermal_colors, trust[..., None]
        ),
        refinement_strength=trust.mean(1),
    )
