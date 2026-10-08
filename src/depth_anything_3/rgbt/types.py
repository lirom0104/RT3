"""Source-aligned Gaussian reconstruction output."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from depth_anything_3.specs import Gaussians


@dataclass
class RGBTOutput:
    gaussians: Gaussians
    thermal_colors: torch.Tensor  # B,G,1, scalar display intensity, not temperature
    depth: torch.Tensor
    extrinsics: torch.Tensor
    intrinsics: torch.Tensor
    depth_conf: torch.Tensor
    regularization: torch.Tensor
    features: torch.Tensor | None = (
        None  # Optional source-view features for refinement.
    )
    refinement_strength: torch.Tensor | None = None
    fused_rendering: bool = False
    thermal_harmonics: torch.Tensor | None = (
        None  # Optional scalar SH coefficients B,G,9.
    )
