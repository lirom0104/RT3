"""Current RGB-T Gaussian network.

Historical independently trained refiners live under :mod:`rgbt.legacy` and
are selected by the checkpoint loader when an old checkpoint is opened.
"""

from copy import deepcopy

import torch
from torch import nn

from .learned_joint import LearnedJointRefiner
from .reconstruction import GaussianReconstruction
from .renderer import render_rgbt
from .types import RGBTOutput

__all__ = ["RGBTGaussianNetwork", "RGBTOutput", "render_rgbt"]


class RGBTGaussianNetwork(nn.Module):
    """Frozen DA3 reconstruction followed by one shared RGB-T refiner."""

    def __init__(
        self,
        foundation,
        reconstruction,
        learned_joint,
        trainable_modules=None,
        architecture="rgbt-gaussian-network",
        joint_refinement=True,
        geometry_correction=None,
    ):
        super().__init__()
        if architecture != "rgbt-gaussian-network":
            raise ValueError(f"Unsupported architecture: {architecture}")
        if not joint_refinement:
            raise ValueError("Current checkpoints require joint_refinement=True")
        if trainable_modules is None:
            trainable_modules = ["refiner"]
        if trainable_modules != ["refiner"]:
            raise ValueError(
                "Current architecture requires trainable_modules=['refiner']"
            )

        self.config = deepcopy(
            {
                "architecture": architecture,
                "foundation": foundation,
                "reconstruction": reconstruction,
                "trainable_modules": trainable_modules,
                "joint_refinement": True,
                "learned_joint": learned_joint,
            }
        )
        self.joint_refinement = True
        self.learned_joint = True
        self.geometry_corrector = None
        if geometry_correction is not None:
            from .multiview_depth import MultiViewDepthCorrector

            self.geometry_corrector = MultiViewDepthCorrector(**geometry_correction)
            self.config["geometry_correction"] = deepcopy(geometry_correction)
        self.reconstruction = GaussianReconstruction(foundation, **reconstruction)
        if learned_joint.get("mode") == "native_residual":
            from .native_refiner import NativeGaussianRefiner

            if reconstruction.get("geometry_mode") != "native":
                raise ValueError(
                    "Native residual refinement requires native Gaussian initialization"
                )
            self.refiner = NativeGaussianRefiner(**learned_joint)
        else:
            self.refiner = LearnedJointRefiner(**learned_joint)
        for module in self.frozen_stages:
            module.requires_grad_(False).eval()

    @property
    def frozen_stages(self):
        return (self.reconstruction,)

    @property
    def last_association(self):
        return self.refiner.last_association

    @property
    def last_guard_audit(self):
        return getattr(self.refiner, "last_guard_audit", None)

    def train(self, mode=True):
        super().train(mode)
        for module in self.frozen_stages:
            module.eval()
        return self

    @torch.no_grad()
    def compute_prior(
        self, rgb, thermal, extrinsics=None, intrinsics=None, source_valid=None
    ):
        """Return the cacheable frozen reconstruction before shared refinement."""
        return self.reconstruction(rgb, thermal, extrinsics, intrinsics)

    def forward(
        self,
        rgb,
        thermal,
        extrinsics=None,
        intrinsics=None,
        source_valid=None,
        initial=None,
        association=None,
        prepared=None,
    ):
        """Reconstruct shared RGB/T Gaussians from source views only."""
        if rgb.ndim != 5 or rgb.shape[0] != 1 or rgb.shape[2] != 3:
            raise ValueError("RGB input must have shape [1, views, 3, height, width]")
        batch, views, _, height, width = rgb.shape
        if (
            thermal.shape != (batch, views, 1, height, width)
            or height % 14
            or width % 14
        ):
            raise ValueError(
                "Paired RGB/T grids must match; height and width must be multiples of 14"
            )
        if (extrinsics is None) != (intrinsics is None):
            raise ValueError("Supply source intrinsics and extrinsics together")
        if source_valid is not None and source_valid.shape != (
            batch,
            views,
            1,
            height,
            width,
        ):
            raise ValueError("Source validity must match the paired image grid")
        if association is not None:
            raise ValueError(
                "Learned joint refinement needs raw reconstruction, not legacy priors"
            )
        if initial is None:
            with torch.no_grad():
                initial = self.reconstruction(rgb, thermal, extrinsics, intrinsics)
        corrector = getattr(self, "geometry_corrector", None)
        if corrector is not None and prepared is not None:
            raise ValueError("Legacy prepared geometry is invalid after depth correction")
        correct_before = (
            corrector is not None
            and corrector.config["apply_before"]
        )
        if correct_before:
            initial = corrector(rgb, initial, source_valid)
        output = self.refiner(
            rgb,
            thermal,
            extrinsics,
            intrinsics,
            initial=initial,
            source_valid=source_valid,
            prepared=prepared,
        )
        if corrector is not None and not correct_before:
            output = corrector(rgb, output, source_valid)
        return output
