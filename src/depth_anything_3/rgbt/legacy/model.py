"""Load the sequential or fused three-stage historical RGB-T architecture."""

from copy import deepcopy

import torch
from torch import nn

from .crossview import CrossViewRefiner
from .joint_refinement import JointRefiner
from ..reconstruction import GaussianReconstruction
from .refinement import NeighborhoodRefiner
from ..renderer import render_rgbt
from .surface import SurfaceRefiner
from ..types import RGBTOutput

__all__ = ["LegacyRGBTGaussianNetwork", "RGBTOutput", "render_rgbt"]


class LegacyRGBTGaussianNetwork(nn.Module):
    """Historical reconstruction and three independently parameterized stages."""

    def __init__(
        self,
        foundation,
        reconstruction,
        neighborhood=None,
        crossview=None,
        surface=None,
        trainable_modules=None,
        architecture="rgbt-gaussian-network",
        joint_refinement=False,
    ):
        super().__init__()
        if architecture != "rgbt-gaussian-network":
            raise ValueError(f"Unsupported architecture: {architecture}")
        self.config = deepcopy(
            {
                "architecture": architecture,
                "foundation": foundation,
                "reconstruction": reconstruction,
                "neighborhood": neighborhood,
                "crossview": crossview,
                "surface": surface,
                "trainable_modules": ["surface_refiner"]
                if trainable_modules is None
                else trainable_modules,
            }
        )
        self.joint_refinement = bool(joint_refinement)
        self.learned_joint = False
        if self.joint_refinement:
            self.config["joint_refinement"] = True
        expected_trainable = ["surface_refiner"]
        if self.config["trainable_modules"] != expected_trainable:
            raise ValueError(
                f"This architecture requires trainable_modules={expected_trainable}"
            )
        self.reconstruction = GaussianReconstruction(foundation, **reconstruction)
        if self.joint_refinement:
            self.refiner = JointRefiner(neighborhood, crossview, surface)
        else:
            self.neighborhood_refiner = NeighborhoodRefiner(**neighborhood)
            self.crossview_refiner = CrossViewRefiner(**crossview)
            self.surface_refiner = SurfaceRefiner(**surface)
        for module in self.frozen_stages:
            module.requires_grad_(False).eval()

    @property
    def neighborhood_refiner(self):
        return (
            self.refiner.local
            if self.joint_refinement
            else self._modules["neighborhood_refiner"]
        )

    @property
    def crossview_refiner(self):
        return (
            self.refiner.crossview
            if self.joint_refinement
            else self._modules["crossview_refiner"]
        )

    @property
    def surface_refiner(self):
        return (
            self.refiner.surface
            if self.joint_refinement
            else self._modules["surface_refiner"]
        )

    @property
    def frozen_stages(self):
        return (self.reconstruction, self.neighborhood_refiner, self.crossview_refiner)

    @property
    def last_association(self):
        return self.surface_refiner.last_association

    @property
    def last_guard_audit(self):
        return getattr(self.surface_refiner, "last_guard_audit", None)

    def train(self, mode=True):
        super().train(mode)
        for module in self.frozen_stages:
            module.eval()
        return self

    @torch.no_grad()
    def compute_prior(
        self, rgb, thermal, extrinsics=None, intrinsics=None, source_valid=None
    ):
        """Cacheable frozen reconstruction, using source observations only."""
        initial = self.reconstruction(rgb, thermal, extrinsics, intrinsics)
        if self.joint_refinement:
            return self.refiner.compute_prior(
                rgb, thermal, extrinsics, intrinsics, initial, source_valid
            )
        initial = self.neighborhood_refiner(
            rgb,
            thermal,
            extrinsics,
            intrinsics,
            initial=initial,
            source_valid=source_valid,
        )
        return self.crossview_refiner(
            rgb,
            thermal,
            extrinsics,
            intrinsics,
            initial=initial,
            source_valid=source_valid,
        )

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
        """Reconstruct shared RGB/T Gaussians; target views are renderer inputs only."""
        if rgb.ndim != 5 or rgb.shape[0] != 1 or rgb.shape[2] != 3:
            raise ValueError("RGB input must have shape [1, views, 3, height, width]")
        b, views, _, h, w = rgb.shape
        if thermal.shape != (b, views, 1, h, w) or h % 14 or w % 14:
            raise ValueError(
                "Paired RGB/T grids must match; height and width must be multiples of 14"
            )
        if (extrinsics is None) != (intrinsics is None):
            raise ValueError("Supply source intrinsics and extrinsics together")
        if source_valid is not None and source_valid.shape != (b, views, 1, h, w):
            raise ValueError("Source validity must match the paired image grid")
        if self.joint_refinement:
            prior_ready = initial is not None
            if initial is None:
                with torch.no_grad():
                    initial = self.reconstruction(rgb, thermal, extrinsics, intrinsics)
            return self.refiner(
                rgb,
                thermal,
                extrinsics,
                intrinsics,
                initial=initial,
                source_valid=source_valid,
                association=association,
                prior_ready=prior_ready,
            )
        if initial is None:
            initial = self.compute_prior(
                rgb, thermal, extrinsics, intrinsics, source_valid
            )
        return self.surface_refiner(
            rgb,
            thermal,
            extrinsics,
            intrinsics,
            source_valid=source_valid,
            initial=initial,
            association=association,
        )
