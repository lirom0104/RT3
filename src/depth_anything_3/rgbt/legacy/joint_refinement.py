"""Historical three-stage refiner with shared source geometry."""

import torch
from torch import nn

from ..source_geometry import SourceGeometry
from .crossview import CrossViewRefiner
from .refinement import NeighborhoodRefiner
from .surface import SurfaceRefiner


class JointRefiner(nn.Module):
    """Unified local, cross-view and surface refinement with shared source geometry."""

    def __init__(self, neighborhood, crossview, surface):
        super().__init__()
        self.local = NeighborhoodRefiner(**neighborhood)
        self.crossview = CrossViewRefiner(**crossview)
        self.surface = SurfaceRefiner(**surface)
        for module in self.frozen_stages:
            module.requires_grad_(False).eval()

    @property
    def frozen_stages(self):
        return (self.local, self.crossview)

    def train(self, mode=True):
        super().train(mode)
        for module in self.frozen_stages:
            module.eval()
        return self

    @torch.no_grad()
    def compute_prior(
        self,
        rgb,
        thermal,
        extrinsics,
        intrinsics,
        initial,
        source_valid=None,
        source_geometry=None,
    ):
        if source_geometry is None:
            source_geometry = SourceGeometry(rgb)
        output = self.local(
            rgb,
            thermal,
            extrinsics,
            intrinsics,
            initial=initial,
            source_valid=source_valid,
            share_projection=True,
        )
        return self.crossview(
            rgb,
            thermal,
            extrinsics,
            intrinsics,
            initial=output,
            source_valid=source_valid,
            source_geometry=source_geometry,
            share_projection=True,
        )

    def forward(
        self,
        rgb,
        thermal,
        extrinsics=None,
        intrinsics=None,
        *,
        initial,
        source_valid=None,
        association=None,
        prior_ready=False,
    ):
        source_geometry = SourceGeometry(rgb)
        if not prior_ready:
            initial = self.compute_prior(
                rgb,
                thermal,
                extrinsics,
                intrinsics,
                initial,
                source_valid,
                source_geometry,
            )
        return self.surface(
            rgb,
            thermal,
            extrinsics,
            intrinsics,
            initial=initial,
            source_valid=source_valid,
            association=association,
            source_geometry=source_geometry,
        )
