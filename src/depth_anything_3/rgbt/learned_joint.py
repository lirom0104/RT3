"""A single trained encoder for local, cross-view, geometry and appearance updates.

The frozen preparation uses only the initial reconstruction and source images.
The teacher is never called at inference. Prepared data is cacheable for training.
"""

from dataclasses import replace

import torch
import torch.nn.functional as F
from torch import nn

from depth_anything_3.utils.geometry import affine_inverse

from .calibration import calibrate_output
from .attention import ProjectedNeighborhood, WindowNeighborhood
from .geometry import gaussian_depth
from .source_geometry import SourceGeometry
from .local_trust import spatial_trust
from .gaussian_update import GaussianUpdate
from .renderer import render_rgbt
from .source_anchors import source_anchor_field
from .surface_consensus import (
    anchor_consensus,
    apply_consensus,
    build_consensus,
    plane_covariance,
)
from .surface_reliability import (
    guard_surface,
    projection_association,
    select_geometry,
    source_color_residual,
    visible_appearance,
)


class SharedUpdate(GaussianUpdate):
    """One state encoder and output head with local and window attention."""

    def __init__(
        self,
        channels=96,
        attention_pairs=1,
        iterations=1,
        thermal_edge=False,
        thermal_edge_channels=16,
        thermal_edge_geometry_limit=0.25,
    ):
        super().__init__(
            channels=channels,
            blocks=0,
            iterations=iterations,
            render_feedback=True,
            depth_limit=0.10,
            color_limit=0.30,
            scale_limit=1.6,
            opacity_limit=4.0,
            rgb_limit=0.04,
            support_context=True,
            position_strength=0.2,
            source_guard=False,
            fused_rendering=True,
            compact=True,
            thermal_edge=thermal_edge,
            thermal_edge_channels=thermal_edge_channels,
            thermal_edge_geometry_limit=thermal_edge_geometry_limit,
        )
        self.blocks = nn.ModuleList(
            [
                block(channels)
                for _ in range(attention_pairs)
                for block in (ProjectedNeighborhood, WindowNeighborhood)
            ]
        )
        self.aligned_points = True
        self._prepared_feedback = None
        self._feedback_gaussians = None

    def source_feedback(self, output, E, K, h, w):
        if (
            self._prepared_feedback is not None
            and output.gaussians is self._feedback_gaussians
        ):
            return self._prepared_feedback
        return render_rgbt(output, E, K, h, w)

    def forward(self, *args, feedback=None, extra_geometry=None, **kwargs):
        # Only used synchronously inside this module's forward, never across calls.
        self._prepared_feedback = feedback
        self._feedback_gaussians = kwargs["initial"].gaussians
        self._extra_geometry = extra_geometry
        try:
            return super().forward(*args, share_projection=True, **kwargs)
        finally:
            self._prepared_feedback = None
            self._feedback_gaussians = None
            self._extra_geometry = None


class LearnedJointRefiner(nn.Module):
    """One shared encoder and head, reused for bounded RGB-T state updates."""

    def __init__(
        self,
        channels=96,
        attention_pairs=1,
        trust=True,
        color_feedback=True,
        flexible_geometry=False,
        iterations=1,
        global_context=False,
        geometry_context=False,
        ordered_updates=False,
        phase_modulation=False,
        preserve_first_update=False,
        shared_residual=False,
        thermal_edge=False,
        thermal_edge_channels=16,
        thermal_edge_geometry_limit=0.25,
    ):
        super().__init__()
        self.config = {
            "channels": channels,
            "attention_pairs": attention_pairs,
            "trust": trust,
            "color_feedback": color_feedback,
            "flexible_geometry": flexible_geometry,
            "iterations": iterations,
            "global_context": global_context,
            "geometry_context": geometry_context,
            "ordered_updates": ordered_updates,
            "phase_modulation": phase_modulation,
            "preserve_first_update": preserve_first_update,
            "shared_residual": shared_residual,
            "thermal_edge": thermal_edge,
            "thermal_edge_channels": thermal_edge_channels,
            "thermal_edge_geometry_limit": thermal_edge_geometry_limit,
        }
        self.core = SharedUpdate(
            channels,
            attention_pairs,
            iterations,
            thermal_edge=thermal_edge,
            thermal_edge_channels=thermal_edge_channels,
            thermal_edge_geometry_limit=thermal_edge_geometry_limit,
        )
        if ordered_updates:
            if iterations != 1:
                raise ValueError("Ordered refinement owns the shared update loop")
            self.core.phase_embedding = nn.Embedding(3, channels)
            nn.init.zeros_(self.core.phase_embedding.weight)
            self.core.ordered_projection = nn.Conv2d(16, channels, 1)
            nn.init.zeros_(self.core.ordered_projection.weight)
            nn.init.zeros_(self.core.ordered_projection.bias)
            if phase_modulation:
                self.core.phase_modulation = nn.Embedding(
                    3, (len(self.core.blocks) + 2) * channels * 2
                )
                nn.init.zeros_(self.core.phase_modulation.weight)
        elif phase_modulation:
            raise ValueError("Phase modulation requires ordered updates")
        if preserve_first_update:
            if not ordered_updates or attention_pairs != 2 or channels != 96:
                raise ValueError(
                    "Preserving the initial local update requires ordered 96-channel, two-pair encoding"
                )
            self.core.preserve_first_update = True
        if shared_residual:
            if not ordered_updates:
                raise ValueError("Shared residual requires ordered updates")
            self.core.shared_residual = nn.Sequential(
                nn.Conv2d(channels, 64, 3, padding=1),
                nn.GELU(),
                nn.Conv2d(64, channels, 1),
            )
            nn.init.zeros_(self.core.shared_residual[-1].weight)
            nn.init.zeros_(self.core.shared_residual[-1].bias)
        if self.core.state_conv is not None:
            nn.init.zeros_(self.core.state_conv.weight)
            nn.init.zeros_(self.core.state_conv.bias)
        if global_context:
            self.core.global_context = nn.Sequential(
                nn.Conv2d(channels, channels, 1),
                nn.GELU(),
                nn.Conv2d(channels, channels, 1),
            )
            nn.init.zeros_(self.core.global_context[-1].weight)
            nn.init.zeros_(self.core.global_context[-1].bias)
        if geometry_context:
            self.core.geometry_context = nn.Conv2d(8, channels, 1)
            nn.init.zeros_(self.core.geometry_context.weight)
            nn.init.zeros_(self.core.geometry_context.bias)
        if flexible_geometry:
            self.core.depth_limit = 0.30
            self.core.min_update_pixel_scale = 0.001
            self.core.max_update_pixel_scale = 30.0
            self.core.rotation_update_limit = 1.5
        self.trust = trust
        self.color_feedback = color_feedback
        self.last_association = None

    @torch.no_grad()
    def prepare(self, rgb, thermal, initial, source_valid=None, known_cameras=True):
        """Cache frozen, source-only geometry and one render of the prepared state."""
        with torch.autocast(device_type=rgb.device.type, enabled=False):
            if self.config["ordered_updates"]:
                from .ordered_joint import prepare_initial

                return prepare_initial(rgb, initial)
            return self._prepare(rgb, thermal, initial, source_valid, known_cameras)

    def _prepare(
        self,
        rgb,
        thermal,
        initial,
        source_valid,
        known_cameras,
        association=None,
        observations=None,
    ):
        raw = initial
        if observations is None:
            observations = SourceGeometry(rgb).observations(initial)
        if known_cameras:
            calibrated, _ = calibrate_output(initial, rgb, observations=observations)
            initial = replace(
                initial,
                gaussians=replace(
                    initial.gaussians,
                    means=initial.gaussians.means.lerp(calibrated.gaussians.means, 0.1),
                    scales=initial.gaussians.scales.lerp(
                        calibrated.gaussians.scales, 0.1
                    ),
                ),
                depth=initial.depth.lerp(calibrated.depth, 0.1),
            )
        initial = replace(initial, fused_rendering=True)
        if association is None:
            association = build_consensus(initial, rgb, source_valid)
            association = anchor_consensus(
                initial, rgb, association, observations=observations
            )
        association = projection_association(initial, association, source_valid)
        surface = plane_covariance(apply_consensus(initial, association), association)
        association, audit = guard_surface(
            initial, surface, association, rgb, thermal, source_valid, 0.15
        )
        surface = select_geometry(initial, surface, association["ids"])
        surface = visible_appearance(
            surface,
            rgb,
            thermal,
            association,
            source_valid,
            thermal_blend=1.0,
            rgb_blend=0.5,
            thermal_sigma=1.0,
        )
        correction, _ = source_anchor_field(
            initial, rgb, source_valid, radius=64.0, observations=observations
        )
        correction = correction * (association["ids"] < 0)
        b, _v, h, w = initial.depth.shape
        centers = (
            affine_inverse(initial.extrinsics)[..., :3, 3][:, :, None, None]
            .expand(-1, -1, h, w, -1)
            .reshape(b, -1, 3)
        )
        ratio = correction.exp().reshape(b, -1, 1)
        surface = replace(
            surface,
            gaussians=replace(
                surface.gaussians,
                means=centers + (surface.gaussians.means - centers) * ratio,
                scales=surface.gaussians.scales * ratio,
            ),
        )
        feedback = render_rgbt(surface, surface.extrinsics, surface.intrinsics, h, w)
        prepared = {
            "base": surface,
            "association": association,
            "feedback": feedback,
            "guard_audit": audit,
        }
        prepared["geometry_context"] = self.geometry_features(raw, prepared)
        return prepared

    @staticmethod
    @torch.no_grad()
    def geometry_features(raw, prepared):
        base, association = prepared["base"], prepared["association"]
        b, v, h, w = base.depth.shape
        depth = gaussian_depth(raw).clamp_min(1e-4)
        shift = (base.gaussians.means - raw.gaussians.means).reshape(
            b, v, h, w, 3
        ) / depth[..., None]
        ids = association["ids"]
        normal = torch.zeros_like(shift)
        if len(association["normals"]):
            normal = association["normals"][ids.clamp_min(0)] * (ids >= 0)[..., None]
        ratio = (gaussian_depth(base).clamp_min(1e-4) / depth).log()[..., None]
        features = torch.cat([shift, normal, (ids >= 0).float()[..., None], ratio], -1)
        features = features.permute(0, 1, 4, 2, 3).flatten(0, 1)
        return F.interpolate(features, (h // 4, w // 4), mode="area")

    def forward(
        self,
        rgb,
        thermal,
        extrinsics=None,
        intrinsics=None,
        *,
        initial,
        source_valid=None,
        prepared=None,
        return_proposal=False,
    ):
        if self.config["ordered_updates"]:
            from .ordered_joint import ordered_forward

            return ordered_forward(
                self,
                rgb,
                thermal,
                extrinsics,
                intrinsics,
                initial,
                source_valid,
                prepared,
                return_proposal,
            )
        if prepared is None:
            prepared = self.prepare(
                rgb, thermal, initial, source_valid, extrinsics is not None
            )
        base = prepared["base"]
        self.last_association = prepared["association"]
        proposal = self.core(
            rgb,
            thermal,
            extrinsics,
            intrinsics,
            initial=base,
            source_valid=source_valid,
            feedback=prepared["feedback"],
            extra_geometry=prepared.get("geometry_context"),
        )
        updated = proposal
        if self.trust:
            updated = spatial_trust(
                base, updated, rgb, thermal, source_valid, mode="balanced", detach=True
            )
        if self.color_feedback:
            with torch.autocast(device_type=rgb.device.type, enabled=False):
                correction = source_color_residual(
                    updated, rgb, thermal, source_valid, thermal_consensus=True
                )
            sh = updated.gaussians.harmonics.clone()
            sh[..., 0] += correction[..., :3] / 0.28209479177387814
            updated = replace(
                updated,
                gaussians=replace(updated.gaussians, harmonics=sh),
                thermal_colors=(updated.thermal_colors + correction[..., 3:]).clamp(
                    0, 1
                ),
            )
        return (updated, proposal) if return_proposal else updated

    def initialize_from_teacher(self, teacher):
        """Warm-start compatible local encoding and one block of each attention type."""
        if self.config["channels"] != teacher.neighborhood_refiner.config["channels"]:
            return
        self.core.neck.load_state_dict(teacher.neighborhood_refiner.neck.state_dict())
        self.core.head.load_state_dict(teacher.neighborhood_refiner.head.state_dict())
        for index, block in enumerate(self.core.blocks):
            source = (
                teacher.neighborhood_refiner
                if index % 2 == 0
                else teacher.crossview_refiner
            )
            block.load_state_dict(
                source.blocks[min(index // 2, len(source.blocks) - 1)].state_dict()
            )
