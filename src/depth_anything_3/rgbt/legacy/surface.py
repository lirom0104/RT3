"""Reliable shared surfaces, bounded depth correction and appearance refinement."""

from dataclasses import replace

import torch
import torch.nn.functional as F
from torch import nn

from depth_anything_3.utils.geometry import affine_inverse

from ..geometry import gaussian_depth
from ..surface_consensus import (
    anchor_consensus,
    apply_consensus,
    build_consensus,
    consensus_appearance,
    plane_covariance,
)
from ..surface_reliability import (
    geometry_feedback,
    guard_surface,
    projection_association,
    select_geometry,
    source_color_residual,
    visible_appearance,
)


class SurfaceRefiner(nn.Module):
    def __init__(
        self,
        channels=48,
        learn_planes=True,
        flatten=0.0,
        rgb_limit=0.04,
        anchored=False,
        appearance_strength=1.0,
        appearance_trust=False,
        surface_only=False,
        plane_cov=False,
        thermal_fusion=0.0,
        rgb_fusion=0.0,
        verified_only=False,
        large_plane_fallback=False,
        geometry_guard=True,
        guard_margin=0.15,
        visibility_fusion=True,
        thermal_sigma=0.10,
        dense_depth_limit=0.0,
        plane_normal_limit=0.015,
        plane_offset_limit=0.01,
        appearance_trust_mode="balanced",
        appearance_feedback=False,
        rgb_feedback_step=0.0,
        thermal_feedback_step=0.0,
        differentiate_feedback=False,
        thermal_feedback_blur=0,
        thermal_feedback_consensus=False,
        anchor_strength=0.0,
        anchor_radius=32.0,
        compact=False,
    ):
        super().__init__()
        self.trust_mode = "surface"
        self.learn_planes = learn_planes
        self.flatten = flatten
        self.rgb_limit = rgb_limit
        self.anchored = anchored
        self.appearance_strength = appearance_strength
        self.appearance_trust = appearance_trust
        self.surface_only = surface_only
        self.plane_cov = plane_cov
        self.thermal_fusion = thermal_fusion
        self.rgb_fusion = rgb_fusion
        self.verified_only = verified_only
        self.large_plane_fallback = large_plane_fallback
        self.geometry_guard = geometry_guard
        self.guard_margin = guard_margin
        self.visibility_fusion = visibility_fusion
        self.thermal_sigma = thermal_sigma
        self.dense_depth_limit = dense_depth_limit
        self.plane_normal_limit = plane_normal_limit
        self.plane_offset_limit = plane_offset_limit
        self.appearance_trust_mode = appearance_trust_mode
        self.feedback_projection = nn.Conv2d(8, channels, 1)
        nn.init.zeros_(self.feedback_projection.weight)
        nn.init.zeros_(self.feedback_projection.bias)
        self.depth_head = nn.Sequential(
            nn.Conv2d(channels + 8, 32, 3, padding=1), nn.SiLU(), nn.Conv2d(32, 1, 1)
        )
        nn.init.zeros_(self.depth_head[-1].weight)
        nn.init.zeros_(self.depth_head[-1].bias)
        self.appearance_feedback = appearance_feedback
        self.rgb_feedback_step = rgb_feedback_step
        self.thermal_feedback_step = thermal_feedback_step
        self.differentiate_feedback = differentiate_feedback
        self.thermal_feedback_blur = thermal_feedback_blur
        self.thermal_feedback_consensus = thermal_feedback_consensus
        self.anchor_strength = anchor_strength
        self.anchor_radius = anchor_radius
        if appearance_feedback:
            self.render_projection = nn.Conv2d(8, channels, 1)
            nn.init.zeros_(self.render_projection.weight)
            nn.init.zeros_(self.render_projection.bias)
        self.projection = nn.Conv2d(64, 24, 1)
        self.neck = nn.Sequential(
            nn.Conv2d(44, channels, 3, padding=1),
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(8, channels),
            nn.SiLU(),
        )
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )
        self.appearance = nn.Sequential(
            nn.Conv2d(channels + 4, channels, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(channels, 8, 1),
        )
        self.plane_head = (
            nn.Sequential(nn.Linear(channels, 32), nn.SiLU(), nn.Linear(32, 4))
            if learn_planes or not compact
            else None
        )
        heads = [self.appearance[-1]]
        if self.plane_head is not None:
            heads.append(self.plane_head[-1])
        for head in heads:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        self.config = {
            "channels": channels,
            "learn_planes": learn_planes,
            "flatten": flatten,
            "rgb_limit": rgb_limit,
            "anchored": anchored,
            "appearance_strength": appearance_strength,
            "appearance_trust": appearance_trust,
            "surface_only": surface_only,
            "plane_cov": plane_cov,
            "thermal_fusion": thermal_fusion,
            "rgb_fusion": rgb_fusion,
            "verified_only": verified_only,
            "large_plane_fallback": large_plane_fallback,
            "geometry_guard": geometry_guard,
            "guard_margin": guard_margin,
            "visibility_fusion": visibility_fusion,
            "thermal_sigma": thermal_sigma,
            "dense_depth_limit": dense_depth_limit,
            "plane_normal_limit": plane_normal_limit,
            "plane_offset_limit": plane_offset_limit,
            "appearance_trust_mode": appearance_trust_mode,
            "appearance_feedback": appearance_feedback,
            "rgb_feedback_step": rgb_feedback_step,
            "thermal_feedback_step": thermal_feedback_step,
            "differentiate_feedback": differentiate_feedback,
            "thermal_feedback_blur": thermal_feedback_blur,
            "thermal_feedback_consensus": thermal_feedback_consensus,
            "anchor_strength": anchor_strength,
            "anchor_radius": anchor_radius,
            "compact": compact,
        }
        self.last_association = None

    def surface(self, initial, association, rgb, thermal, source_valid):
        association = projection_association(initial, association, source_valid)
        surface = apply_consensus(initial, association, flatten=self.flatten)
        if self.plane_cov:
            surface = plane_covariance(surface, association)
        if self.geometry_guard:
            association, audit = guard_surface(
                initial,
                surface,
                association,
                rgb,
                thermal,
                source_valid,
                self.guard_margin,
            )
            self.last_guard_audit = audit
            surface = select_geometry(initial, surface, association["ids"])
        if self.thermal_fusion or self.rgb_fusion:
            if self.visibility_fusion:
                surface = visible_appearance(
                    surface,
                    rgb,
                    thermal,
                    association,
                    source_valid,
                    self.thermal_fusion,
                    self.rgb_fusion,
                    thermal_sigma=self.thermal_sigma,
                )
            else:
                surface = consensus_appearance(
                    surface,
                    rgb,
                    thermal,
                    association,
                    self.thermal_fusion,
                    self.rgb_fusion,
                )
        return surface, association

    def forward(
        self,
        rgb,
        thermal,
        extrinsics=None,
        intrinsics=None,
        source_valid=None,
        initial=None,
        association=None,
        source_geometry=None,
    ):
        if initial is None:
            raise ValueError("Surface refinement requires a cross-view reconstruction")
        if association is None:
            association = build_consensus(initial, rgb, source_valid)
            if self.anchored:
                association = anchor_consensus(
                    initial,
                    rgb,
                    association,
                    observations=None
                    if source_geometry is None
                    else source_geometry.observations(initial),
                )
        if self.verified_only:
            ids = association["ids"].clone()
            for index, record in enumerate(
                association.get(
                    "anchor_audit", [{"accepted": False}] * len(association["groups"])
                )
            ):
                group = association["groups"][index]
                area = sum(group["counts"]) / len(group["counts"])
                large = (
                    self.large_plane_fallback
                    and len(group["views"]) >= max(2, (rgb.shape[1] + 1) // 2)
                    and area >= 0.20 * (rgb.shape[-2] * rgb.shape[-1] / 16)
                )
                if not record["accepted"] and not large:
                    ids[ids == index] = -1
            association = {**association, "ids": ids}
        b, v, _, h, w = rgb.shape
        with torch.autocast(device_type=rgb.device.type, enabled=False):
            surface, association = self.surface(
                initial, association, rgb, thermal, source_valid
            )
            z = gaussian_depth(initial).clamp_min(1e-5)
            shift = (surface.gaussians.means - initial.gaussians.means).reshape(
                b, v, h, w, 3
            ) / z[..., None]
            gs = initial.gaussians
            ids = association["ids"]
            normal = torch.zeros_like(shift)
            if len(association["normals"]):
                normal = (
                    association["normals"][ids.clamp_min(0)] * (ids >= 0)[..., None]
                )
            colors = torch.cat(
                [
                    gs.harmonics[..., 0] * 0.28209479177387814 + 0.5,
                    initial.thermal_colors,
                ],
                -1,
            ).reshape(b, v, h, w, 4)
            geometry = (
                torch.cat(
                    [
                        colors,
                        shift,
                        normal,
                        gs.scales.reshape(b, v, h, w, 3).clamp_min(1e-6).log() / 5,
                        gs.opacities.reshape(b, v, h, w, 1),
                        (z / z.flatten(2).median(-1).values[:, :, None, None]).log()[
                            ..., None
                        ],
                        (ids >= 0).float()[..., None],
                    ],
                    -1,
                )
                .permute(0, 1, 4, 2, 3)
                .flatten(0, 1)
            )
        low = (h // 4, w // 4)
        features = self.projection(
            F.interpolate(
                initial.features.flatten(0, 1).float(),
                low,
                mode="bilinear",
                align_corners=False,
            )
        )
        source = torch.cat([rgb, thermal], 2).flatten(0, 1)
        x = self.neck(
            torch.cat(
                [
                    features,
                    F.interpolate(
                        torch.cat([source, geometry], 1),
                        low,
                        mode="bilinear",
                        align_corners=False,
                    ),
                ],
                1,
            )
        )
        if self.appearance_feedback:
            from ..renderer import render_rgbt

            with torch.no_grad():
                rendered = render_rgbt(
                    surface, surface.extrinsics, surface.intrinsics, h, w
                )
                observed = torch.cat([rgb, thermal], 2)
                predicted = torch.cat([rendered["rgb"], rendered["thermal"]], 2)
                render_features = F.interpolate(
                    torch.cat([observed - predicted, predicted], 2).flatten(0, 1),
                    low,
                    mode="bilinear",
                    align_corners=False,
                )
            x = x + self.render_projection(render_features.to(x.dtype))
        if self.dense_depth_limit:
            with torch.autocast(device_type=rgb.device.type, enabled=False):
                feedback = geometry_feedback(initial, rgb, thermal, source_valid)
            x = x + self.feedback_projection(feedback.to(x.dtype))
        else:
            feedback = None
        x = x + self.block(x)
        plane_delta = x.new_zeros(len(association["normals"]), 4)
        if self.learn_planes and len(association["normals"]):
            labels = (
                F.interpolate(ids.float().flatten(0, 1)[:, None], low, mode="nearest")[
                    :, 0
                ]
                .long()
                .flatten()
            )
            xx = x.permute(0, 2, 3, 1).reshape(-1, x.shape[1])
            pooled = []
            for g in range(len(association["normals"])):
                keep = labels == g
                pooled.append(xx[keep].mean(0) if keep.any() else xx.mean(0) * 0)
            plane_delta = self.plane_head(torch.stack(pooled)).float().tanh()
            normals = F.normalize(
                association["normals"] + self.plane_normal_limit * plane_delta[:, :3],
                dim=-1,
            )
            pts = surface.gaussians.means.reshape(-1, 3)
            centers = torch.stack(
                [
                    pts[ids.flatten() == g].mean(0) if (ids == g).any() else pts.mean(0)
                    for g in range(len(normals))
                ]
            )
            radii = normals.new_tensor([g["radius"] for g in association["groups"]])
            offsets = (
                association["offsets"]
                - ((normals - association["normals"]) * centers).sum(-1)
                + self.plane_offset_limit * radii * plane_delta[:, 3]
            )
            association = {**association, "normals": normals, "offsets": offsets}
            with torch.autocast(device_type=rgb.device.type, enabled=False):
                surface, association = self.surface(
                    initial, association, rgb, thermal, source_valid
                )
        depth_delta = x.new_zeros(b, v, h, w)
        if self.dense_depth_limit:
            small_delta = self.depth_head(torch.cat([x, feedback.to(x.dtype)], 1))
            depth_delta = (
                F.interpolate(
                    small_delta.float(), (h, w), mode="bilinear", align_corners=False
                )[:, 0]
                .tanh()
                .reshape(b, v, h, w)
            )
            support = F.interpolate(feedback[:, 7:8], (h, w), mode="nearest")[
                :, 0
            ].reshape(b, v, h, w)
            active = (association["ids"] < 0).float() * support
            if source_valid is not None:
                active = active * (source_valid[:, :, 0] > 0.5)
            depth_delta = depth_delta * active
            ratio = (self.dense_depth_limit * depth_delta).exp().reshape(b, -1, 1)
            centers = (
                affine_inverse(initial.extrinsics)[..., :3, 3][:, :, None, None]
                .expand(-1, -1, h, w, -1)
                .reshape(b, -1, 3)
            )
            gs = surface.gaussians
            surface = replace(
                surface,
                gaussians=replace(
                    gs,
                    means=centers + (gs.means - centers) * ratio,
                    scales=gs.scales * ratio,
                ),
            )
        if self.anchor_strength:
            from ..source_anchors import source_anchor_field

            with torch.autocast(device_type=rgb.device.type, enabled=False):
                if association.get("source_anchor_radius") == self.anchor_radius:
                    correction = association["source_anchor_delta"]
                else:
                    correction, _ = source_anchor_field(
                        initial,
                        rgb,
                        source_valid,
                        self.anchor_radius,
                        observations=None
                        if source_geometry is None
                        else source_geometry.observations(initial),
                    )
                correction = correction * (association["ids"] < 0)
                ratio = (self.anchor_strength * correction).exp().reshape(b, -1, 1)
                centers = (
                    affine_inverse(initial.extrinsics)[..., :3, 3][:, :, None, None]
                    .expand(-1, -1, h, w, -1)
                    .reshape(b, -1, 3)
                )
                gs = surface.gaussians
                surface = replace(
                    surface,
                    gaussians=replace(
                        gs,
                        means=centers + (gs.means - centers) * ratio,
                        scales=gs.scales * ratio,
                    ),
                )
        self.last_depth_delta = depth_delta
        delta = (
            self.appearance(
                torch.cat(
                    [
                        F.interpolate(x, (h, w), mode="bilinear", align_corners=False),
                        source,
                    ],
                    1,
                )
            )
            .float()
            .tanh()
        )
        delta = delta * self.appearance_strength
        if self.surface_only:
            delta = delta * (association["ids"] >= 0).float().reshape(b * v, 1, h, w)
        delta = delta.reshape(b, v, 8, h, w).permute(0, 1, 3, 4, 2).reshape(b, -1, 8)
        ds, da, dr, dt = delta.split([3, 1, 3, 1], -1)
        gs = surface.gaussians
        scales = gs.scales * torch.exp(0.65 * ds)
        alpha = (
            torch.logit(gs.opacities.clamp(1e-5, 1 - 1e-5)) + 1.5 * da[..., 0]
        ).sigmoid()
        sh = gs.harmonics.clone()
        sh[..., 0] = sh[..., 0] + self.rgb_limit * dr / 0.28209479177387814
        self.last_association = association
        updated = replace(
            surface,
            gaussians=replace(gs, scales=scales, opacities=alpha, harmonics=sh),
            thermal_colors=(surface.thermal_colors + 0.12 * dt).clamp(0, 1),
            regularization=delta.square().mean()
            + 0.1 * depth_delta.square().mean()
            + (plane_delta.square().mean() if plane_delta.numel() else 0),
            fused_rendering=True,
        )
        if self.appearance_trust:
            from ..local_trust import spatial_trust

            updated = spatial_trust(
                surface,
                updated,
                rgb,
                thermal,
                source_valid,
                detach=not self.training,
                mode=self.appearance_trust_mode,
            )
        if self.rgb_feedback_step or self.thermal_feedback_step:
            with (
                torch.set_grad_enabled(
                    torch.is_grad_enabled() and self.differentiate_feedback
                ),
                torch.autocast(device_type=rgb.device.type, enabled=False),
            ):
                correction = source_color_residual(
                    updated,
                    rgb,
                    thermal,
                    source_valid,
                    self.thermal_feedback_blur,
                    self.thermal_feedback_consensus,
                )
            sh = updated.gaussians.harmonics.clone()
            sh[..., 0] += (
                self.rgb_feedback_step * correction[..., :3] / 0.28209479177387814
            )
            updated = replace(
                updated,
                gaussians=replace(updated.gaussians, harmonics=sh),
                thermal_colors=(
                    updated.thermal_colors
                    + self.thermal_feedback_step * correction[..., 3:]
                ).clamp(0, 1),
            )
        return updated
