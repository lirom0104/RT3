"""Shared source-conditioned Gaussian state update.

All observations are source images. Target cameras/images are accepted only by
the renderer/training loop, never by this module. The frozen initial reconstruction is
the initial state; zero initialized updates reproduce it exactly.
"""

from dataclasses import replace

import torch
import torch.nn.functional as F
from torch import nn

from depth_anything_3.utils.geometry import affine_inverse

from .attention import ProjectedNeighborhood
from .geometry import gaussian_depth
from .reconstruction import ConvBlock
from .renderer import render_rgbt


class GaussianUpdate(nn.Module):
    def __init__(
        self,
        channels=96,
        blocks=2,
        iterations=1,
        render_feedback=False,
        depth_limit=0.10,
        color_limit=0.15,
        scale_limit=1.2,
        opacity_limit=3.0,
        rgb_limit=None,
        support_context=False,
        position_strength=1.0,
        source_guard=False,
        fused_rendering=False,
        compact=False,
        thermal_edge=False,
        thermal_edge_channels=16,
        thermal_edge_geometry_limit=0.25,
    ):
        super().__init__()
        self.config = {
            "channels": channels,
            "blocks": blocks,
            "iterations": iterations,
            "render_feedback": render_feedback,
            "depth_limit": depth_limit,
            "color_limit": color_limit,
            "scale_limit": scale_limit,
            "opacity_limit": opacity_limit,
            "rgb_limit": rgb_limit,
            "support_context": support_context,
            "position_strength": position_strength,
            "source_guard": source_guard,
            "fused_rendering": fused_rendering,
            "compact": compact,
            "thermal_edge": thermal_edge,
            "thermal_edge_channels": thermal_edge_channels,
            "thermal_edge_geometry_limit": thermal_edge_geometry_limit,
        }
        self.iterations = iterations
        self.render_feedback = render_feedback
        self.depth_limit, self.color_limit = depth_limit, color_limit
        self.scale_limit, self.opacity_limit = scale_limit, opacity_limit
        self.rgb_limit = color_limit if rgb_limit is None else rgb_limit
        self.support_context = support_context
        self.position_strength = position_strength
        self.source_guard = source_guard
        self.fused_rendering = fused_rendering
        self.thermal_edge = thermal_edge
        self.thermal_edge_geometry_limit = thermal_edge_geometry_limit
        if source_guard and (
            depth_limit != 0 or position_strength != 0 or self.rgb_limit != 0
        ):
            raise ValueError(
                "Source trust selection currently requires fixed centers, rotations and RGB SH"
            )
        # 64 foundation features + RGB/T + 14 geometry/appearance + RGB/T/alpha feedback.
        self.neck = ConvBlock(64 + 4 + 14 + 5 + (2 if support_context else 0), channels)
        self.blocks = nn.ModuleList(
            [ProjectedNeighborhood(channels) for _ in range(blocks)]
        )
        self.state_conv = (
            nn.Conv2d(channels, channels, 1) if iterations > 1 or not compact else None
        )
        self.head = nn.Sequential(
            nn.Conv2d(channels + 4, channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels, 15, 1),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)
        if thermal_edge:
            # Thermal gradients are useful only where they disagree with weak RGB
            # structure. The final projections are zero initialized so enabling the
            # branch starts exactly at the RGB-only refiner.
            self.thermal_edge_encoder = nn.Sequential(
                nn.Conv2d(8, thermal_edge_channels, 3, padding=1),
                nn.GELU(),
                nn.Conv2d(thermal_edge_channels, thermal_edge_channels, 3, padding=1),
                nn.GELU(),
            )
            self.thermal_edge_context = nn.Conv2d(thermal_edge_channels, channels, 1)
            self.thermal_edge_gate = nn.Conv2d(thermal_edge_channels, 1, 1)
            self.thermal_edge_geometry = nn.Conv2d(thermal_edge_channels, 3, 1)
            for module in [
                self.thermal_edge_context,
                self.thermal_edge_geometry,
            ]:
                nn.init.zeros_(module.weight)
                nn.init.zeros_(module.bias)
            nn.init.zeros_(self.thermal_edge_gate.weight)
            nn.init.constant_(self.thermal_edge_gate.bias, -1.5)
            sobel_x = torch.tensor(
                [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
            ) / 8.0
            sobel_y = sobel_x.t()
            self.register_buffer("thermal_edge_sobel_x", sobel_x[None, None])
            self.register_buffer("thermal_edge_sobel_y", sobel_y[None, None])

    def forward(
        self,
        rgb,
        thermal,
        extrinsics=None,
        intrinsics=None,
        initial=None,
        return_initial=False,
        source_valid=None,
        share_projection=False,
    ):
        if initial is None:
            raise ValueError(
                "Neighborhood refinement requires an initial reconstruction"
            )
        o = replace(initial, fused_rendering=self.fused_rendering)
        b, v, _, h, w = rgb.shape
        lh, lw = (h + 3) // 4, (w + 3) // 4
        source = torch.cat([rgb, thermal], 2).flatten(0, 1)
        source_low = F.interpolate(source, (lh, lw), mode="area")
        edge_features = edge_gate = None
        if self.thermal_edge:
            rgb_low, thermal_low = source_low[:, :3], source_low[:, 3:4]
            luminance = (
                rgb_low[:, :1] * 0.299
                + rgb_low[:, 1:2] * 0.587
                + rgb_low[:, 2:3] * 0.114
            )
            thermal_dx = F.conv2d(
                thermal_low, self.thermal_edge_sobel_x, padding=1
            )
            thermal_dy = F.conv2d(
                thermal_low, self.thermal_edge_sobel_y, padding=1
            )
            rgb_dx = F.conv2d(luminance, self.thermal_edge_sobel_x, padding=1)
            rgb_dy = F.conv2d(luminance, self.thermal_edge_sobel_y, padding=1)
            thermal_mag = torch.sqrt(thermal_dx.square() + thermal_dy.square() + 1e-6)
            rgb_mag = torch.sqrt(rgb_dx.square() + rgb_dy.square() + 1e-6)
            edge_input = torch.cat(
                [
                    thermal_low,
                    thermal_dx,
                    thermal_dy,
                    thermal_mag,
                    rgb_mag,
                    (thermal_mag - rgb_mag).abs(),
                    thermal_mag / (thermal_mag + rgb_mag + 1e-3),
                    rgb_mag / (thermal_mag + rgb_mag + 1e-3),
                ],
                1,
            )
            edge_features = self.thermal_edge_encoder(edge_input)
            edge_gate = torch.sigmoid(self.thermal_edge_gate(edge_features))
            if source_valid is not None:
                valid_low = F.interpolate(
                    source_valid.flatten(0, 1), (lh, lw), mode="area"
                )
                edge_gate = edge_gate * valid_low
        feature = F.interpolate(
            initial.features.flatten(0, 1),
            (lh, lw),
            mode="bilinear",
            align_corners=False,
        )
        E, K = o.extrinsics.float(), o.intrinsics.float()
        with torch.autocast(device_type=rgb.device.type, enabled=False):
            c2w = affine_inverse(E)
            z0 = gaussian_depth(initial).detach()
            if self.support_context:
                if v > 1:
                    pairs = torch.triu_indices(v, v, 1, device=rgb.device)
                    centers = c2w[..., :3, 3]
                    baseline = (
                        (centers[:, pairs[0]] - centers[:, pairs[1]])
                        .norm(dim=-1)
                        .median(-1)
                        .values.clamp_min(1e-4)
                    )
                else:
                    baseline = z0.flatten(1).median(-1).values.clamp_min(1e-4)
                support = torch.stack(
                    [
                        (z0 / baseline[:, None, None, None])
                        .clamp_min(1e-5)
                        .log()
                        .clamp(-6, 8),
                        torch.full_like(z0, v / 6),
                    ],
                    2,
                ).flatten(0, 1)
                support = F.interpolate(support, (lh, lw), mode="area")
        median = z0.flatten(2).median(-1).values[:, :, None, None].clamp_min(1e-4)
        pixel = (
            z0 / K[..., :2, :2].diagonal(dim1=-2, dim2=-1).mean(-1)[:, :, None, None]
        )
        state = None
        regularization = z0.new_zeros(())
        for _ in range(self.iterations):
            gs = o.gaussians
            with torch.autocast(device_type=rgb.device.type, enabled=False):
                z = gaussian_depth(o)
            conf = o.depth_conf.float().clamp_min(1e-5).log()
            conf = (conf - conf.flatten(2).mean(-1)[:, :, None, None]).clamp(-5, 5)
            scales = gs.scales.reshape(b, v, h, w, 3)
            rotations = gs.rotations.reshape(b, v, h, w, 4)
            opacity = gs.opacities.reshape(b, v, h, w, 1)
            dc = gs.harmonics[..., 0].reshape(b, v, h, w, 3) * 0.28209479177387814 + 0.5
            tc = o.thermal_colors.reshape(b, v, h, w, 1)
            geo = torch.cat(
                [
                    (z / median).clamp_min(1e-5).log()[..., None],
                    conf[..., None],
                    (scales / pixel[..., None].clamp_min(1e-6))
                    .clamp_min(1e-5)
                    .log()
                    .clamp(-5, 5),
                    rotations,
                    opacity,
                    dc,
                    tc,
                ],
                -1,
            )
            geo = geo.permute(0, 1, 4, 2, 3).flatten(0, 1)
            geo = F.interpolate(geo, (lh, lw), mode="area")
            if self.render_feedback:
                # Source-only feedback; detached render prevents unbounded recurrent graphs.
                with torch.no_grad():
                    rendered = (
                        self.source_feedback(o, E, K, h, w)
                        if hasattr(self, "source_feedback")
                        else render_rgbt(o, E, K, h, w)
                    )
                    err = torch.cat(
                        [
                            rendered["rgb"] - rgb,
                            rendered["thermal"] - thermal,
                            rendered["alpha"],
                        ],
                        2,
                    )
                    err = F.interpolate(err.flatten(0, 1), (lh, lw), mode="area")
            else:
                err = geo.new_zeros(b * v, 5, lh, lw)
            inputs = [feature, source_low, geo, err]
            if self.support_context:
                inputs.append(support)
            x = self.neck(torch.cat(inputs, 1))
            phase = getattr(self, "_phase_index", None)
            preserve_first = (
                getattr(self, "preserve_first_update", False) and phase == 0
            )
            if phase is not None and not preserve_first:
                x = x + self.phase_embedding.weight[phase][None, :, None, None]
            modulation = getattr(self, "phase_modulation", None)
            phase_affine = None
            if phase is not None and modulation is not None and not preserve_first:
                phase_affine = modulation.weight[phase].reshape(
                    len(self.blocks) + 2, 2, x.shape[1]
                )
                x = (
                    x * (1 + phase_affine[0, 0][None, :, None, None])
                    + phase_affine[0, 1][None, :, None, None]
                )
            ordered_context = getattr(self, "_ordered_context", None)
            if ordered_context is not None:
                x = x + self.ordered_projection(ordered_context)
            geometry_context = getattr(self, "geometry_context", None)
            extra_geometry = getattr(self, "_extra_geometry", None)
            if geometry_context is not None and extra_geometry is not None:
                x = x + geometry_context(
                    F.interpolate(extra_geometry, (lh, lw), mode="area")
                )
            context = getattr(self, "global_context", None)
            if context is not None:
                pooled = x.reshape(b, v, -1, lh, lw).mean((1, 3, 4))[:, :, None, None]
                x = x + context(pooled)[:, None].expand(-1, v, -1, -1, -1).flatten(0, 1)
            if state is not None:
                x = x + self.state_conv(state)
            shared_residual = getattr(self, "shared_residual", None)
            if shared_residual is not None and phase in [1, 2]:
                x = x + shared_residual(x)
            if self.thermal_edge and phase in [1, 2]:
                x = x + self.thermal_edge_context(edge_features) * edge_gate
            x = x.reshape(b, v, -1, lh, lw)
            pts = (
                gs.means.detach()
                .reshape(b, v, h, w, 3)
                .permute(0, 1, 4, 2, 3)
                .flatten(0, 1)
            )
            interpolation = (
                {"mode": "bilinear", "align_corners": False}
                if getattr(self, "aligned_points", False)
                else {"mode": "nearest"}
            )
            pts = (
                F.interpolate(pts, (lh, lw), **interpolation)
                .reshape(b, v, 3, lh, lw)
                .permute(0, 1, 3, 4, 2)
            )
            # Geometry is unchanged across attention blocks in this iteration.
            # Scope this cache to the iteration, never across Gaussian updates.
            correspondence = {} if share_projection else None
            for block_index, blk in enumerate(self.blocks):
                if not (
                    getattr(self, "preserve_first_update", False)
                    and phase == 0
                    and block_index % 2 == 1
                ):
                    x = blk(x, pts, E, K, (h, w), correspondence=correspondence)
                if phase_affine is not None:
                    scale, shift = phase_affine[block_index + 1]
                    x = (
                        x * (1 + scale[None, None, :, None, None])
                        + shift[None, None, :, None, None]
                    )
            state = x.flatten(0, 1)
            if shared_residual is not None and phase in [1, 2]:
                state = state + shared_residual(state)
            if phase_affine is not None:
                scale, shift = phase_affine[-1]
                state = (
                    state * (1 + scale[None, :, None, None])
                    + shift[None, :, None, None]
                )
            # Full-resolution RGB/T skip preserves edges in the update field.
            up = F.interpolate(state, (h, w), mode="bilinear", align_corners=False)
            delta = self.head(torch.cat([up, source], 1)).float().tanh()
            delta = delta.reshape(b, v, 15, h, w).permute(0, 1, 3, 4, 2)
            if self.thermal_edge and phase in [1, 2]:
                edge_delta = self.thermal_edge_geometry(edge_features)
                edge_delta = F.interpolate(edge_delta, (h, w), mode="bilinear")
                edge_delta = edge_delta.reshape(b, v, 3, h, w).permute(0, 1, 3, 4, 2)
                edge_gate_full = F.interpolate(edge_gate, (h, w), mode="bilinear")
                edge_gate_full = edge_gate_full.reshape(b, v, 1, h, w)
                edge_delta = (
                    edge_delta.tanh()
                    * edge_gate_full.permute(0, 1, 3, 4, 2)
                    * self.thermal_edge_geometry_limit
                )
                delta = torch.cat(
                    [
                        delta[..., :2] + edge_delta[..., :2],
                        delta[..., 2:3] + edge_delta[..., 2:3],
                        delta[..., 3:],
                    ],
                    -1,
                )
            if getattr(self, "capture_delta", False):
                self.last_delta = delta
            regularization = regularization + delta.square().mean()
            xy, dz, ds, dq, da, dr, dt = delta.split([2, 1, 3, 4, 1, 3, 1], -1)
            depth_mask = getattr(self, "_depth_mask", None)
            if depth_mask is not None:
                dz = dz * depth_mask[..., None]
            with torch.autocast(device_type=rgb.device.type, enabled=False):
                cam = (
                    torch.einsum(
                        "bvij,bvhwj->bvhwi",
                        E[..., :3, :3],
                        gs.means.reshape(b, v, h, w, 3),
                    )
                    + E[..., :3, 3][:, :, None, None]
                )
                uvz = torch.einsum("bvij,bvhwj->bvhwi", K, cam)
                uv = uvz[..., :2] / uvz[..., 2:].clamp_min(1e-6) + xy * (
                    0.5 * self.position_strength
                )
                final_z = (z * torch.exp(dz[..., 0] * self.depth_limit)).clamp(min=1e-4)
                # Total displacement bounded relative to the initial surface, across iterations.
                if getattr(self, "bound_total_depth", True):
                    final_z = torch.maximum(
                        z0 * (1 - self.depth_limit),
                        torch.minimum(final_z, z0 * (1 + self.depth_limit)),
                    )
                ray = torch.einsum(
                    "bvij,bvhwj->bvhwi",
                    K.inverse(),
                    torch.cat([uv, torch.ones_like(uv[..., :1])], -1),
                )
                cam = ray * final_z[..., None]
                means = (
                    torch.einsum("bvij,bvhwj->bvhwi", c2w[..., :3, :3], cam)
                    + c2w[..., :3, 3][:, :, None, None]
                )
                new_scales = scales * torch.exp(ds * self.scale_limit)
                if getattr(self, "_scale_with_depth", False):
                    new_scales = new_scales * (final_z / z.clamp_min(1e-6))[..., None]
                new_scales = torch.maximum(
                    pixel[..., None] * getattr(self, "min_update_pixel_scale", 0.10),
                    torch.minimum(
                        new_scales,
                        pixel[..., None] * getattr(self, "max_update_pixel_scale", 6.0),
                    ),
                )
                quat = F.normalize(
                    rotations
                    + dq
                    * getattr(
                        self, "rotation_update_limit", 0.1 * self.position_strength
                    ),
                    dim=-1,
                )
                alpha = (
                    torch.logit(opacity.clamp(1e-5, 1 - 1e-5)) + da * self.opacity_limit
                ).sigmoid()
                sh = gs.harmonics.clone()
                rgb_direction = getattr(self, "_rgb_residual_direction", None)
                if rgb_direction is not None:
                    dr = dr * rgb_direction
                sh[..., 0] = (
                    sh[..., 0]
                    + dr.reshape(b, -1, 3) * self.rgb_limit / 0.28209479177387814
                )
                new_gs = replace(
                    gs,
                    means=means.reshape(b, -1, 3),
                    scales=new_scales.reshape(b, -1, 3),
                    rotations=quat.reshape(b, -1, 4),
                    opacities=alpha.reshape_as(gs.opacities),
                    harmonics=sh,
                )
                o = replace(
                    o,
                    gaussians=new_gs,
                    thermal_colors=(tc + dt * self.color_limit)
                    .clamp(0, 1)
                    .reshape(b, -1, 1),
                    regularization=regularization,
                )
        if self.source_guard:
            o = self.guard_update(initial, o, rgb, source_valid, thermal=thermal)
        return (o, initial) if return_initial else o

    def guard_update(self, initial, updated, rgb, valid=None, thermal=None):
        """Select trust strength using observed source RGB only, never target data.

        Five fixed network-update candidates are scored; no optimizer or weight
        fitting runs at inference. Selection is discrete and conservative on ties.
        """
        b, v, _, h, w = rgb.shape
        if valid is None:
            valid = torch.ones_like(rgb[:, :, :1])
        if valid.shape != (b, v, 1, h, w):
            raise ValueError("Source validity must match the source image grid")
        strengths = rgb.new_tensor([0.0, 0.25, 0.5, 0.75, 1.0])
        errors = []
        with (
            torch.no_grad(),
            torch.autocast(device_type=rgb.device.type, enabled=False),
        ):
            for strength in strengths:
                gs = replace(
                    initial.gaussians,
                    scales=initial.gaussians.scales.lerp(
                        updated.gaussians.scales, strength
                    ),
                    opacities=initial.gaussians.opacities.lerp(
                        updated.gaussians.opacities, strength
                    ),
                )
                candidate = replace(initial, gaussians=gs)
                rendered = render_rgbt(
                    candidate,
                    initial.extrinsics,
                    initial.intrinsics,
                    h,
                    w,
                    rgb_only=True,
                )
                difference = (rendered["rgb"].clamp(0, 1) - rgb).square() * valid
                errors.append(
                    difference.flatten(1).sum(-1)
                    / (valid.flatten(1).sum(-1) * 3).clamp_min(1)
                )
            strength = strengths[torch.stack(errors, 1).argmin(1)]
        gs = replace(
            initial.gaussians,
            scales=initial.gaussians.scales.lerp(
                updated.gaussians.scales, strength[:, None, None]
            ),
            opacities=initial.gaussians.opacities.lerp(
                updated.gaussians.opacities, strength[:, None]
            ),
        )
        return replace(
            updated,
            gaussians=gs,
            thermal_colors=initial.thermal_colors.lerp(
                updated.thermal_colors, strength[:, None, None]
            ),
            refinement_strength=strength,
        )
