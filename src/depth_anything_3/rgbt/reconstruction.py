"""DA3 geometry + sparse thermal adapters + shared Gaussians, separate appearances.

Inputs are RGB and registered grayscale thermal in [0,1]. No target images are
accepted by forward. Known source cameras are optional; with them the output is
in their coordinate system. Without them DA3 predicts the coordinate system.
"""

from __future__ import annotations

from dataclasses import replace

import torch
import torch.nn.functional as F
from torch import nn

from depth_anything_3.cfg import create_object
from depth_anything_3.model.reference_view_selector import reorder_by_reference
from depth_anything_3.rgbt.geometry import depth_to_range, pixel_grid
from depth_anything_3.utils.geometry import (
    affine_inverse,
    as_homogeneous,
    map_pdf_to_opacity,
)

from .types import RGBTOutput


class ConvBlock(nn.Sequential):
    def __init__(self, cin, cout, stride=1):
        super().__init__(
            nn.Conv2d(cin, cout, 3, stride, 1),
            nn.GroupNorm(8, cout),
            nn.GELU(),
            nn.Conv2d(cout, cout, 3, 1, 1),
            nn.GroupNorm(8, cout),
            nn.GELU(),
        )


class ThermalEncoder(nn.Module):
    def __init__(self, channels=64, levels=4):
        super().__init__()
        if not 1 <= levels <= 4:
            raise ValueError("Thermal pyramid must have between one and four levels")
        self.stem = ConvBlock(1, channels, 2)
        self.stages = nn.ModuleList(
            [ConvBlock(channels, channels, 2) for _ in range(levels - 1)]
        )

    def forward(self, thermal):
        x = self.stem(thermal.flatten(0, 1) * 2 - 1)
        maps = [x]
        for stage in self.stages:
            x = stage(x)
            maps.append(x)
        return maps


class DeformableThermalAdapter(nn.Module):
    """Multi-scale, multi-head sampled cross-attention, implemented with grid_sample.

    Learned offsets are local residuals on a registered image grid. This module
    does not estimate a general RGB-to-thermal camera calibration.
    """

    def __init__(self, dim, channels=64, hidden=96, heads=4, points=4, levels=4):
        super().__init__()
        assert hidden % heads == 0
        self.heads, self.points, self.levels = heads, points, levels
        self.norm = nn.LayerNorm(dim)
        self.query = nn.Linear(dim, hidden)
        self.values = nn.ModuleList(
            [nn.Conv2d(channels, hidden, 1) for _ in range(levels)]
        )
        self.offsets = nn.Linear(hidden, heads * levels * points * 2)
        self.weights = nn.Linear(hidden, heads * levels * points)
        self.out = nn.Linear(hidden, dim)
        self.spatial_gate = nn.Linear(hidden, 1)
        self.gamma = nn.Parameter(torch.zeros(()))
        nn.init.zeros_(self.offsets.weight)
        nn.init.zeros_(self.weights.weight)
        nn.init.zeros_(self.weights.bias)
        # A noncollapsed initial sampling pattern; output projection stays nonzero
        # so the zero residual gate can receive a gradient on the first update.
        with torch.no_grad():
            angle = torch.arange(heads) * (2 * torch.pi / heads)
            direction = torch.stack([angle.cos(), angle.sin()], -1)
            bias = (
                direction[:, None, None]
                * torch.arange(1, points + 1)[None, None, :, None]
            )
            self.offsets.bias.copy_(bias.expand(heads, levels, points, 2).reshape(-1))

    def forward(self, tokens, maps, shape):
        b, v, n, dim = tokens.shape
        h, w = shape
        special = n - h * w
        q = self.query(self.norm(tokens[:, :, special:])).reshape(b * v, h * w, -1)
        offsets = self.offsets(q).reshape(
            b * v, h * w, self.heads, self.levels, self.points, 2
        )
        weights = self.weights(q).reshape(
            b * v, h * w, self.heads, self.levels * self.points
        )
        weights = weights.softmax(-1).reshape(
            b * v, h * w, self.heads, self.levels, self.points
        )
        yy, xx = torch.meshgrid(
            (torch.arange(h, device=q.device) + 0.5) / h,
            (torch.arange(w, device=q.device) + 0.5) / w,
            indexing="ij",
        )
        reference = torch.stack([xx, yy], -1).reshape(1, h * w, 1, 1, 2)
        combined = 0
        for level, (feature, proj) in enumerate(zip(maps, self.values)):
            value = proj(feature)
            _, hidden, fh, fw = value.shape
            grid = reference + offsets[:, :, :, level].tanh() * q.new_tensor(
                [4 / fw, 4 / fh]
            )
            grid = (
                (2 * grid - 1)
                .permute(0, 2, 1, 3, 4)
                .reshape(b * v * self.heads, h * w, self.points, 2)
            )
            value = value.reshape(b * v * self.heads, hidden // self.heads, fh, fw)
            sampled = F.grid_sample(
                value.float(), grid.float(), align_corners=False, padding_mode="border"
            )
            sampled = sampled.reshape(
                b * v, self.heads, hidden // self.heads, h * w, self.points
            )
            sampled = sampled.permute(0, 3, 1, 4, 2)
            combined = combined + (sampled * weights[:, :, :, level, :, None]).sum(3)
        delta = (
            self.out(combined.flatten(-2).to(q.dtype)) * self.spatial_gate(q).sigmoid()
        )
        patches = tokens[:, :, special:] + self.gamma.tanh() * delta.reshape(
            b, v, h * w, dim
        )
        return torch.cat([tokens[:, :, :special], patches], 2)


class GaussianReconstruction(nn.Module):
    def __init__(
        self,
        foundation,
        channels=64,
        adapter_hidden=96,
        adapter_layers=None,
        rgb_source_blend=0.0,
        geometry_mode="legacy",
        depth_residual_limit=0.03,
        rgb_residual_limit=None,
        min_pixel_scale=0.15,
        max_pixel_scale=2.0,
        opacity_floor=0.0,
        compact=False,
    ):
        super().__init__()
        config = foundation
        self.da3 = create_object(config)
        self.da3.requires_grad_(False)
        vit = self.da3.backbone.pretrained
        self.adapter_layers = (
            list(adapter_layers)
            if adapter_layers is not None
            else [
                len(vit.blocks) // 4 - 1,
                len(vit.blocks) // 2 - 1,
                3 * len(vit.blocks) // 4 - 1,
                len(vit.blocks) - 1,
            ]
        )
        self.thermal_encoder = ThermalEncoder(
            channels, levels=1 if compact and not self.adapter_layers else 4
        )
        self.adapters = nn.ModuleDict(
            {
                str(i): DeformableThermalAdapter(
                    vit.embed_dim, channels, adapter_hidden
                )
                for i in self.adapter_layers
            }
        )
        feature_dim = config["head"]["dim_in"]
        self.feature_projections = nn.ModuleList(
            [nn.Conv2d(feature_dim, channels, 1) for _ in range(4)]
        )
        self.neck = ConvBlock(channels * 2 + 4, channels)
        self.geometry_head = nn.Conv2d(
            channels, 11, 1
        )  # xy(2), scale(3), rotation(4), depth(1), opacity(1)
        effective_rgb_limit = (
            (0.0 if geometry_mode == "anchored" else 0.1)
            if rgb_residual_limit is None
            else rgb_residual_limit
        )
        self.rgb_head = (
            None if compact and effective_rgb_limit == 0 else nn.Conv2d(channels, 3, 1)
        )
        self.thermal_head = nn.Conv2d(channels, 1, 1)
        for head in [self.geometry_head, self.rgb_head, self.thermal_head]:
            if head is None:
                continue
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        self.register_buffer(
            "rgb_mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "rgb_std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1),
            persistent=False,
        )
        self.adapters_enabled = True
        if geometry_mode not in ("legacy", "anchored", "native"):
            raise ValueError(geometry_mode)
        self.geometry_mode = geometry_mode
        self.depth_residual_limit = float(depth_residual_limit)
        if not 0 <= self.depth_residual_limit < 1:
            raise ValueError("Depth residual limit must stay in [0,1)")
        self.rgb_residual_limit = float(
            (0.0 if geometry_mode == "anchored" else 0.1)
            if rgb_residual_limit is None
            else rgb_residual_limit
        )
        self.geometry_strength = 1.0
        self.min_pixel_scale = float(min_pixel_scale)
        self.max_pixel_scale = float(max_pixel_scale)
        self.opacity_floor = float(opacity_floor)
        if (
            not 0 < self.min_pixel_scale <= self.max_pixel_scale
            or not 0 <= self.opacity_floor < 1
        ):
            raise ValueError("Invalid Gaussian coverage limits")
        self.rgb_source_blend = float(rgb_source_blend)
        self.config = {
            "channels": channels,
            "adapter_hidden": adapter_hidden,
            "adapter_layers": self.adapter_layers,
            "rgb_source_blend": self.rgb_source_blend,
            "geometry_mode": self.geometry_mode,
            "depth_residual_limit": self.depth_residual_limit,
            "rgb_residual_limit": self.rgb_residual_limit,
            "min_pixel_scale": self.min_pixel_scale,
            "max_pixel_scale": self.max_pixel_scale,
            "opacity_floor": self.opacity_floor,
            "compact": compact,
        }

    def train(self, mode=True):
        super().train(mode)
        # Frozen weights may still need autograd to propagate to earlier adapters.
        self.da3.eval()
        self.da3.backbone.pretrained.train(mode)
        self.da3.backbone.pretrained.gradient_checkpointing = mode
        return self

    def trainable_state(self):
        names = {name for name, p in self.named_parameters() if p.requires_grad}
        return {k: v.detach().cpu() for k, v in self.state_dict().items() if k in names}

    def forward(self, rgb, thermal, extrinsics=None, intrinsics=None, teacher=False):
        b, v, _, h, w = rgb.shape
        if thermal.shape != (b, v, 1, h, w) or h % 14 or w % 14:
            raise ValueError(
                "Paired RGB/T grids must agree and H,W must be multiples of 14."
            )
        maps = self.thermal_encoder(thermal)

        def inject(index, tokens, shape, permutation):
            if (
                teacher
                or self.geometry_mode == "native"
                or not self.adapters_enabled
                or str(index) not in self.adapters
            ):
                return tokens
            reordered = maps
            if permutation is not None:
                reordered = [
                    reorder_by_reference(
                        m.reshape(b, v, *m.shape[1:]), permutation
                    ).flatten(0, 1)
                    for m in maps
                ]
            return self.adapters[str(index)](tokens, reordered, shape)

        x = (rgb - self.rgb_mean) / self.rgb_std
        with torch.autocast(device_type=rgb.device.type, enabled=False):
            cam_token = (
                self.da3.cam_enc(extrinsics.float(), intrinsics.float(), (h, w))
                if extrinsics is not None
                else None
            )
        feats, _ = self.da3.backbone(
            x,
            cam_token=cam_token,
            ref_view_strategy="first",
            token_adapter=inject if self.adapter_layers and self.adapters_enabled else None,
        )
        # DA3's pretrained heads require float32 activations.
        with torch.autocast(device_type=rgb.device.type, enabled=False):
            feats32 = [(f.float(), c.float()) for f, c in feats]
            depth_out = self.da3._process_depth_head(feats32, h, w)
            depth_out = self.da3._process_camera_estimation(feats32, h, w, depth_out)
            gs_out = self.da3.gs_head(
                feats32, h, w, patch_start_idx=0, images=x.float()
            )
        features = (
            sum(
                proj(
                    f.flatten(0, 1).transpose(1, 2).reshape(b * v, -1, h // 14, w // 14)
                )
                for proj, (f, _) in zip(self.feature_projections, feats)
            )
            / 4
        )
        size = maps[0].shape[-2:]
        features = F.interpolate(features, size, mode="bilinear", align_corners=False)
        pixels = F.interpolate(
            torch.cat([rgb, thermal], 2).flatten(0, 1),
            size,
            mode="bilinear",
            align_corners=False,
        )
        features = self.neck(torch.cat([features, maps[0], pixels], 1))

        def predict(head):
            return (
                F.interpolate(
                    head(features).float(), (h, w), mode="bilinear", align_corners=False
                )
                .reshape(b, v, -1, h, w)
                .permute(0, 1, 3, 4, 2)
            )

        geometry = predict(self.geometry_head).tanh()
        rgb_delta = (
            predict(self.rgb_head).tanh()
            if self.rgb_head is not None
            else torch.zeros_like(geometry[..., :3])
        )
        thermal_delta = predict(self.thermal_head).tanh()
        if teacher or self.geometry_mode == "native":
            geometry = torch.zeros_like(geometry)
            rgb_delta = torch.zeros_like(rgb_delta)
            thermal_delta = torch.zeros_like(thermal_delta)
        else:
            geometry = geometry * self.geometry_strength
        with torch.autocast(device_type=rgb.device.type, enabled=False):
            raw = gs_out.raw_gs.float()
            depth = depth_out.depth.float().clamp_min(1e-4)
            anchored = self.geometry_mode == "anchored"
            # Anchor Gaussians to camera-z depth with bounded residuals instead of
            # the unbounded pretrained Gaussian depth offset.
            raw = torch.cat(
                [
                    (0.25 * raw[..., :2].tanh() + 0.1 * geometry[..., :2])
                    if anchored
                    else raw[..., :2] + geometry[..., :2],
                    raw[..., 2:5] + (0.1 if anchored else 1.0) * geometry[..., 2:5],
                    raw[..., 5:9] + (0.02 if anchored else 0.1) * geometry[..., 5:9],
                    raw[..., 9:-1],
                    torch.zeros_like(raw[..., -1:])
                    if anchored
                    else raw[..., -1:] + 0.05 * depth[..., None] * geometry[..., 9:10],
                ],
                -1,
            )
            opacity = (
                torch.logit(
                    map_pdf_to_opacity(gs_out.raw_gs_conf).clamp(1e-5, 1 - 1e-5)
                )
                + (0.15 if anchored else 1.0) * geometry[..., 10]
            ).sigmoid()
            if anchored:
                opacity = self.opacity_floor + (1 - self.opacity_floor) * opacity
            ext = as_homogeneous(depth_out.extrinsics.float())
            intr = depth_out.intrinsics.float()
            if extrinsics is not None:
                if v < 2:
                    raise ValueError(
                        "Known-camera scale alignment requires at least two source views."
                    )
                pred_centers = affine_inverse(ext)[..., :3, 3]
                given_centers = affine_inverse(extrinsics.float())[..., :3, 3]
                pairs = torch.triu_indices(v, v, 1, device=rgb.device)
                pred_dist = (
                    pred_centers[:, pairs[0]] - pred_centers[:, pairs[1]]
                ).norm(dim=-1)
                given_dist = (
                    given_centers[:, pairs[0]] - given_centers[:, pairs[1]]
                ).norm(dim=-1)
                scale = (
                    (given_dist / pred_dist.clamp_min(1e-4))
                    .median(-1)
                    .values.detach()
                    .clamp(0.001 if anchored else 0.01, 1000 if anchored else 100)
                )
                depth = depth * scale[:, None, None, None]
                raw = torch.cat(
                    [raw[..., :-1], raw[..., -1:] * scale[:, None, None, None, None]],
                    -1,
                )
                ext, intr = extrinsics.float(), intrinsics.float()
            # Bound negative offsets to keep all Gaussians in front of source cameras.
            raw = torch.cat(
                [raw[..., :-1], torch.maximum(raw[..., -1:], -0.95 * depth[..., None])],
                -1,
            )
            xy = pixel_grid(h, w, rgb.device)[None, None] + raw[..., :2]
            gs_depth = depth
            if anchored:
                final_z = depth * (1 + self.depth_residual_limit * geometry[..., 9])
                gs_depth = depth_to_range(final_z, intr, xy)
            gs = self.da3.gs_adapter(ext, intr, gs_depth, opacity, raw, (h, w))
            if anchored:
                # Keep splats local to their supporting pixels. This is expressed
                # in world units using the actual depth and source focal length.
                pixel_scale = (
                    final_z
                    / intr[..., :2, :2]
                    .diagonal(dim1=-2, dim2=-1)
                    .mean(-1)[:, :, None, None]
                )
                lower = (self.min_pixel_scale * pixel_scale).reshape(b, -1, 1)
                upper = (self.max_pixel_scale * pixel_scale).reshape(b, -1, 1)
                gs = replace(
                    gs, scales=torch.maximum(lower, torch.minimum(gs.scales, upper))
                )
            # Sample thermal appearance at the Gaussian's actual subpixel anchor.
            yy, xx = torch.meshgrid(
                torch.arange(h, device=rgb.device) + 0.5,
                torch.arange(w, device=rgb.device) + 0.5,
                indexing="ij",
            )
            xy = torch.stack([xx, yy], -1)[None, None] + raw[..., :2]
            grid = 2 * xy / raw.new_tensor([w, h]) - 1
            sh = gs.harmonics.clone()
            if self.rgb_source_blend and self.geometry_mode != "native":
                observed_rgb = F.grid_sample(
                    rgb.flatten(0, 1).float(),
                    grid.flatten(0, 1),
                    align_corners=False,
                    padding_mode="border",
                )
                observed_rgb = observed_rgb.permute(0, 2, 3, 1).reshape(b, -1, 3)
                sh = sh * (1 - self.rgb_source_blend)
                sh[..., 0] = (
                    sh[..., 0]
                    + self.rgb_source_blend * (observed_rgb - 0.5) / 0.28209479177387814
                )
            sh[..., 0] = (
                sh[..., 0]
                + self.rgb_residual_limit
                * rgb_delta.reshape(b, -1, 3)
                / 0.28209479177387814
            )
            gs = replace(gs, harmonics=sh)
            t = F.grid_sample(
                thermal.flatten(0, 1).float(),
                grid.flatten(0, 1),
                align_corners=False,
                padding_mode="border",
            )
            t = t.reshape(b, v, h, w, 1)
            t = (t + 0.15 * thermal_delta).clamp(0, 1).reshape(b, -1, 1)
            reg = (
                geometry.square().mean()
                + 0.1 * rgb_delta.square().mean()
                + 0.1 * thermal_delta.square().mean()
            )
        return RGBTOutput(
            gs,
            t,
            depth,
            ext,
            intr,
            depth_out.depth_conf,
            reg,
            features.reshape(b, v, *features.shape[1:]),
        )
