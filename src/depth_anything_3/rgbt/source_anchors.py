"""Local depth correction supported only by triangulated source observations.

Unlike a camera-wide scale fit, interpolation respects image distance, color,
and depth discontinuities. Conflicting or absent observations attenuate the
correction. This module never consumes target images or reconstruction points.
"""

import torch
import torch.nn.functional as F

from .calibration import triangulate_sources
from .geometry import gaussian_depth


@torch.no_grad()
def interpolate_anchor_field(depth, rgb, valid, observations, radius=32.0):
    """Return bounded log-depth corrections, with zero for unsupported regions."""
    _views, h, w = depth.shape
    low = (max(1, h // 4), max(1, w // 4))
    y, x = torch.meshgrid(
        torch.arange(low[0], device=depth.device),
        torch.arange(low[1], device=depth.device),
        indexing="ij",
    )
    xy = torch.stack([(x + 0.5) * w / low[1], (y + 0.5) * h / low[0]], -1).reshape(
        -1, 2
    )
    result, counts = [], []
    for i, data in enumerate(observations):
        delta = torch.zeros(low, device=depth.device)
        count = 0
        if len(data) >= 8:
            p = torch.as_tensor(data, device=depth.device, dtype=torch.float32)
            grid = 2 * p[:, :2] / p.new_tensor([w, h]) - 1

            def sample(t, grid=grid):
                return F.grid_sample(t[None], grid[None, None], align_corners=False)[
                    0, :, 0
                ].T

            pred = sample(depth[i, None])[:, 0]
            error = (p[:, 2].clamp_min(1e-6) / pred.clamp_min(1e-6)).log()
            color = sample(rgb[i])
            keep = (error.abs() < 0.25) & (pred > 0) & (p[:, 2] > 0)
            keep &= (grid.abs() < 0.98).all(-1)
            if valid is not None:
                keep &= sample(valid[i])[:, 0] > 0.99
            edge = (
                F.max_pool2d(depth[i, None, None], 5, 1, 2)
                + F.max_pool2d(-depth[i, None, None], 5, 1, 2)
            )[0] / depth[i, None].clamp_min(1e-6)
            keep &= sample(edge)[:, 0] < 0.10
            p, error, color = p[keep], error[keep], color[keep]
            count = len(p)
            if count:
                zz = F.interpolate(
                    depth[i, None, None], low, mode="bilinear", align_corners=False
                ).flatten()
                cc = (
                    F.interpolate(
                        rgb[i, None], low, mode="bilinear", align_corners=False
                    )[0]
                    .permute(1, 2, 0)
                    .reshape(-1, 3)
                )
                parts = []
                # Bound peak memory even when a textured image has many matches.
                for start in range(0, len(xy), 1024):
                    sl = slice(start, start + 1024)
                    weight = torch.exp(
                        -(xy[sl, None] - p[None, :, :2]).square().sum(-1)
                        / (2 * radius**2)
                        - 0.5
                        * (
                            (zz[sl, None].clamp_min(1e-6).log() - p[None, :, 2].log())
                            / 0.12
                        ).square()
                        - 0.5 * (cc[sl, None] - color[None]).square().mean(-1) / 0.25**2
                    )
                    mass = weight.sum(-1)
                    mean = (weight * error[None]).sum(-1) / mass.clamp_min(1e-6)
                    variance = (weight * (error[None] - mean[:, None]).square()).sum(
                        -1
                    ) / mass.clamp_min(1e-6)
                    confidence = (mass / 4).clamp(0, 1) * torch.exp(-variance / 0.04**2)
                    parts.append(mean.clamp(-0.12, 0.12) * confidence)
                delta = torch.cat(parts).reshape(low)
        counts.append(count)
        result.append(delta)
    correction = F.interpolate(
        torch.stack(result)[:, None], (h, w), mode="bilinear", align_corners=False
    )[:, 0]
    if valid is not None:
        correction = correction * (valid[:, 0] > 0.5)
    return correction, counts


@torch.no_grad()
def source_anchor_field(output, rgb, valid, radius=32.0, observations=None):
    depth = gaussian_depth(output).float()
    fields, counts = [], []
    for b in range(len(rgb)):
        observed = (
            triangulate_sources(rgb[b], output.extrinsics[b], output.intrinsics[b])
            if observations is None
            else observations[b]
        )
        field, count = interpolate_anchor_field(
            depth[b],
            rgb[b].float(),
            None if valid is None else valid[b],
            observed,
            radius,
        )
        fields.append(field)
        counts.append(count)
    return torch.stack(fields), counts
