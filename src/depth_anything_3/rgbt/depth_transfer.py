"""Transfer independently supported surfaces between source cameras only."""

import torch
from .multiview_depth import project
from .geometry import gaussian_depth


@torch.no_grad()
def transfer_depth(rgb, initial, depth, confidence, valid):
    b, v, h, w = depth.shape
    if b != 1 or v < 2:
        return depth, torch.zeros_like(confidence)
    E = initial.extrinsics[0].float()
    K = initial.intrinsics[0].float()
    centers = torch.linalg.inv(E)[:, :3, 3]
    z0 = gaussian_depth(initial)[0].float().clamp_min(1e-6)
    rays = (
        initial.gaussians.means[0].float().reshape(v, h, w, 3) - centers[:, None, None]
    ) / z0[..., None]
    xyz = centers[:, None, None] + rays * depth[0, ..., None]
    colors = rgb[0].float().permute(0, 2, 3, 1)
    result = depth.clone()
    reliability = torch.zeros_like(confidence)
    for i in range(v):
        indices = []
        zs = []
        ws = []
        for j in range(v):
            if i == j:
                continue
            grid, z = project(xyz[j], E[i], K[i], h, w)
            uv = (grid + 1) * grid.new_tensor([w, h]) * 0.5 - 0.5
            base = uv.floor().long()
            fraction = uv - base
            for dx, dy in [(0, 0), (1, 0), (0, 1), (1, 1)]:
                x, y = base[..., 0] + dx, base[..., 1] + dy
                inside = (
                    (x >= 0)
                    & (x < w)
                    & (y >= 0)
                    & (y < h)
                    & (z > 0)
                    & (confidence[0, j] > 0.55)
                    & (valid[0, j] > 0.5)
                )
                xx = x.clamp(0, w - 1)
                yy = y.clamp(0, h - 1)
                difference = (colors[j] - colors[i, yy, xx]).square().mean(-1)
                weight = confidence[0, j] * torch.exp(-difference / (0.06**2))
                weight *= (fraction[..., 0] if dx else 1 - fraction[..., 0]) * (
                    fraction[..., 1] if dy else 1 - fraction[..., 1]
                )
                inside &= (
                    (valid[0, i, yy, xx] > 0.5)
                    & (difference < 0.10**2)
                    & (weight > 0.025)
                )
                indices.append((yy * w + xx)[inside])
                zs.append(z[inside])
                ws.append(weight[inside])
        ids = torch.cat(indices)
        values = torch.cat(zs)
        weight = torch.cat(ws)
        if ids.numel() == 0:
            continue
        front = torch.full((h * w,), float("inf"), device=depth.device)
        front.scatter_reduce_(0, ids, values, reduce="amin", include_self=True)
        weight = weight * (values < front[ids] * 1.06)
        numerator = torch.zeros_like(front)
        denominator = torch.zeros_like(front)
        numerator.scatter_add_(0, ids, weight / values.clamp_min(1e-6))
        denominator.scatter_add_(0, ids, weight)
        z = (denominator / numerator.clamp_min(1e-6)).reshape(h, w)
        mass = denominator.reshape(h, w)
        accepted = (mass > 0.25) & (confidence[0, i] < 0.55) & (valid[0, i] > 0.5)
        result[0, i] = torch.where(accepted, z, result[0, i])
        reliability[0, i] = torch.where(accepted, (mass / 1.0).clamp(0.65, 1), 0)
    return result, reliability
