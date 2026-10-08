"""Local and cross-view attention shared by current and historical refiners."""

import torch
import torch.nn.functional as F
from torch import nn


class ProjectedNeighborhood(nn.Module):
    """Attend to nearby 3D points from local pixels and projected source views.

    Projection bounds candidates without constructing an N-by-N distance matrix.
    Depth discontinuities and occluded matches are masked in camera space.
    """

    def __init__(self, channels):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.qk = nn.Conv2d(channels, 64, 1)
        self.value = nn.Conv2d(channels, channels, 1)
        self.out = nn.Conv2d(channels, channels, 1)
        self.ffn = nn.Sequential(
            nn.GroupNorm(8, channels),
            nn.Conv2d(channels, channels * 2, 1),
            nn.GELU(),
            nn.Conv2d(channels * 2, channels, 1),
        )

    @staticmethod
    @torch.no_grad()
    def project_pair(points, z, E, K, i, j, image_shape):
        ih, iw = image_shape
        with torch.autocast(device_type=points.device.type, enabled=False):
            cam = (
                torch.einsum("bij,bhwj->bhwi", E[:, j, :3, :3], points[:, i])
                + E[:, j, :3, 3][:, None, None]
            )
            uvz = torch.einsum("bij,bhwj->bhwi", K[:, j], cam)
            grid = (
                2
                * (uvz[..., :2] / uvz[..., 2:].clamp_min(1e-5))
                / cam.new_tensor([iw, ih])
                - 1
            )
            zj = F.grid_sample(z[:, j, None], grid, align_corners=False)[:, 0]
            good = (grid.abs() < 0.98).all(-1) & (cam[..., 2] > 0)
            good &= (cam[..., 2] - zj).abs() < zj.clamp_min(1e-4) * 0.10
        return grid, good

    def forward(self, x, points, E, K, image_shape, correspondence=None):
        b, v, c, h, w = x.shape
        correspondence = {} if correspondence is None else correspondence
        norm = self.norm(x.flatten(0, 1))
        q, k = self.qk(norm).reshape(b, v, 64, h, w).chunk(2, 2)
        values = self.value(norm).reshape(b, v, c, h, w)
        result = []
        # Neighborhood selection is discrete geometry; gradients flow through features.
        with torch.no_grad(), torch.autocast(device_type=x.device.type, enabled=False):
            cams = torch.einsum("bvij,bvhwj->bvhwi", E[..., :3, :3], points)
            cams = cams + E[..., :3, 3][:, :, None, None]
            z = cams[..., 2]
        for i in range(v):
            keys, vals, masks = (
                [k[:, i]],
                [values[:, i]],
                [torch.ones_like(z[:, i], dtype=torch.bool)],
            )
            # Four same-view neighbors, excluding discontinuous surfaces.
            for dy, dx in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                kk = torch.roll(k[:, i], (dy, dx), (-2, -1))
                vv = torch.roll(values[:, i], (dy, dx), (-2, -1))
                zz = torch.roll(z[:, i], (dy, dx), (-2, -1))
                good = (zz - z[:, i]).abs() < z[:, i].abs().clamp_min(1e-4) * 0.10
                if dy > 0:
                    good[:, :1] = False
                if dy < 0:
                    good[:, -1:] = False
                if dx > 0:
                    good[:, :, :1] = False
                if dx < 0:
                    good[:, :, -1:] = False
                keys.append(kk)
                vals.append(vv)
                masks.append(good)
            for j in range(v):
                if i == j:
                    continue
                key = ("projected", i, j)
                if key not in correspondence:
                    correspondence[key] = self.project_pair(
                        points, z, E, K, i, j, image_shape
                    )
                grid, good = correspondence[key]
                keys.append(
                    F.grid_sample(k[:, j].float(), grid, align_corners=False).to(
                        x.dtype
                    )
                )
                vals.append(
                    F.grid_sample(values[:, j].float(), grid, align_corners=False).to(
                        x.dtype
                    )
                )
                masks.append(good)
            kk = torch.stack(keys, 2)
            logits = (q[:, i, :, None].float() * kk.float()).sum(1) / (32**0.5)
            logits = logits.masked_fill(~torch.stack(masks, 1), -1e4)
            weights = logits.softmax(1).to(x.dtype)
            aggregate = (torch.stack(vals, 2) * weights[:, None]).sum(2)
            result.append(aggregate)
        update = self.out(torch.stack(result, 1).flatten(0, 1))
        out = x.flatten(0, 1) + update
        return (out + self.ffn(out)).reshape(b, v, c, h, w)


class WindowNeighborhood(ProjectedNeighborhood):
    """Search a projected 3x3 window and use its closest 3D surface candidate.

    Geometry and features use the same pixel-center convention. Confidence is
    continuous within a hard occlusion bound rather than accepting every match
    with identical prior weight. Window search can tolerate projection errors.
    """

    @staticmethod
    @torch.no_grad()
    def window_pair(points, E, K, i, j, image_shape):
        b, _views, h, w, _ = points.shape
        ih, iw = image_shape
        with torch.autocast(device_type=points.device.type, enabled=False):
            cam = (
                torch.einsum("bij,bhwj->bhwi", E[:, j, :3, :3], points[:, i])
                + E[:, j, :3, 3][:, None, None]
            )
            uvz = torch.einsum("bij,bhwj->bhwi", K[:, j], cam)
            grid = (
                2
                * uvz[..., :2]
                / uvz[..., 2:].clamp_min(1e-6)
                / cam.new_tensor([iw, ih])
                - 1
            )
            offsets = cam.new_tensor(
                [(dx * 2 / w, dy * 2 / h) for dy in [-1, 0, 1] for dx in [-1, 0, 1]]
            )
            candidates = grid[:, None] + offsets[None, :, None, None]
            sampled = F.grid_sample(
                points[:, j].permute(0, 3, 1, 2),
                candidates.reshape(b, 9 * h, w, 2),
                align_corners=False,
            )
            sampled = sampled.reshape(b, 3, 9, h, w).permute(0, 2, 3, 4, 1)
            distance = (sampled - points[:, i, None]).norm(dim=-1) / cam[
                ..., 2
            ].abs().clamp_min(1e-4)[:, None]
            inside = (candidates.abs() < 0.985).all(-1) & (cam[..., 2] > 0)[:, None]
            distance = distance.masked_fill(~inside, 1e3)
            best, index = distance.min(1)
            chosen = candidates.gather(
                1, index[:, None, :, :, None].expand(-1, 1, -1, -1, 2)
            )[:, 0]
            prior = (-best / 0.025).clamp_min(-1e4)
            prior = prior.masked_fill(best > 0.08, -1e4)
        return chosen, prior

    def forward(self, x, points, E, K, image_shape, correspondence=None):
        b, views, c, h, w = x.shape
        correspondence = {} if correspondence is None else correspondence
        norm = self.norm(x.flatten(0, 1))
        q, k = self.qk(norm).reshape(b, views, 64, h, w).chunk(2, 2)
        value = self.value(norm).reshape(b, views, c, h, w)
        results = []
        for i in range(views):
            keys, vals = [k[:, i]], [value[:, i]]
            priors = [torch.zeros((b, h, w), device=x.device)]
            for j in range(views):
                if i == j:
                    continue
                key = ("window", i, j)
                if key not in correspondence:
                    correspondence[key] = self.window_pair(
                        points, E, K, i, j, image_shape
                    )
                chosen, prior = correspondence[key]
                keys.append(
                    F.grid_sample(k[:, j].float(), chosen, align_corners=False).to(
                        x.dtype
                    )
                )
                vals.append(
                    F.grid_sample(value[:, j].float(), chosen, align_corners=False).to(
                        x.dtype
                    )
                )
                priors.append(prior)
            logits = (q[:, i, :, None].float() * torch.stack(keys, 2).float()).sum(
                1
            ) / 32**0.5
            weights = (logits + torch.stack(priors, 1)).softmax(1).to(x.dtype)
            results.append((torch.stack(vals, 2) * weights[:, None]).sum(2))
        update = self.out(torch.stack(results, 1).flatten(0, 1))
        out = x.flatten(0, 1) + update
        return (out + self.ffn(out)).reshape(b, views, c, h, w)
