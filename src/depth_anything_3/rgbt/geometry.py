"""Camera-z anchored Gaussians and visibility-aware multiview depth constraints."""

import torch
import torch.nn.functional as F


def pixel_grid(height, width, device):
    y, x = torch.meshgrid(
        torch.arange(height, device=device) + 0.5,
        torch.arange(width, device=device) + 0.5,
        indexing="ij",
    )
    return torch.stack([x, y], -1)


def depth_to_range(depth, intrinsics, xy):
    """The upstream GS adapter uses unit rays; DA3 depth is camera-z depth."""
    homogeneous = torch.cat([xy, torch.ones_like(xy[..., :1])], -1)
    rays = torch.einsum("bvij,bvhwj->bvhwi", intrinsics.inverse(), homogeneous)
    return depth * rays.norm(dim=-1)


def gaussian_depth(output):
    """Actual z coordinates of the final Gaussian centers in their source cameras."""
    b, v, h, w = output.depth.shape
    points = output.gaussians.means.reshape(b, v, h, w, 3)
    return (
        torch.einsum("bvij,bvhwj->bvhwi", output.extrinsics[..., :3, :3], points)
        + output.extrinsics[..., :3, 3][:, :, None, None]
    )[..., 2]


def thermal_edge_confidence(rgb, thermal):
    """Highlight source pixels with strong thermal edges and weak RGB edges."""
    b, v, _, h, w = rgb.shape
    rgb = rgb.float()
    thermal = thermal.float()
    luminance = (
        rgb[:, :, :1] * 0.299
        + rgb[:, :, 1:2] * 0.587
        + rgb[:, :, 2:3] * 0.114
    )
    images = torch.cat([luminance, thermal], 2).flatten(0, 1)
    sobel_x = images.new_tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
    )[None, None] / 8.0
    sobel_y = sobel_x.transpose(-1, -2)
    gradients_x = F.conv2d(
        images, sobel_x.repeat(2, 1, 1, 1), padding=1, groups=2
    )
    gradients_y = F.conv2d(
        images, sobel_y.repeat(2, 1, 1, 1), padding=1, groups=2
    )
    magnitude = (gradients_x.square() + gradients_y.square() + 1e-8).sqrt()
    magnitude = magnitude.reshape(b, v, 2, h, w)
    scale = torch.quantile(magnitude.flatten(-2), 0.95, dim=-1)[..., None, None]
    strength = (magnitude / scale.clamp_min(1e-4)).clamp(0, 1)
    return strength[:, :, 1:2] * (1 - strength[:, :, :1])


def multiview_depth_loss(output, valid, stride=4):
    """Project actual centers into other source depth maps; mask occlusion/borders.

    Detached reference depths and masks prevent moving both sides together or
    minimizing the loss by shrinking visibility. Sparse depth fixes global scale.
    """
    z = gaussian_depth(output)
    b, v, h, w = z.shape
    points = output.gaussians.means.reshape(b, v, h, w, 3)
    losses = []
    for i in range(v):
        p = points[:, i, ::stride, ::stride]
        source_valid = valid[:, i, 0, ::stride, ::stride] > 0.5
        for j in range(v):
            if i == j:
                continue
            cam = torch.einsum("bij,bhwj->bhwi", output.extrinsics[:, j, :3, :3], p)
            cam = cam + output.extrinsics[:, j, :3, 3][:, None, None]
            uvz = torch.einsum("bij,bhwj->bhwi", output.intrinsics[:, j], cam)
            uv = uvz[..., :2] / uvz[..., 2:].clamp_min(1e-6)
            grid = 2 * uv / uv.new_tensor([w, h]) - 1
            reference = F.grid_sample(
                z[:, j : j + 1].detach(), grid, align_corners=False
            )[:, 0]
            reference_valid = (
                F.grid_sample(valid[:, j], grid, align_corners=False, mode="nearest")[
                    :, 0
                ]
                > 0.5
            )
            inside = (grid.detach().abs() < 0.98).all(-1)
            visible = cam[..., 2].detach() <= reference * 1.05
            mask = source_valid & reference_valid & inside & visible & (cam[..., 2] > 0)
            mask &= reference > 0
            difference = (
                cam[..., 2].clamp_min(1e-6).log() - reference.clamp_min(1e-6).log()
            ).abs()
            losses.append((difference * mask).sum() / mask.sum().clamp_min(1))
    return torch.stack(losses).mean() if losses else z.sum() * 0
