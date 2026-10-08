"""One source-conditioned residual update on the pretrained DA3 Gaussians.

Preserve the jointly learned position offsets, anisotropic covariance and SH.
There are no reference predictions, source renders, candidate selection or
target-dependent corrections in this path. Training supervises the rendered
result, including edges and alpha, rather than imitating an anchored teacher.
"""

import torch
import torch.nn.functional as F
from gsplat.cuda._wrapper import spherical_harmonics
from torch import nn

from .attention import ProjectedNeighborhood, WindowNeighborhood
from .gaussian_update import GaussianUpdate


class NativeGaussianRefiner(nn.Module):
    def __init__(
        self,
        mode="native_residual",
        channels=96,
        attention_pairs=2,
        depth_limit=0.10,
        scale_limit=0.7,
        opacity_limit=2.0,
        rgb_limit=0.25,
        color_limit=0.3,
        position_strength=1.0,
        rotation_update_limit=0.02,
        source_rgb_residual=False,
    ):
        super().__init__()
        if mode != "native_residual":
            raise ValueError(mode)
        self.config = {
            "mode": mode,
            "channels": channels,
            "attention_pairs": attention_pairs,
            "depth_limit": depth_limit,
            "scale_limit": scale_limit,
            "opacity_limit": opacity_limit,
            "rgb_limit": rgb_limit,
            "color_limit": color_limit,
            "position_strength": position_strength,
            "rotation_update_limit": rotation_update_limit,
            "source_rgb_residual": source_rgb_residual,
        }
        self.core = GaussianUpdate(
            channels=channels,
            blocks=0,
            iterations=1,
            render_feedback=False,
            support_context=True,
            compact=True,
            fused_rendering=True,
            depth_limit=depth_limit,
            scale_limit=scale_limit,
            opacity_limit=opacity_limit,
            rgb_limit=rgb_limit,
            color_limit=color_limit,
            position_strength=position_strength,
        )
        self.core.blocks = nn.ModuleList(
            [
                block(channels)
                for _ in range(attention_pairs)
                for block in (ProjectedNeighborhood, WindowNeighborhood)
            ]
        )
        # Native splats include long, thin surface elements. Clamping all three
        # axes to a pixel-size interval destroys their learned covariance.
        self.core.min_update_pixel_scale = 0.0
        self.core.max_update_pixel_scale = float("inf")
        self.core.rotation_update_limit = rotation_update_limit
        self.core.bound_total_depth = False
        self.core.aligned_points = True
        self.core._scale_with_depth = True
        self.last_association = None
        self.source_rgb_residual = source_rgb_residual

    @staticmethod
    def observed_rgb_direction(initial, rgb):
        """Source color evidence at each Gaussian's actual projected center."""
        b, v, _, h, w = rgb.shape
        gs = initial.gaussians
        points = gs.means.reshape(b, v, h, w, 3)
        E, K = initial.extrinsics.float(), initial.intrinsics.float()
        cam = (
            torch.einsum("bvij,bvhwj->bvhwi", E[..., :3, :3], points)
            + E[..., :3, 3][:, :, None, None]
        )
        uvz = torch.einsum("bvij,bvhwj->bvhwi", K, cam)
        grid = (
            2 * uvz[..., :2] / uvz[..., 2:].clamp_min(1e-5) / rgb.new_tensor([w, h]) - 1
        )
        observed = F.grid_sample(
            rgb.flatten(0, 1).float(), grid.flatten(0, 1), align_corners=False
        )
        observed = observed.permute(0, 2, 3, 1).reshape(b, v, h, w, 3)
        # Camera-to-point directions in world coordinates; retain all RGB SH.
        directions = torch.einsum("bvji,bvhwj->bvhwi", E[..., :3, :3], cam)
        predicted = (
            spherical_harmonics(
                2,
                directions.reshape(-1, 3).contiguous(),
                gs.harmonics.reshape(-1, 3, 9).transpose(-1, -2).contiguous(),
            )
            + 0.5
        )
        valid = (grid.abs() < 1).all(-1) & (cam[..., 2] > 0)
        return (observed - predicted.reshape(b, v, h, w, 3)).clamp(-1, 1) * valid[
            ..., None
        ]

    def forward(
        self,
        rgb,
        thermal,
        extrinsics=None,
        intrinsics=None,
        initial=None,
        source_valid=None,
        prepared=None,
    ):
        if prepared is not None:
            raise ValueError("Native residual inference does not use rendered feedback")
        with torch.no_grad(), torch.autocast(rgb.device.type, enabled=False):
            self.core._rgb_residual_direction = (
                self.observed_rgb_direction(initial, rgb)
                if self.source_rgb_residual
                else None
            )
        try:
            return self.core(
                rgb,
                thermal,
                extrinsics,
                intrinsics,
                initial=initial,
                source_valid=source_valid,
                share_projection=True,
            )
        finally:
            self.core._rgb_residual_direction = None
