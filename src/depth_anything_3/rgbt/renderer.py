"""RGB, thermal and expected depth in a shared rasterization pass."""

from dataclasses import replace

import torch
from gsplat import rasterization
from gsplat.cuda._wrapper import spherical_harmonics

from depth_anything_3.utils.geometry import affine_inverse


def render_rgbt_fused(output, extrinsics, intrinsics, height, width):
    gs = output.gaussians
    results = {"rgb": [], "thermal": [], "depth": [], "alpha": []}
    for b in range(gs.means.shape[0]):
        E = extrinsics[b].float()
        centers = affine_inverse(E)[..., :3, 3]
        dirs = gs.means[b].float()[None] - centers[:, None]
        coeff = gs.harmonics[b].transpose(-1, -2).contiguous().float()
        coeff = coeff[None].expand(E.shape[0], -1, -1, -1).contiguous()
        rgb = (spherical_harmonics(2, dirs.contiguous(), coeff) + 0.5).clamp_min(0)
        thermal = output.thermal_colors[b].float()[None].expand(E.shape[0], -1, -1)
        if output.thermal_harmonics is not None:
            tc = (
                output.thermal_harmonics[b, :, :, None]
                .float()[None]
                .expand(E.shape[0], -1, -1, 3)
                .contiguous()
            )
            thermal = (
                spherical_harmonics(2, dirs.contiguous(), tc)[..., :1] + 0.5
            ).clamp_min(0)
        image, alpha, _ = rasterization(
            means=gs.means[b].float(),
            quats=gs.rotations[b].float(),
            scales=gs.scales[b].float(),
            opacities=gs.opacities[b].float().reshape(-1),
            colors=torch.cat([rgb, thermal], -1),
            viewmats=E,
            Ks=intrinsics[b].float(),
            width=width,
            height=height,
            packed=False,
            near_plane=0.001,
            far_plane=1e5,
            render_mode="RGB+ED",
        )
        results["rgb"].append(image[..., :3].permute(0, 3, 1, 2))
        results["thermal"].append(image[..., 3:4].permute(0, 3, 1, 2))
        results["depth"].append(image[..., 4])
        results["alpha"].append(alpha.permute(0, 3, 1, 2))
    return {name: torch.stack(values) for name, values in results.items()}


def render_rgbt(output, extrinsics, intrinsics, height, width, rgb_only=False):
    """Exact pixel intrinsics, including the principal point; same geometry twice."""
    if output.fused_rendering and not rgb_only:
        return render_rgbt_fused(output, extrinsics, intrinsics, height, width)
    from gsplat import rasterization

    gs = output.gaussians
    rgbs, thermals, depths, alphas = [], [], [], []
    for i in range(gs.means.shape[0]):
        common = {
            "means": gs.means[i].float(),
            "quats": gs.rotations[i].float(),
            "scales": gs.scales[i].float(),
            "opacities": gs.opacities[i].float().reshape(-1),
            "viewmats": extrinsics[i].float(),
            "Ks": intrinsics[i].float(),
            "width": width,
            "height": height,
            "packed": False,
            "near_plane": 0.001,
            "far_plane": 1e5,
        }
        r, a, _ = rasterization(
            **common,
            colors=gs.harmonics[i].transpose(-1, -2).contiguous().float(),
            sh_degree=2,
            render_mode="RGB+ED",
        )
        if rgb_only:
            t = torch.zeros_like(r[..., :1])
        else:
            if output.thermal_harmonics is not None:
                t, _, _ = rasterization(
                    **common,
                    colors=output.thermal_harmonics[i, :, :, None]
                    .expand(-1, -1, 3)
                    .contiguous()
                    .float(),
                    sh_degree=2,
                    render_mode="RGB",
                )
                t = t[..., :1]
            else:
                t, _, _ = rasterization(
                    **common, colors=output.thermal_colors[i].float(), render_mode="RGB"
                )
        rgbs.append(r[..., :3].permute(0, 3, 1, 2))
        thermals.append(t.permute(0, 3, 1, 2))
        depths.append(r[..., -1])
        alphas.append(a.permute(0, 3, 1, 2))
    return {
        "rgb": torch.stack(rgbs),
        "thermal": torch.stack(thermals),
        "depth": torch.stack(depths),
        "alpha": torch.stack(alphas),
    }


def leave_one_out(output, E, K, h, w, selected=None):
    _b, views, sh, sw = output.depth.shape
    if views == 1:
        # A single capture has no cross-view observations, including MM5 depth use.
        return {
            "rgb": E.new_zeros(_b, 1, 3, h, w),
            "thermal": E.new_zeros(_b, 1, 1, h, w),
            "depth": E.new_zeros(_b, 1, h, w),
            "alpha": E.new_zeros(_b, 1, 1, h, w),
        }
    pixels = sh * sw
    renders = []
    for i in range(views) if selected is None else [selected]:
        keep = torch.arange(views * pixels, device=E.device) // pixels != i
        gs = output.gaussians
        subset = replace(
            gs,
            **{
                name: getattr(gs, name)[:, keep]
                for name in ["means", "scales", "rotations", "harmonics", "opacities"]
            },
        )
        o = replace(
            output, gaussians=subset, thermal_colors=output.thermal_colors[:, keep]
        )
        renders.append(render_rgbt(o, E[:, i : i + 1], K[:, i : i + 1], h, w))
    return {key: torch.cat([r[key] for r in renders], 1) for key in renders[0]}
