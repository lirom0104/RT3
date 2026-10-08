"""Source-only validity, geometry rejection and visibility for shared surfaces."""

from dataclasses import replace

import torch
import torch.nn.functional as F

from depth_anything_3.utils.geometry import affine_inverse

from .geometry import gaussian_depth
from .renderer import render_rgbt


@torch.no_grad()
def projection_association(output, association, source_valid=None):
    """One point mask for center projection, covariance and observation fusion."""
    ids = association["ids"].clone()
    if len(association["normals"]):
        b, v, h, w = ids.shape
        n = association["normals"][ids.clamp_min(0)]
        d = association["offsets"][ids.clamp_min(0)]
        centers = affine_inverse(output.extrinsics)[..., :3, 3][:, :, None, None]
        ray = output.gaussians.means.reshape(b, v, h, w, 3).float() - centers
        denom = (n * ray).sum(-1)
        ratio = -(d + (n * centers).sum(-1)) / torch.where(
            denom.abs() > 1e-8, denom, torch.ones_like(denom)
        )
        valid = (ratio > 0.75) & (ratio < 1.25) & (denom.abs() > 1e-5)
        ids[~valid] = -1
    if source_valid is not None:
        ids[source_valid[:, :, 0] <= 0.5] = -1
    return {**association, "ids": ids}


@torch.no_grad()
def guard_surface(
    initial, candidate, association, rgb, thermal, valid=None, margin=0.15
):
    """Reject whole plane groups using observed RGB/T renders, before appearance.

    Hard group decisions preserve accepted common planes. They are detached;
    accepted candidate centers retain gradients when selected below.
    """
    ids = association["ids"].clone()
    if not (ids >= 0).any():
        return association, []
    h, w = rgb.shape[-2:]
    renders = [
        render_rgbt(o, initial.extrinsics, initial.intrinsics, h, w)
        for o in [initial, candidate]
    ]
    errors = []
    for r in renders:
        errors.append(
            (r["rgb"].clamp(0, 1) - rgb).abs().mean(2)
            + 0.25 * (r["thermal"].clamp(0, 1) - thermal).abs().mean(2)
            + 0.03 * (0.9 - r["alpha"][:, :, 0]).clamp_min(0)
        )
    audit = []
    for i in range(len(association["normals"])):
        mask = ids == i
        if valid is not None:
            mask &= valid[:, :, 0] > 0.5
        if not mask.any():
            continue
        old, new = [float(e[mask].mean()) for e in errors]
        accept = new <= old * (1 + margin) + 0.002
        audit.append({"group": i, "before": old, "after": new, "accepted": accept})
        if not accept:
            ids[ids == i] = -1
    return {**association, "ids": ids}, audit


def select_geometry(initial, candidate, ids):
    """Restore all geometric parameters of rejected points with exact identity."""
    mask = ids.reshape(ids.shape[0], -1) >= 0
    gs = candidate.gaussians
    return replace(
        candidate,
        gaussians=replace(
            gs,
            **{
                k: torch.where(
                    mask[..., None], getattr(gs, k), getattr(initial.gaussians, k)
                )
                for k in ["means", "scales", "rotations"]
            },
        ),
    )


def visible_appearance(
    output,
    rgb,
    thermal,
    association,
    source_valid=None,
    thermal_blend=1.0,
    rgb_blend=0.5,
    depth_tolerance=0.03,
    thermal_sigma=0.10,
):
    """Fuse only depth-consistent samples, with separate thermal disagreement weights."""
    gs = output.gaussians
    b, v, h, w = output.depth.shape
    if b != 1 or not (association["ids"] >= 0).any():
        return output
    points = gs.means[0].float()
    ids = association["ids"][0].reshape(-1)
    E, K = output.extrinsics[0].float(), output.intrinsics[0].float()
    own = torch.cat(
        [gs.harmonics[0, ..., 0] * 0.28209479177387814 + 0.5, output.thermal_colors[0]],
        -1,
    )
    total, mass = own.clone(), torch.ones_like(own)
    images = torch.cat([rgb, thermal], 2)[0].float()
    depth = gaussian_depth(output).detach()
    origin = torch.arange(len(points), device=points.device) // (h * w)
    for j in range(v):
        cam = points @ E[j, :3, :3].T + E[j, :3, 3]
        pix = cam @ K[j].T
        grid = (
            2 * pix[:, :2] / pix[:, 2:].clamp_min(1e-6) / pix.new_tensor([w, h]) - 1
        ).reshape(1, 1, -1, 2)
        sampled = F.grid_sample(images[j : j + 1], grid, align_corners=False)[0, :, 0].T
        labels = F.grid_sample(
            association["ids"][:, j : j + 1].float(),
            grid,
            mode="nearest",
            align_corners=False,
        )[0, 0, 0].long()
        z = F.grid_sample(depth[:, j : j + 1], grid, align_corners=False)[0, 0, 0]
        relative = (cam[:, 2] - z).abs() / z.clamp_min(1e-5)
        active = (
            (grid[0, 0].abs() < 0.98).all(-1)
            & (labels == ids)
            & (ids >= 0)
            & (origin != j)
            & (cam[:, 2] > 0)
            & (z > 0)
            & (relative < depth_tolerance)
        )
        if source_valid is not None:
            sample_valid = F.grid_sample(
                source_valid[:, j].float(), grid, mode="nearest", align_corners=False
            )[0, 0, 0]
            active &= sample_valid > 0.5
        weight = active.float().detach() * torch.exp(
            -(sampled[:, :3] - own[:, :3]).square().mean(-1) / 0.02
        )
        tw = weight * torch.exp(
            -(sampled[:, 3] - own[:, 3]).square() / max(thermal_sigma**2, 1e-6)
        )
        weights = torch.cat([weight[:, None].expand(-1, 3), tw[:, None]], -1)
        total = total + weights * sampled
        mass = mass + weights
    average = total / mass
    sh = gs.harmonics.clone()
    sh[..., 0] += (
        rgb_blend * (average[None, :, :3] - own[None, :, :3]) / 0.28209479177387814
    )
    return replace(
        output,
        gaussians=replace(gs, harmonics=sh),
        thermal_colors=output.thermal_colors.lerp(average[None, :, 3:], thermal_blend),
    )


@torch.no_grad()
def geometry_feedback(output, rgb, thermal, valid=None):
    """Low-resolution cross-source evidence for bounded nonplanar depth updates."""
    b, v, h, w = output.depth.shape
    low = (h // 4, w // 4)
    points = output.gaussians.means.reshape(b, v, h, w, 3)
    small = F.interpolate(
        points.permute(0, 1, 4, 2, 3).flatten(0, 1),
        low,
        mode="bilinear",
        align_corners=False,
    ).reshape(b, v, 3, *low)
    images = torch.cat([rgb, thermal], 2).float()
    own = F.interpolate(
        images.flatten(0, 1), low, mode="bilinear", align_corners=False
    ).reshape(b, v, 4, *low)
    depth = gaussian_depth(output)
    feats = []
    for i in range(v):
        p = small[:, i].permute(0, 2, 3, 1)
        accum = p.new_zeros(b, 6, *low)
        mass = p.new_zeros(b, 1, *low)
        for j in range(v):
            if i == j:
                continue
            cam = torch.einsum("bij,bhwj->bhwi", output.extrinsics[:, j, :3, :3], p)
            cam += output.extrinsics[:, j, :3, 3][:, None, None]
            pix = torch.einsum("bij,bhwj->bhwi", output.intrinsics[:, j], cam)
            grid = (
                2 * pix[..., :2] / pix[..., 2:].clamp_min(1e-6) / p.new_tensor([w, h])
                - 1
            )
            z = F.grid_sample(depth[:, j : j + 1], grid, align_corners=False)
            dz = (z - cam[..., 2][:, None]) / z.clamp_min(1e-5)
            obs = F.grid_sample(images[:, j], grid, align_corners=False)
            active = (
                (grid.abs() < 0.98).all(-1)[:, None]
                & (cam[..., 2][:, None] > 0)
                & (z > 0)
                & (dz.abs() < 0.06)
            )
            if valid is not None:
                active &= (
                    F.grid_sample(
                        valid[:, j].float(), grid, mode="nearest", align_corners=False
                    )
                    > 0.5
                )
            weight = active.float() * torch.exp(-dz.abs() / 0.025)
            residual = obs - own[:, i]
            accum += torch.cat([residual, dz, residual[:, 3:4].abs()], 1) * weight
            mass += weight
        support = mass / max(1, v - 1)
        feats.append(
            torch.cat([accum / mass.clamp_min(1), support, (mass > 0.2).float()], 1)
        )
    return torch.stack(feats, 1).flatten(0, 1)


def source_color_residual(
    output, rgb, thermal, valid=None, thermal_blur=0, thermal_consensus=False
):
    """One bounded source-render residual, gathered only at visible centers.

    No scene optimization or target access. Detached residuals provide observed
    feedback; the caller explicitly controls whether its derivative is retained.
    """
    gs = output.gaussians
    b, v, h, w = output.depth.shape
    correction = gs.means.new_zeros(b, gs.means.shape[1], 4)
    if b != 1 or v < 2:
        return correction
    rendered = render_rgbt(output, output.extrinsics, output.intrinsics, h, w)
    residual = torch.cat([rgb - rendered["rgb"], thermal - rendered["thermal"]], 2)
    residual = residual.clamp(-0.15, 0.15)
    if thermal_blur:
        k = int(thermal_blur)
        if k < 1 or k % 2 != 1:
            raise ValueError("Thermal residual blur must be a positive odd kernel")
        t = residual[:, :, 3:].flatten(0, 1)
        support = torch.ones_like(t) if valid is None else valid.flatten(0, 1).float()
        smoothed = F.avg_pool2d(t * support, k, 1, k // 2) / F.avg_pool2d(
            support, k, 1, k // 2
        ).clamp_min(1e-6)
        residual = torch.cat([residual[:, :, :3], smoothed.reshape(b, v, 1, h, w)], 2)
    observed = torch.cat([rgb, thermal], 2)
    own = torch.cat(
        [gs.harmonics[0, ..., 0] * 0.28209479177387814 + 0.5, output.thermal_colors[0]],
        -1,
    )
    mass = torch.zeros_like(correction)
    thermal_square = torch.zeros_like(correction[..., 3])
    thermal_abs = torch.zeros_like(thermal_square)
    views = torch.zeros_like(thermal_square)
    points = gs.means[0].float()
    for j in range(v):
        E, K = output.extrinsics[0, j], output.intrinsics[0, j]
        cam = points @ E[:3, :3].T + E[:3, 3]
        pix = cam @ K.T
        grid = (
            2 * pix[:, :2] / pix[:, 2:].clamp_min(1e-6) / pix.new_tensor([w, h]) - 1
        ).reshape(1, 1, -1, 2)
        z = F.grid_sample(rendered["depth"][:, j : j + 1], grid, align_corners=False)[
            0, 0, 0
        ]
        alpha = F.grid_sample(rendered["alpha"][:, j], grid, align_corners=False)[
            0, 0, 0
        ]
        active = (
            (grid[0, 0].abs() < 0.98).all(-1)
            & (cam[:, 2] > 0)
            & (z > 0)
            & (alpha > 0.5)
            & ((cam[:, 2] - z).abs() < 0.03 * z)
        )
        if valid is not None:
            active &= (
                F.grid_sample(
                    valid[:, j].float(), grid, mode="nearest", align_corners=False
                )[0, 0, 0]
                > 0.5
            )
        obs = F.grid_sample(observed[:, j], grid, align_corners=False)[0, :, 0].T
        err = F.grid_sample(residual[:, j], grid, align_corners=False)[0, :, 0].T
        weight = active.float() * torch.exp(
            -(obs[:, :3] - own[:, :3]).square().mean(-1) / 0.02
        )
        tw = weight * torch.exp(-(obs[:, 3] - own[:, 3]).square() / 0.01)
        weights = torch.cat([weight[:, None].expand(-1, 3), tw[:, None]], -1)
        correction = correction + (weights * err)[None]
        mass = mass + weights[None]
        thermal_square = thermal_square + (tw * err[:, 3].square())[None]
        thermal_abs = thermal_abs + (tw * err[:, 3].abs())[None]
        views = views + (tw > 0.1).float()[None]
    result = correction / mass.clamp_min(1)
    if thermal_consensus:
        # Disagreement between independently visible source residuals should not
        # create texture in a modality whose observation is smooth.
        coherent = correction[..., 3].abs() / (thermal_abs + 1e-6)
        repeatable = correction[..., 3].square() / (
            mass[..., 3] * thermal_square + 1e-6
        )
        gate = (coherent * repeatable).clamp(0, 1) * (views >= 2).float()
        result = torch.cat([result[..., :3], result[..., 3:] * gate[..., None]], -1)
    return result
