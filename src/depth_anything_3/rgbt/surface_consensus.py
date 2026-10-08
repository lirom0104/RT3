"""Associate observed planar patches across sources and construct shared surfaces.

Only source geometry/images/cameras are used. Distinct patches from the same
source cannot join one component. Dense ordering is retained for diagnostics.
"""

from dataclasses import replace

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from depth_anything_3.utils.geometry import affine_inverse, mat_to_quat, quat_to_mat


def fit_plane(points):
    center = points.mean(0)
    _, _, vh = np.linalg.svd(points - center, full_matrices=False)
    normal = vh[-1]
    return normal, -normal @ center


def patches(points, camera, valid, stride=4, max_planes=6):
    _h, _w, _ = points.shape
    small = points[::stride, ::stride]
    dx = np.gradient(small, axis=1)
    dy = np.gradient(small, axis=0)
    normal = np.cross(dx, dy)
    normal /= np.maximum(np.linalg.norm(normal, axis=-1, keepdims=True), 1e-8)
    facing = ((camera - small) * normal).sum(-1)
    normal *= np.where(facing[..., None] < 0, -1, 1)
    radius = np.linalg.norm(small - camera, axis=-1)
    active = valid[::stride, ::stride] > 0.5
    active[:2] = active[-2:] = False
    active[:, :2] = active[:, -2:] = False
    active &= np.isfinite(small).all(-1) & (radius > 1e-5)
    labels = np.full(small.shape[:2], -1, np.int32)
    result = []
    for plane_id in range(max_planes):
        ids = np.flatnonzero(active)
        if len(ids) < 100:
            break
        xyz = small.reshape(-1, 3)[ids]
        nn = normal.reshape(-1, 3)[ids]
        rr = radius.reshape(-1)[ids]
        choices = np.linspace(0, len(ids) - 1, min(192, len(ids))).astype(int)
        ns = nn[choices]
        ds = -(ns * xyz[choices]).sum(-1)
        distance = np.abs(xyz @ ns.T + ds) / rr[:, None]
        score = ((distance < 0.008) & (nn @ ns.T > 0.92)).sum(0)
        best = score.argmax()
        if score[best] < max(100, 0.025 * active.size):
            break
        n, d = ns[best], ds[best]
        for _ in range(3):
            keep = (np.abs(xyz @ n + d) / rr < 0.014) & (nn @ n > 0.9)
            if keep.sum() < 80:
                break
            n, d = fit_plane(xyz[keep])
            if n @ (camera - xyz[keep].mean(0)) < 0:
                n, d = -n, -d
        keep = (np.abs(xyz @ n + d) / rr < 0.016) & (nn @ n > 0.88)
        if keep.sum() < 80:
            break
        mask = np.zeros_like(active, np.uint8)
        mask.reshape(-1)[ids[keep]] = 1
        count, components, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        for component in range(1, count):
            if stats[component, cv2.CC_STAT_AREA] < 24:
                mask[components == component] = 0
        active[mask > 0] = False
        if mask.sum() < 80:
            continue
        labels[mask > 0] = len(result)
        xyz = small[mask > 0]
        result.append(
            {
                "normal": n,
                "offset": float(d),
                "center": xyz.mean(0),
                "radius": float(np.median(np.linalg.norm(xyz - camera, axis=-1))),
                "count": len(xyz),
                "points": xyz[:: max(1, len(xyz) // 400)],
            }
        )
    return labels, result


@torch.no_grad()
def build_consensus(output, rgb, source_valid=None, max_offset=0.12):
    b, views, h, w = output.depth.shape
    if b != 1:
        raise ValueError("Surface association currently requires batch size one")
    xyz = output.gaussians.means.float().reshape(views, h, w, 3)
    E, K = output.extrinsics[0].float(), output.intrinsics[0].float()
    cameras = affine_inverse(E)[:, :3, 3]
    pts, cams, es, ks = [x.cpu().numpy() for x in [xyz, cameras, E, K]]
    valid = (
        np.ones((views, h, w), np.float32)
        if source_valid is None
        else source_valid[0, :, 0].float().cpu().numpy()
    )
    labels, planes = [], []
    for v in range(views):
        lab, found = patches(pts[v], cams[v], valid[v])
        labels.append(lab)
        for local, plane in enumerate(found):
            plane.update(view=v, local=local)
            planes.append(plane)
    edges = []
    for i, a in enumerate(planes):
        for j in range(i + 1, len(planes)):
            c = planes[j]
            if a["view"] == c["view"] or a["normal"] @ c["normal"] < 0.96:
                continue
            separation = max(
                abs(a["center"] @ c["normal"] + c["offset"]),
                abs(c["center"] @ a["normal"] + a["offset"]),
            )
            if separation > max_offset * min(a["radius"], c["radius"]):
                continue
            overlap = []
            for p, q in [(a, c), (c, a)]:
                v = q["view"]
                cam = p["points"] @ es[v, :3, :3].T + es[v, :3, 3]
                pix = cam @ ks[v].T
                uv = np.rint((pix[:, :2] / np.maximum(pix[:, 2:], 1e-6)) / 4).astype(
                    int
                )
                lab = labels[v]
                inside = (
                    (uv[:, 0] >= 0)
                    & (uv[:, 0] < lab.shape[1])
                    & (uv[:, 1] >= 0)
                    & (uv[:, 1] < lab.shape[0])
                    & (cam[:, 2] > 0)
                )
                ok = np.zeros(len(cam), bool)
                ok[inside] = lab[uv[inside, 1], uv[inside, 0]] == q["local"]
                overlap.append(ok.mean())
            if min(overlap) < 0.12 or max(overlap) < 0.3:
                continue
            edges.append((sum(overlap), i, j))
    groups = [{i} for i in range(len(planes))]
    for _, i, j in sorted(edges, reverse=True):
        a = next(g for g in groups if i in g)
        c = next(g for g in groups if j in g)
        if a is c or {planes[k]["view"] for k in a} & {planes[k]["view"] for k in c}:
            continue
        # Every pair must agree in orientation and offset, avoiding chain drift.
        if any(
            planes[k]["normal"] @ planes[l]["normal"] < 0.96
            or abs(planes[k]["center"] @ planes[l]["normal"] + planes[l]["offset"])
            > max_offset * min(planes[k]["radius"], planes[l]["radius"])
            for k in a
            for l in c
        ):
            continue
        a.update(c)
        groups.remove(c)
    dense_ids = torch.full((views, h, w), -1, device=xyz.device, dtype=torch.long)
    normals, offsets, descriptions = [], [], []
    for group in groups:
        if len(group) < 2:
            continue
        members = [planes[k] for k in sorted(group)]
        weights = np.sqrt([p["count"] for p in members])
        weights /= weights.sum()
        n = sum(p["normal"] * weight for p, weight in zip(members, weights))
        n /= np.linalg.norm(n)
        d = -sum((p["center"] @ n) * weight for p, weight in zip(members, weights))
        gid = len(normals)
        for p in members:
            v = p["view"]
            mask = torch.from_numpy((labels[v] == p["local"]).astype(np.float32)).to(
                xyz.device
            )[None, None]
            mask = F.interpolate(mask, (h, w), mode="nearest")[0, 0] > 0.5
            local_error = (
                xyz[v] @ torch.as_tensor(p["normal"], device=xyz.device) + p["offset"]
            ).abs()
            mask &= local_error < 0.02 * (xyz[v] - cameras[v]).norm(dim=-1)
            dense_ids[v][mask] = gid
        normals.append(n)
        offsets.append(d)
        descriptions.append(
            {
                "views": [p["view"] for p in members],
                "counts": [p["count"] for p in members],
                "original_offsets": [float(-p["center"] @ n) for p in members],
                "radius": float(np.median([p["radius"] for p in members])),
            }
        )
    return {
        "ids": dense_ids[None],
        "normals": torch.as_tensor(
            np.array(normals).reshape(-1, 3), device=xyz.device, dtype=torch.float32
        ),
        "offsets": torch.as_tensor(offsets, device=xyz.device, dtype=torch.float32),
        "groups": descriptions,
    }


def apply_consensus(output, association, strength=1.0, flatten=0.0):
    """Project along each original source ray, preserving source pixel alignment."""
    gs = output.gaussians
    b, _v, h, w = output.depth.shape
    ids = association["ids"].reshape(b, -1)
    if not len(association["normals"]) or not (ids >= 0).any():
        return output
    n = association["normals"][ids.clamp_min(0)]
    d = association["offsets"][ids.clamp_min(0)]
    centers = (
        affine_inverse(output.extrinsics)[..., :3, 3][:, :, None, None]
        .expand(-1, -1, h, w, -1)
        .reshape(b, -1, 3)
    )
    ray = gs.means.float() - centers
    denom = (n * ray).sum(-1)
    ratio = -(d + (n * centers).sum(-1)) / torch.where(
        denom.abs() > 1e-8, denom, torch.ones_like(denom)
    )
    valid = (ids >= 0) & (ratio > 0.75) & (ratio < 1.25) & (denom.abs() > 1e-5)
    ratio = torch.where(valid, 1 + strength * (ratio - 1), torch.ones_like(ratio))
    means = centers + ray * ratio[..., None]
    scales = gs.scales * ratio[..., None]
    rotations = gs.rotations
    if flatten:
        seed = torch.zeros_like(n)
        seed[..., 0] = 1
        seed = torch.where((n[..., 0].abs() > 0.9)[..., None], seed.roll(1, -1), seed)
        u = F.normalize(torch.linalg.cross(seed, n), dim=-1)
        vv = torch.linalg.cross(n, u)
        rotation = torch.stack([u, vv, n], -1)
        quat = mat_to_quat(rotation)[..., [3, 0, 1, 2]]
        old_r = quat_to_mat(gs.rotations[..., [1, 2, 3, 0]])
        cov = old_r @ torch.diag_embed(scales.square()) @ old_r.transpose(-1, -2)
        tangent = (
            torch.stack(
                [
                    (u * (cov @ u[..., None])[..., 0]).sum(-1),
                    (vv * (cov @ vv[..., None])[..., 0]).sum(-1),
                ],
                -1,
            )
            .clamp_min(1e-12)
            .sqrt()
        )
        thin = scales.amin(-1) * (1 - flatten + 0.08 * flatten)
        scales = torch.where(
            valid[..., None], torch.cat([tangent, thin[..., None]], -1), scales
        )
        rotations = torch.where(valid[..., None], quat, rotations)
    return replace(
        output, gaussians=replace(gs, means=means, scales=scales, rotations=rotations)
    )


def surface_metrics(output, association):
    ids = association["ids"].reshape(-1)
    points = output.gaussians.means.detach().float().reshape(-1, 3)
    views, h, w = output.depth.shape[1:]
    source = torch.arange(len(points), device=points.device) // (h * w)
    opacity = output.gaussians.opacities.detach().float().reshape(-1)
    rows = []
    for i, group in enumerate(association["groups"]):
        residual = (
            points[ids == i] @ association["normals"][i] + association["offsets"][i]
        )
        if not len(residual):
            continue
        medians = [
            float(residual[source[ids == i] == v].median() / group["radius"])
            for v in range(views)
            if ((ids == i) & (source == v)).any()
        ]
        rotations = quat_to_mat(
            output.gaussians.rotations.detach()
            .float()
            .reshape(-1, 4)[ids == i][:, [1, 2, 3, 0]]
        )
        scales = output.gaussians.scales.detach().float().reshape(-1, 3)[ids == i]
        axis = torch.einsum("j,njk->nk", association["normals"][i], rotations)
        sigma = ((axis * scales).square().sum(-1)).sqrt() / group["radius"]
        opacity_keep = opacity[ids == i] > 0.1
        visible_residual = residual[opacity_keep]
        rows.append(
            {
                "views": group["views"],
                "count": len(residual),
                "normalized_thickness": float(
                    (residual.quantile(0.95) - residual.quantile(0.05))
                    / group["radius"]
                ),
                "normalized_plane_error": float(
                    residual.abs().mean() / group["radius"]
                ),
                "source_median_offsets": medians,
                "source_median_spread": max(medians) - min(medians),
                "normal_sigma_p90": float(sigma.quantile(0.9)),
                "opacity_above_01_fraction": float(opacity_keep.float().mean()),
                "opacity_above_01_thickness": float(
                    (visible_residual.quantile(0.95) - visible_residual.quantile(0.05))
                    / group["radius"]
                )
                if len(visible_residual)
                else None,
            }
        )
    return {
        "groups": rows,
        "coverage": float((ids >= 0).float().mean()),
        "thickness": float(np.mean([r["normalized_thickness"] for r in rows]))
        if rows
        else 0.0,
        "total_gaussians": len(points),
        "assigned_gaussians": int((ids >= 0).sum()),
        "note": "Same source-derived masks for both versions. Thickness is internal consistency, not physical accuracy. Opacity threshold is a diagnostic, not visibility contribution.",
    }


@torch.no_grad()
def anchor_consensus(output, rgb, association, observations=None):
    """Estimate plane position from triangulated source correspondences only."""
    from .calibration import triangulate_sources

    obs = (
        triangulate_sources(rgb[0], output.extrinsics[0], output.intrinsics[0])
        if observations is None
        else observations[0]
    )
    E = output.extrinsics[0].float()
    K = output.intrinsics[0].float()
    c2w = affine_inverse(E)
    ids = association["ids"][0]
    _v, h, w = ids.shape
    points = []
    groups = []
    for i, data in enumerate(obs):
        if not len(data):
            continue
        p = torch.from_numpy(data).to(E)
        uv = p[:, :2].long()
        inside = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
        p = p[inside]
        uv = uv[inside]
        g = ids[i, uv[:, 1], uv[:, 0]]
        cam = (
            torch.cat([p[:, :2], torch.ones_like(p[:, :1])], -1) @ K[i].inverse().T
        ) * p[:, 2:]
        points.append(cam @ c2w[i, :3, :3].T + c2w[i, :3, 3])
        groups.append(g)
    if not points:
        return association
    points = torch.cat(points).cpu().numpy()
    gids = torch.cat(groups).cpu().numpy()
    nn = association["normals"].cpu().numpy().copy()
    dd = association["offsets"].cpu().numpy().copy()
    audit = []
    for i, group in enumerate(association["groups"]):
        pts = points[gids == i]
        radius = group["radius"]
        record = {"count": len(pts), "accepted": False}
        if len(pts) >= 24:
            pts = pts[np.abs(pts @ nn[i] + dd[i]) < 0.12 * radius]
            if len(pts) >= 24:
                n, d = fit_plane(pts)
                if n @ nn[i] < 0:
                    n, d = -n, -d
                for _ in range(3):
                    error = np.abs(pts @ n + d)
                    keep = error < max(0.004 * radius, float(np.median(error)) * 2)
                    if keep.sum() < 20:
                        break
                    n, d = fit_plane(pts[keep])
                    if n @ nn[i] < 0:
                        n, d = -n, -d
                spread = np.linalg.svd(
                    pts[keep] - pts[keep].mean(0), compute_uv=False
                ) / np.sqrt(keep.sum())
                if (
                    keep.sum() >= 20
                    and n @ nn[i] > 0.95
                    and spread[1] > 0.025 * radius
                    and spread[2] < 0.006 * radius
                ):
                    nn[i], dd[i] = n, d
                    record["accepted"] = True
                    record["plane_rms"] = float(spread[2] / radius)
        audit.append(record)
    return {
        **association,
        "normals": torch.as_tensor(nn, device=E.device),
        "offsets": torch.as_tensor(dd, device=E.device),
        "anchor_audit": audit,
    }


@torch.no_grad()
def plane_covariance(output, association, thickness=0.08):
    """Analytic thin covariance preserving source-camera projected ellipses.

    Eigenvectors at repeated eigenvalues do not have stable derivatives. This
    covariance initialization is detached; shared centers retain their original
    gradient graph and learned appearance can still update the resulting scales.
    """
    gs = output.gaussians
    b, _v, h, w = output.depth.shape
    ids = association["ids"].reshape(b, -1)
    if not len(association["normals"]):
        return output
    normal = association["normals"][ids.clamp_min(0)]
    centers = (
        affine_inverse(output.extrinsics)[..., :3, 3][:, :, None, None]
        .expand(-1, -1, h, w, -1)
        .reshape(b, -1, 3)
    )
    ray = gs.means - centers
    dot = (ray * normal).sum(-1)
    jac = (
        torch.eye(3, device=ray.device)[None, None]
        - ray[..., None]
        * normal[..., None, :]
        / torch.where(dot.abs() > 1e-7, dot, torch.ones_like(dot))[..., None, None]
    )
    rotation = quat_to_mat(gs.rotations[..., [1, 2, 3, 0]])
    cov = rotation @ torch.diag_embed(gs.scales.square()) @ rotation.transpose(-1, -2)
    cov = jac @ cov @ jac.transpose(-1, -2)
    cov = (
        cov
        + (gs.scales.amin(-1) * thickness).square()[..., None, None]
        * normal[..., None]
        * normal[..., None, :]
    )
    eigen = [torch.linalg.eigh(x) for x in cov.reshape(-1, 3, 3).split(16384)]
    values = torch.cat([x[0] for x in eigen]).reshape(b, -1, 3)
    vectors = torch.cat([x[1] for x in eigen]).reshape(b, -1, 3, 3)
    values = values.clamp_min(1e-12)
    vectors = vectors.clone()
    vectors[..., :, 0] *= torch.linalg.det(vectors)[..., None]
    quat = mat_to_quat(vectors)[..., [3, 0, 1, 2]]
    valid = (
        (ids >= 0)
        & (dot.abs() > 1e-5)
        & (values[..., -1].sqrt() < gs.scales.amax(-1) * 5)
    )
    return replace(
        output,
        gaussians=replace(
            gs,
            scales=torch.where(valid[..., None], values.sqrt(), gs.scales),
            rotations=torch.where(valid[..., None], quat, gs.rotations),
        ),
    )


def consensus_appearance(
    output, rgb, thermal, association, thermal_blend=0.5, rgb_blend=0.0
):
    """Fuse observations only when projected pixels belong to the same surface."""
    gs = output.gaussians
    b, v, h, w = output.depth.shape
    if b != 1 or not len(association["normals"]):
        return output
    points = gs.means[0].float()
    ids = association["ids"][0].reshape(-1)
    E = output.extrinsics[0].float()
    K = output.intrinsics[0].float()
    own = torch.cat(
        [gs.harmonics[0, ..., 0] * 0.28209479177387814 + 0.5, output.thermal_colors[0]],
        -1,
    )
    total = own.clone()
    mass = torch.ones_like(own[:, :1])
    images = torch.cat([rgb, thermal], 2)[0].float()
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
            align_corners=False,
            mode="nearest",
        )[0, 0, 0].long()
        weight = (
            (grid[0, 0].abs() < 0.98).all(-1)
            & (labels == ids)
            & (ids >= 0)
            & (origin != j)
            & (cam[:, 2] > 0)
        ).float()
        # Reject large RGB disagreement, which suggests a bad match or occlusion.
        weight = weight * torch.exp(
            -((sampled[:, :3] - own[:, :3]).square().mean(-1)) / 0.02
        )
        total = total + weight[:, None] * sampled
        mass = mass + weight[:, None]
    average = total / mass
    t = output.thermal_colors.lerp(average[None, :, 3:], thermal_blend)
    sh = gs.harmonics.clone()
    sh[..., 0] = (
        sh[..., 0]
        + rgb_blend * (average[None, :, :3] - own[None, :, :3]) / 0.28209479177387814
    )
    return replace(output, gaussians=replace(gs, harmonics=sh), thermal_colors=t)
