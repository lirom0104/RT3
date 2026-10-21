"""Transfer RGB boundaries while anchoring depth and cameras to a stable run.

This is a CPU postprocessor, not a newly trained model. Nearest depth sampling
and robust log-depth weights avoid averaging across occluding surfaces.
"""
import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt, maximum_filter, minimum_filter

from .detail_refinement import refine_rgb_depth


def _positive_depth(depth):
    depth = np.asarray(depth)
    if (depth.ndim != 3 or not np.issubdtype(depth.dtype, np.floating)
            or not np.isfinite(depth).all() or (depth <= 0).any()):
        raise ValueError("Expected finite positive floating VxHxW depths")
    return depth


def _log_gradient(value):
    dx = np.pad(np.abs(np.diff(value, axis=1)), ((0, 0), (0, 1)))
    dy = np.pad(np.abs(np.diff(value, axis=0)), ((0, 1), (0, 0)))
    return np.maximum(dx, dy)


def _thin_surface_cores(base_log, teacher_log, radius, max_gradient):
    """Find stable centers of narrow near surfaces, excluding background holes.

    Both sides of a teacher discontinuity define its distance transform. Medial
    ridges provide anchors even when a whole thin object lies in the boundary
    band. A locally smooth base and the near half of the teacher depth range
    keep blurred transitions and newly opened background gaps out of the cores.
    No object annotations, teacher absolute scale or camera fitting are used.
    """
    edge = _log_gradient(teacher_log) > .06
    edge[:, 1:] |= np.abs(np.diff(teacher_log, axis=1)) > .06
    edge[1:] |= np.abs(np.diff(teacher_log, axis=0)) > .06
    if radius == 0 or not edge.any():
        return np.zeros_like(edge)
    distance = distance_transform_edt(~edge)
    ridge = (distance >= 1) & (distance >= maximum_filter(distance, size=5) - .5)
    low = minimum_filter(teacher_log, size=2 * radius + 1)
    high = maximum_filter(teacher_log, size=2 * radius + 1)
    return (ridge & (teacher_log < (low + high) / 2) & (high - low > .12)
            & (_log_gradient(base_log) < max_gradient))


def thin_surface_regions(base, teacher, thermal_mask, *, radius=8,
                         core_gradient=.01, expansion=3):
    """Small near-surface neighborhoods for geometry-validated detail export."""
    base, teacher = _positive_depth(base), _positive_depth(teacher)
    thermal_mask = np.asarray(thermal_mask, dtype=bool)
    if base.shape != teacher.shape or thermal_mask.shape != (len(base),):
        raise ValueError("Depth shapes and modality mask must match")
    regions = np.zeros(base.shape, dtype=bool)
    for i in np.flatnonzero(~thermal_mask):
        p, g = np.log(base[i]), np.log(teacher[i])
        cores = _thin_surface_cores(p, g, radius, core_gradient)
        if not cores.any():
            continue
        distance, nearest = distance_transform_edt(~cores, return_indices=True)
        regions[i] = (distance <= expansion) & (np.abs(g - g[tuple(nearest)]) < .04)
    return regions


def transfer_boundaries(base, teacher, thermal_mask, *, radius=8,
                        iterations=2000, sigma=.025, anchor=1e-5,
                        thin_anchor=1., core_gradient=.01):
    """Solve a teacher-edge weighted harmonic depth in narrow boundary bands.

    Outside those bands, the stable base strongly anchors depth. The initial
    residual transfer restores isolated thin holes which diffusion alone cannot
    recover. Stable medial cores additionally anchor narrow foreground objects:
    without them an entire handlebar can retain the biased initial depth and
    disappear during multiview export. Background holes keep the weak band
    weight so their sharp opening is preserved. Set thin_anchor=0 to reproduce
    the earlier behavior. Teacher absolute depths and cameras are not used as
    geometry anchors.
    """
    base, teacher = _positive_depth(base), _positive_depth(teacher)
    thermal_mask = np.asarray(thermal_mask, dtype=bool)
    if base.shape != teacher.shape or thermal_mask.shape != (len(base),):
        raise ValueError("Depth shapes and modality mask must match")
    if (not isinstance(radius, int) or radius < 0 or not isinstance(iterations, int)
            or iterations < 1 or not np.isfinite([sigma, anchor, thin_anchor, core_gradient]).all()
            or sigma <= 0 or anchor <= 0 or thin_anchor < 0 or core_gradient <= 0):
        raise ValueError("Invalid boundary transfer parameters")
    initial, _ = refine_rgb_depth(teacher, base, thermal_mask, sigma=32.)
    output = base.copy()
    for i in np.flatnonzero(~thermal_mask):
        p, g = np.log(base[i]), np.log(teacher[i])

        band = maximum_filter(np.maximum(_log_gradient(p), _log_gradient(g)),
                              size=2 * radius + 1) > .06
        alpha = np.where(band, anchor, 1.).astype(np.float32)
        if thin_anchor:
            cores = _thin_surface_cores(p, g, radius, core_gradient) & band
            alpha[cores] = max(anchor, thin_anchor)
        wx = np.exp(-np.square(np.diff(g, axis=1) / sigma))
        wy = np.exp(-np.square(np.diff(g, axis=0) / sigma))
        denominator = alpha.copy()
        denominator[:, 1:] += wx
        denominator[:, :-1] += wx
        denominator[1:] += wy
        denominator[:-1] += wy
        value = np.log(initial[i]).copy()
        for _ in range(iterations):
            total = alpha * p
            total[:, 1:] += wx * value[:, :-1]
            total[:, :-1] += wx * value[:, 1:]
            total[1:] += wy * value[:-1]
            total[:-1] += wy * value[1:]
            value = total / denominator
        output[i] = np.exp(value)
    return _positive_depth(output)


def reconcile_depths(depth, extrinsics, intrinsics, thermal_mask, paired_views,
                     *, iterations=5, tolerance=.025, neighbors=4):
    """Consensus in one fixed world frame; also count final RGB-view support.

    A pixel is supported when another RGB view agrees within tolerance in log
    depth, including reprojection through the fixed base cameras. Thermal
    observations influence RGB depth with weight .25. Unsupported predictions
    remain in the depth arrays but can be excluded explicitly from PLY export.
    """
    depth = _positive_depth(depth).copy()
    v, h, w = depth.shape
    e, k = np.asarray(extrinsics), np.asarray(intrinsics)
    mask = np.asarray(thermal_mask, dtype=bool)
    pairs = np.asarray(paired_views)
    if (e.shape != (v, 3, 4) or k.shape != (v, 3, 3) or mask.shape != (v,)
            or pairs.shape != (v,) or not np.issubdtype(pairs.dtype, np.integer)
            or (pairs < 0).any() or (pairs >= v).any()
            or not np.isfinite(e).all() or not np.isfinite(k).all()):
        raise ValueError("Invalid cameras, modality mask or paired-view indices")
    if (not isinstance(iterations, int) or iterations < 0
            or not isinstance(neighbors, int) or not 1 <= neighbors <= 255
            or not np.isfinite(tolerance) or tolerance <= 0):
        raise ValueError("Invalid consensus parameters")
    if not (~mask).any():
        raise ValueError("Consensus requires at least one RGB view")
    y, x = np.indices((h, w), dtype=np.float32)
    pixels = np.stack((x, y, np.ones_like(x)), axis=-1)
    rays = np.array([pixels @ np.linalg.inv(kk).T for kk in k])
    centers = np.array([-ee[:3, 3] @ ee[:3, :3] for ee in e])
    distances = np.linalg.norm(centers[:, None] - centers[None, :], axis=-1)
    rgb = np.flatnonzero(~mask)
    refs = []
    for i in range(v):
        row = [int(j) for j in rgb[np.argsort(distances[i, rgb])] if j != i][:neighbors]
        if not mask[i] and mask[pairs[i]]:
            row.append(int(pairs[i]))
        refs.append(row)
    support = np.zeros(depth.shape, dtype=np.uint8)
    # Last pass only measures support at the final depths; it does not update.
    for step in range(iterations + 1):
        output = depth.copy()
        for i in range(v):
            numerator = np.log(depth[i]).copy()
            denominator = np.ones((h, w), dtype=np.float32)
            counts = np.zeros((h, w), dtype=np.uint8)
            for j in refs[i]:
                rotation = e[j, :3, :3] @ e[i, :3, :3].T
                translation = e[j, :3, 3] - rotation @ e[i, :3, 3]
                directions = rays[i] @ rotation.T
                camera = directions * depth[i, ..., None] + translation
                projected = camera @ k[j].T
                uv = (projected[..., :2] / projected[..., 2:3]).astype(np.float32)
                dj = cv2.remap(depth[j], uv[..., 0], uv[..., 1], cv2.INTER_NEAREST,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                observation = (dj - translation[2]) / np.maximum(directions[..., 2], 1e-6)
                error = np.abs(np.log(np.maximum(observation, 1e-6) / depth[i]))
                ok = ((dj > 0) & (camera[..., 2] > 0) & (observation > 0)
                      & (error < tolerance))
                if not mask[j]:
                    counts += ok
                if step == iterations:
                    continue
                weight = ok * np.square(np.maximum(0, 1 - np.square(error / tolerance)))
                if mask[j]:
                    weight *= .25
                numerator += weight * np.log(np.maximum(observation, 1e-6))
                denominator += weight
            support[i] = counts
            if step < iterations:
                output[i] = np.exp(numerator / denominator)
        if step < iterations:
            depth = output
    return _positive_depth(depth), support
