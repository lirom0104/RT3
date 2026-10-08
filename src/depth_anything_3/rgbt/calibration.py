"""Source-image-only triangulation to anchor predicted depth to supplied cameras.

No COLMAP points, target pixels, or target depth enter this module. Features are
cached by image contents; camera-dependent triangulation is recomputed per call.
"""

import hashlib
from collections import OrderedDict
from dataclasses import replace

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from depth_anything_3.utils.geometry import affine_inverse

from .geometry import gaussian_depth

_FEATURES = OrderedDict()


def image_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    key = hashlib.sha256(gray.tobytes()).hexdigest()
    if key not in _FEATURES:
        sift = cv2.SIFT_create(nfeatures=2400, contrastThreshold=0.02)
        kp, desc = sift.detectAndCompute(gray, None)
        xy = np.array([p.pt for p in kp], np.float32).reshape(-1, 2) + 0.5
        _FEATURES[key] = (xy, desc)
        if len(_FEATURES) > 512:
            _FEATURES.popitem(last=False)
    _FEATURES.move_to_end(key)
    return _FEATURES[key]


def triangulate_sources(rgb, extrinsics, intrinsics):
    """Return per-view (pixel coordinates, camera-z), with mutual/reprojection gates."""
    images = (
        rgb.detach().float().cpu().permute(0, 2, 3, 1).numpy().clip(0, 1) * 255
    ).astype(np.uint8)
    E = extrinsics.detach().float().cpu().numpy()
    K = intrinsics.detach().float().cpu().numpy()
    centers = np.linalg.inv(E)[:, :3, 3]
    features = [image_features(im) for im in images]
    observations = [[] for _ in images]
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    for i in range(len(images)):
        xi, di = features[i]
        if di is None or len(di) < 8:
            continue
        for j in range(i + 1, len(images)):
            xj, dj = features[j]
            if dj is None or len(dj) < 8:
                continue
            reverse = {
                m.queryIdx: m.trainIdx
                for pair in matcher.knnMatch(dj, di, k=2)
                if len(pair) == 2
                for m, n in [pair]
                if m.distance < 0.75 * n.distance
            }
            matches = [
                (m.queryIdx, m.trainIdx)
                for pair in matcher.knnMatch(di, dj, k=2)
                if len(pair) == 2
                for m, n in [pair]
                if m.distance < 0.75 * n.distance
                and reverse.get(m.trainIdx) == m.queryIdx
            ]
            if len(matches) < 8:
                continue
            ids = np.asarray(matches)
            a, b = xi[ids[:, 0]], xj[ids[:, 1]]
            X = cv2.triangulatePoints(K[i] @ E[i, :3], K[j] @ E[j, :3], a.T, b.T).T
            ok = np.abs(X[:, 3]) > 1e-8
            X = X[:, :3] / np.where(ok, X[:, 3], 1)[:, None]
            ci, cj = X @ E[i, :3, :3].T + E[i, :3, 3], X @ E[j, :3, :3].T + E[j, :3, 3]
            pi, pj = ci @ K[i].T, cj @ K[j].T
            pi, pj = (
                pi[:, :2] / np.maximum(pi[:, 2:], 1e-8),
                pj[:, :2] / np.maximum(pj[:, 2:], 1e-8),
            )
            ri, rj = X - centers[i], X - centers[j]
            cosine = (ri * rj).sum(-1) / np.maximum(
                np.linalg.norm(ri, axis=-1) * np.linalg.norm(rj, axis=-1), 1e-8
            )
            ok &= (ci[:, 2] > 0) & (cj[:, 2] > 0) & (cosine < np.cos(np.deg2rad(0.5)))
            ok &= (np.linalg.norm(pi - a, axis=-1) < 0.75) & (
                np.linalg.norm(pj - b, axis=-1) < 0.75
            )
            for view, coords, depth in [(i, a, ci[:, 2]), (j, b, cj[:, 2])]:
                observations[view].append(
                    np.concatenate([coords[ok], depth[ok, None]], -1)
                )
    return [
        np.concatenate(parts) if parts else np.zeros((0, 3), np.float32)
        for parts in observations
    ]


@torch.no_grad()
def calibrate_output(output, rgb, min_matches=20, max_log_mad=0.12, observations=None):
    """Robust per-camera depth-scale correction; insufficient evidence is identity."""
    z = gaussian_depth(output)
    batch, views, h, w = z.shape
    ratios = torch.ones(batch, views, device=z.device)
    counts = torch.zeros_like(ratios)
    spread = torch.full_like(ratios, float("inf"))
    for b in range(batch):
        obs = (
            triangulate_sources(rgb[b], output.extrinsics[b], output.intrinsics[b])
            if observations is None
            else observations[b]
        )
        for v, data in enumerate(obs):
            if len(data) < min_matches:
                continue
            p = torch.from_numpy(data).to(z)
            grid = (2 * p[:, :2] / p.new_tensor([w, h]) - 1).reshape(1, 1, -1, 2)
            pred = F.grid_sample(
                z[b : b + 1, v : v + 1], grid, align_corners=False
            ).flatten()
            lr = (p[:, 2].clamp_min(1e-6) / pred.clamp_min(1e-6)).log()
            median = lr.median()
            mad = (lr - median).abs().median()
            keep = (lr - median).abs() < max(0.06, float(mad) * 3)
            counts[b, v], spread[b, v] = keep.sum(), mad
            if keep.sum() >= min_matches and mad < max_log_mad and median.abs() < 0.7:
                # Repeated matches at a single image corner are not reliable anchors.
                area = (p[keep, :2].amax(0) - p[keep, :2].amin(0)) / p.new_tensor(
                    [w, h]
                )
                if area.prod() > 0.08:
                    ratios[b, v] = lr[keep].median().exp()
    centers = affine_inverse(output.extrinsics)[..., :3, 3][:, :, None, None]
    points = output.gaussians.means.reshape(batch, views, h, w, 3)
    means = centers + (points - centers) * ratios[:, :, None, None, None]
    scales = (
        output.gaussians.scales.reshape(batch, views, h, w, 3)
        * ratios[:, :, None, None, None]
    )
    gs = replace(
        output.gaussians,
        means=means.reshape(batch, -1, 3),
        scales=scales.reshape(batch, -1, 3),
    )
    calibrated = replace(
        output, gaussians=gs, depth=output.depth * ratios[:, :, None, None]
    )
    return calibrated, {
        "ratios": ratios.cpu().tolist(),
        "inliers": counts.cpu().tolist(),
        "log_mad": spread.cpu().tolist(),
    }
