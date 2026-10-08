"""Robust inverse-depth fitting along source RGB line segments.

Only independently supported depth-search samples vote for a line. No semantic
labels, scene IDs, target images or assumption of vertical world direction enter.
"""

import cv2
import numpy as np
import torch


def refine_lines(rgb, depth, confidence, support, valid):
    """Return source-only line depth hypotheses and a support mask."""
    images = rgb.detach().float().cpu().numpy()
    depths = depth.detach().float().cpu().numpy()
    conf = confidence.detach().float().cpu().numpy()
    agree = support.detach().float().cpu().numpy()
    masks = valid.detach().float().cpu().numpy()
    result = depths.copy()
    weights = np.zeros_like(depths)
    detector = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    for i, (image, z) in enumerate(zip(images, depths)):
        h, w = z.shape
        gray = (image.mean(0).clip(0, 1) * 255 + 0.5).astype(np.uint8)
        detected = detector.detect(gray)[0]
        if detected is None:
            continue
        segments = []
        for raw in detected[:, 0]:
            a, b = raw[:2].astype(float), raw[2:].astype(float)
            length = np.linalg.norm(b - a)
            if length >= 20:
                segments.append((a, b))
        # Join almost collinear fragments of the same image edge.
        merged = []
        for a, b in sorted(segments, key=lambda ab: -np.linalg.norm(ab[1] - ab[0])):
            direction = (b - a) / np.linalg.norm(b - a)
            normal = np.array([-direction[1], direction[0]])
            inserted = False
            for j, (c, d) in enumerate(merged):
                e = (d - c) / np.linalg.norm(d - c)
                if abs(direction @ e) < 0.998:
                    continue
                if (
                    max(
                        abs((a - c) @ np.array([-e[1], e[0]])),
                        abs((b - c) @ np.array([-e[1], e[0]])),
                    )
                    > 1.5
                ):
                    continue
                ts = np.array([(a - c) @ e, (b - c) @ e])
                length = np.linalg.norm(d - c)
                if ts.min() > length + 24 or ts.max() < -24:
                    continue
                merged[j] = (c + e * min(0, ts.min()), c + e * max(length, ts.max()))
                inserted = True
                break
            if not inserted:
                merged.append((a, b))
        for a, b in merged:
            length = np.linalg.norm(b - a)
            if length < 36:
                continue
            tangent = (b - a) / length
            normal = np.array([-tangent[1], tangent[0]])
            t = np.linspace(0, 1, int(length) + 1)
            for offset in [-3, -2, -1, 0, 1, 2, 3]:
                xy = a[None] + t[:, None] * (b - a)[None] + normal[None] * offset
                xy = np.rint(xy - 0.5).astype(int)
                x, y = xy.T
                inside = (x >= 2) & (x < w - 2) & (y >= 2) & (y < h - 2)
                x, y, tt = x[inside], y[inside], t[inside]
                if len(x) < 30:
                    continue
                zz = z[y, x]
                q = conf[i, y, x]
                sup = agree[i, y, x]
                good = (
                    (q > 0.30)
                    & (sup > 0.70)
                    & (masks[i, y, x] > 0.5)
                    & np.isfinite(zz)
                    & (zz > 0)
                )
                ids = np.flatnonzero(good)
                if len(ids) < 12 or np.ptp(tt[ids]) < 0.65:
                    continue
                inv = 1 / zz
                best = None
                score = 0
                choices = ids[
                    np.linspace(0, len(ids) - 1, min(20, len(ids))).astype(int)
                ]
                for ia in choices:
                    for ib in choices:
                        if tt[ib] - tt[ia] < 0.4:
                            continue
                        slope = (inv[ib] - inv[ia]) / (tt[ib] - tt[ia])
                        intercept = inv[ia] - slope * tt[ia]
                        pred = slope * tt + intercept
                        inlier = good & (abs(pred - inv) < 0.025 * inv)
                        s = q[inlier].sum()
                        if s > score:
                            score = s
                            best = inlier
                if (
                    best is None
                    or best.sum() < 12
                    or q[best].sum() < q[good].sum() * 0.65
                    or np.ptp(tt[best]) < 0.65
                ):
                    continue
                A = np.stack([tt[best], np.ones(best.sum())], 1)
                wt = q[best]
                ab = np.linalg.lstsq(A * wt[:, None], inv[best] * wt, rcond=None)[0]
                fitted = ab[0] * tt + ab[1]
                if (fitted <= 0).any():
                    continue
                proposal = 1 / fitted
                # Do not propagate across a source color discontinuity.
                color = image[:, y, x].T
                ref = np.median(color[best], axis=0)
                color_dist = np.linalg.norm(color - ref, axis=1)
                color_limit = max(
                    0.10, float(np.quantile(color_dist[best], 0.95)) + 0.03
                )
                accepted = (color_dist < color_limit) & (masks[i, y, x] > 0.5)
                reliability = float(np.clip(q[best].mean(), 0.5, 1.0))
                use = accepted & (reliability > weights[i, y, x])
                result[i, y[use], x[use]] = proposal[use]
                weights[i, y[use], x[use]] = reliability
    return torch.as_tensor(result, device=depth.device), torch.as_tensor(
        weights, device=depth.device
    )
