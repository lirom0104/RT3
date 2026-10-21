"""Optional coarse adapter geometry plus frozen official RGB detail."""
import cv2
import numpy as np


def refine_rgb_depth(official_depth, adapted_depth, thermal_mask, sigma=32.0):
    """Low-pass only the log-depth correction, not either complete depth map.

    Fit one teacher scale over the entire RGB sequence, without labels or per-view
    fits. Thermal depths remain byte-for-byte unchanged. Fine teacher errors and
    cross-view inconsistencies can remain; this is an explicit inference option.
    """
    teacher = np.asarray(official_depth)
    student = np.asarray(adapted_depth)
    mask = np.asarray(thermal_mask, dtype=bool)
    if teacher.shape != student.shape or student.ndim != 3 or mask.shape != (len(student),):
        raise ValueError("Expected matching VxHxW depths and a V-element modality mask")
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("Detail refinement sigma must be positive and finite")
    if not np.isfinite(teacher).all() or not np.isfinite(student).all() or (teacher <= 0).any() or (student <= 0).any():
        raise ValueError("Depth refinement requires finite positive predictions")
    rgb = ~mask
    output = student.copy()
    if not rgb.any():
        return output, {"sigma_pixels": float(sigma), "teacher_sequence_scale": 1.0, "rgb_views": 0}
    scale = float(np.exp(np.median(np.log(student[rgb]) - np.log(teacher[rgb]))))
    for index in np.flatnonzero(rgb):
        base = teacher[index] * scale
        correction = np.log(student[index]) - np.log(base)
        smooth = cv2.GaussianBlur(correction, (0, 0), sigma, borderType=cv2.BORDER_REFLECT_101)
        output[index] = base * np.exp(smooth)
    if not np.isfinite(output).all() or (output <= 0).any():
        raise FloatingPointError("Nonfinite or nonpositive refined prediction")
    assert np.array_equal(output[mask], student[mask])
    return output, {"sigma_pixels": float(sigma), "teacher_sequence_scale": scale,
                    "rgb_views": int(rgb.sum()), "thermal_depths_exactly_unchanged": True}
