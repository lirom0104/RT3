"""Explicit thermal-to-RGB registration and user-supplied radiometric encoding."""

import cv2
import numpy as np


def register_thermal(thermal, rgb_shape, maps, index):
    """Sample native thermal at supplied RGB-grid coordinates; never infer a warp."""
    mx, my = maps["map_x"][index], maps["map_y"][index]
    if mx.shape != tuple(rgb_shape) or my.shape != tuple(rgb_shape):
        raise ValueError("Registration maps must use the original RGB image grid")
    if not np.isfinite(mx).all() or not np.isfinite(my).all():
        raise ValueError("Registration maps contain nonfinite coordinates")
    h, w = thermal.shape
    valid = (mx >= 0) & (my >= 0) & (mx <= w - 1) & (my <= h - 1)
    if "valid" in maps:
        valid &= maps["valid"][index] > 0
    if valid.mean() < 0.1:
        raise ValueError("Thermal registration has less than 10% valid coverage")
    result = cv2.remap(
        thermal.astype(np.float32),
        mx.astype(np.float32),
        my.astype(np.float32),
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )
    return result, valid.astype(np.float32)


def encode_radiometric(raw, specification):
    """Map physical sensor values into one fixed scene-wide training intensity range."""
    if raw.ndim != 2 or not np.issubdtype(raw.dtype, np.number):
        raise ValueError(
            "Radiometric input must be a single-channel numeric sensor image"
        )
    required = ["raw_scale", "raw_offset", "temperature_min", "temperature_max", "unit"]
    if any(k not in specification for k in required):
        raise ValueError("Radiometric JSON requires " + ", ".join(required))
    low, high = (
        float(specification["temperature_min"]),
        float(specification["temperature_max"]),
    )
    scale, offset = (
        float(specification["raw_scale"]),
        float(specification["raw_offset"]),
    )
    if not np.isfinite([low, high, scale, offset]).all() or high <= low or scale == 0:
        raise ValueError("Invalid radiometric calibration range")
    temperature = raw.astype(np.float32) * scale + offset
    if not np.isfinite(temperature).all():
        raise ValueError("Nonfinite sensor values")
    return np.clip((temperature - low) / (high - low), 0, 1)
