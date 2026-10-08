#!/usr/bin/env python3
"""Feed-forward reconstruction from paired source images; no optimization at inference."""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from depth_anything_3.rgbt.checkpoint import load_rgbt_model
from depth_anything_3.rgbt.data import grayscale
from depth_anything_3.rgbt.export import export_scene
from depth_anything_3.rgbt.input_calibration import encode_radiometric, register_thermal


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights", default="checkpoints/RGBT/model.pt")
    p.add_argument("--rgb", nargs="+", required=True)
    p.add_argument("--thermal", nargs="+", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--long-side", type=int, default=336)
    p.add_argument(
        "--thermal-registration",
        help="NPZ with map_x/map_y N,Hrgb,Wrgb in native thermal pixel coordinates; optional valid mask. No warp is inferred.",
    )
    p.add_argument(
        "--radiometric",
        help="JSON with raw_scale, raw_offset, temperature_min, temperature_max and unit. Requires native single-channel thermal values; same encoding for every view.",
    )
    p.add_argument(
        "--cameras",
        help="Optional NPZ containing source world-to-camera extrinsics N,4,4 and pixel intrinsics N,3,3 for the original input images.",
    )
    p.add_argument(
        "--thermal-percentile",
        action="store_true",
        help="Normalize uint16 independent captures with fixed 1/99 percentiles; not needed for 8-bit dataset images.",
    )
    a = p.parse_args()
    torch.set_num_threads(4)
    if len(a.rgb) != len(a.thermal) or len(a.rgb) < 2:
        raise ValueError("Provide at least two paired source images.")
    rgbs = []
    thermals = []
    native = []
    validity = []
    registration = None
    if a.thermal_registration:
        with np.load(a.thermal_registration) as z:
            registration = {k: z[k].copy() for k in z.files}
        if not all(
            k in registration and len(registration[k]) == len(a.rgb)
            for k in ["map_x", "map_y"]
        ):
            raise ValueError("Registration maps must match the source view count")
    radiometric = json.loads(Path(a.radiometric).read_text()) if a.radiometric else None
    if radiometric and a.thermal_percentile:
        raise ValueError(
            "Per-image percentile normalization would destroy the calibrated temperature range"
        )
    for index, (rp, tp) in enumerate(zip(a.rgb, a.thermal)):
        with Image.open(rp) as im:
            r = np.asarray(im.convert("RGB"), np.float32) / 255
        if radiometric:
            with Image.open(tp) as im:
                t = encode_radiometric(np.asarray(im), radiometric)
        else:
            t = grayscale(tp)
        valid = np.ones(r.shape[:2], np.float32)
        if registration is not None:
            t, valid = register_thermal(t, r.shape[:2], registration, index)
        if r.shape[:2] != t.shape:
            raise ValueError("Register RGB/T to the same grid before inference.")
        native.append(r.shape[:2])
        h, w = r.shape[:2]
        # DA3 patches use 14 pixels; the shared refiner also uses a /4 grid.
        hw = [max(28, round(x * a.long_side / max(h, w) / 28) * 28) for x in (h, w)]
        if rgbs and list(rgbs[0].shape[-2:]) != hw:
            raise ValueError("All views must have the same aspect ratio.")
        if a.thermal_percentile:
            lo, hi = np.percentile(t, [1, 99])
            t = np.clip((t - lo) / max(hi - lo, 1e-6), 0, 1)
        rgbs.append(
            F.interpolate(
                torch.from_numpy(r).permute(2, 0, 1)[None],
                hw,
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )[0]
        )
        thermals.append(
            F.interpolate(
                torch.from_numpy(t)[None, None],
                hw,
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )[0]
        )
        validity.append(
            F.interpolate(torch.from_numpy(valid)[None, None], hw, mode="nearest")[0]
        )
    rgb = torch.stack(rgbs)[None].to(a.device)
    thermal = torch.stack(thermals)[None].to(a.device)
    source_valid = torch.stack(validity)[None].to(a.device)
    E = K = None
    if a.cameras:
        with np.load(a.cameras) as z:
            E = torch.tensor(z["extrinsics"], dtype=torch.float32)
            K = torch.tensor(z["intrinsics"], dtype=torch.float32)
        if E.shape != (len(a.rgb), 4, 4) or K.shape != (len(a.rgb), 3, 3):
            raise ValueError("Camera count/shape mismatch.")
        for i, (h, w) in enumerate(native):
            K[i, 0] *= hw[1] / w
            K[i, 1] *= hw[0] / h
        E, K = E[None].to(a.device), K[None].to(a.device)
    torch.cuda.set_device(a.device)
    model, _state = load_rgbt_model(a.weights, a.device)
    del _state
    torch.cuda.synchronize()
    start = time.time()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        output = model(rgb, thermal, E, K, source_valid=source_valid)
    torch.cuda.synchronize()
    seconds = time.time() - start
    export_scene(output, a.output, hw)
    if radiometric:
        low, high = (
            float(radiometric["temperature_min"]),
            float(radiometric["temperature_max"]),
        )
        np.savez_compressed(
            Path(a.output) / "thermal_values.npz",
            temperature=(
                output.thermal_colors.float().cpu().numpy() * (high - low) + low
            ),
            means=output.gaussians.means.cpu().numpy(),
        )
        Path(a.output, "radiometric.json").write_text(
            json.dumps(
                {
                    **radiometric,
                    "note": "Inverse of supplied fixed sensor encoding; clipped input values remain clipped. This model has not been validated for thermometry.",
                },
                indent=2,
            )
        )
    Path(a.output, "inference.json").write_text(
        json.dumps(
            {
                "source_rgb": a.rgb,
                "source_thermal": a.thermal,
                "weights": str(Path(a.weights).resolve()),
                "source_views": len(a.rgb),
                "size": hw,
                "forward_seconds": seconds,
                "cameras": "provided" if a.cameras else "DA3 predicted",
                "optimization_steps": 0,
                "gaussians": output.gaussians.means.shape[1],
                "thermal_registration": a.thermal_registration
                or "Assumed pre-registered paired grid",
                "radiometric": a.radiometric,
            },
            indent=2,
        )
    )
    print("EXPORTED", a.output, "forward seconds", seconds)


if __name__ == "__main__":
    main()
