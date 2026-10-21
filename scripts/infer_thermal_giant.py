#!/usr/bin/env python3
"""Infer mixed RGB/thermal images with a trained Giant adapter.

Input JSON: [{"image": "/path/rgb.png", "thermal": false}, ...].
The output NPZ contains depth, depth confidence, OpenCV w2c extrinsics and intrinsics.
"""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch
from depth_anything_3.thermal.adapter import load_adapter, load_giant
from depth_anything_3.utils.io.input_processor import InputProcessor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--base-checkpoint")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--process-res", type=int, default=518)
    parser.add_argument("--ref-view-strategy", choices=["first", "middle", "saddle_balanced", "saddle_sim_range"], default="first")
    args = parser.parse_args()
    records = json.loads(Path(args.input).read_text())
    if not records or any(not isinstance(r.get("thermal"), bool) for r in records):
        raise ValueError("Every image record needs an explicit boolean thermal field")
    paths = [str((Path(args.input).resolve().parent / r["image"]).resolve()) for r in records]
    state = torch.load(args.adapter, map_location="cpu", weights_only=True)
    model = load_giant(args.base_checkpoint or state["base_checkpoint"])
    load_adapter(model, state)
    model.to(args.device).eval()
    images, _, _ = InputProcessor()(paths, process_res=args.process_res, sequential=True)
    if images.ndim == 4:
        images = images.unsqueeze(0)
    images = images.to(args.device)
    mask = torch.tensor([[r["thermal"] for r in records]], dtype=torch.bool, device=args.device)
    device_type = torch.device(args.device).type
    with torch.inference_mode(), torch.autocast(device_type, dtype=torch.bfloat16, enabled=device_type == "cuda"):
        output = model(images, thermal_mask=mask, ref_view_strategy=args.ref_view_strategy)
    destination = Path(args.output)
    if destination.suffix != ".npz":
        raise ValueError("Output path must end in .npz")
    destination.parent.mkdir(parents=True, exist_ok=True)
    arrays = {k: output[k][0].float().cpu().numpy() for k in ("depth", "depth_conf", "extrinsics", "intrinsics")}
    np.savez_compressed(destination, **arrays, thermal_mask=mask[0].cpu().numpy(), image_paths=np.asarray(paths))
    print(f"Saved {len(records)} views to {destination}; depth shape={arrays['depth'].shape}")


if __name__ == "__main__":
    main()
