#!/usr/bin/env python3
"""Add the source-only conservative geometry corrector to a frozen checkpoint."""

import argparse, json, hashlib
from pathlib import Path
import torch
from depth_anything_3.rgbt.multiview_depth import MultiViewDepthCorrector


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", default="checkpoints/RGBT/model.pt")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--corrector", type=Path, required=True)
    a = p.parse_args()
    if Path(a.input).resolve() == a.output.resolve():
        raise ValueError(
            "Use a separate output checkpoint so the baseline remains reproducible"
        )
    ck = torch.load(a.input, map_location="cpu", weights_only=False, mmap=True)
    correction = torch.load(a.corrector, map_location="cpu", weights_only=False)
    corrector = MultiViewDepthCorrector(**correction["config"])
    corrector.load_state_dict(correction["state_dict"], strict=True)
    cfg = corrector.config
    model_cfg = dict(ck["model_config"])
    model_cfg["geometry_correction"] = cfg
    state = dict(ck["state_dict"])
    state.update(
        {"geometry_corrector." + k: v.cpu() for k, v in corrector.state_dict().items()}
    )
    metadata = dict(ck)
    metadata["model_config"] = model_cfg
    metadata["state_dict"] = state
    metadata["geometry_candidate"] = (
        "source-only conservative multi-view depth search + line consensus"
    )
    metadata["base_checkpoint"] = str(Path(a.input).resolve())
    metadata["geometry_weights"] = str(a.corrector.resolve())
    metadata["geometry_validation_score"] = correction.get("validation_score")
    metadata["base_sha256"] = sha256(a.input)
    metadata["geometry_status"] = (
        "Experimental: reduces tested artifacts but does not eliminate all ghosts or all metric regressions."
    )
    a.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = a.output.with_suffix(".tmp")
    torch.save(metadata, tmp)
    tmp.replace(a.output)
    print(a.output)
    print("sha256", sha256(a.output))
    print("geometry config", json.dumps(cfg))


if __name__ == "__main__":
    main()
