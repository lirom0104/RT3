#!/usr/bin/env python3
"""Check that physical pruning equals its evaluated intervention, and time both."""

import argparse
import gc
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import cv2
import numpy as np
import torch
from ablate_rgbt import configured

from depth_anything_3.rgbt.checkpoint import load_rgbt_model
from depth_anything_3.rgbt.data import SceneDataset
from depth_anything_3.rgbt.engine import batch_to
from depth_anything_3.rgbt.renderer import render_rgbt


def tensors(output):
    result = {
        name: getattr(output.gaussians, name).detach().cpu()
        for name in ["means", "scales", "rotations", "harmonics", "opacities"]
    }
    for name in ["depth", "extrinsics", "intrinsics", "thermal_colors"]:
        result[name] = getattr(output, name).detach().cpu()
    return result


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--before", required=True)
    p.add_argument("--after", required=True)
    p.add_argument("--changes", required=True)
    p.add_argument("--variant", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--repeats", type=int, default=5)
    a = p.parse_args()
    torch.set_num_threads(4)
    cv2.setNumThreads(2)
    torch.cuda.set_device(a.device)
    changes = json.loads(Path(a.changes).read_text())[a.variant]
    cases = {}
    for v, known in [(3, True), (6, True), (3, False), (6, False)]:
        ds = SceneDataset(
            "outputs/rgbt/data_336",
            "val",
            views=v,
            sampling="nearest" if v == 3 else "wide",
        )
        cases[f"{v}_{'known' if known else 'predicted'}"] = (ds[0], v, known)
    reference, report = (
        {},
        {
            "before": a.before,
            "after": a.after,
            "changes": changes,
            "cases": {},
            "runtime": {},
        },
    )
    for version, path in [("before", a.before), ("after", a.after)]:
        model, checkpoint = load_rgbt_model(path, a.device)
        del checkpoint
        report[version + "_parameters"] = sum(t.numel() for t in model.parameters())
        for name, (item, v, known) in cases.items():
            b = batch_to(item, a.device)
            args = [
                b["rgb"][:, :v],
                b["thermal"][:, :v],
                b["E"][:, :v] if known else None,
                b["K"][:, :v] if known else None,
            ]
            valid = b["valid"][:, :v]

            def forward(overrides, model=model, args=args, valid=valid):
                cv2.setRNGSeed(20260920)
                with (
                    configured(model, overrides),
                    torch.autocast("cuda", dtype=torch.bfloat16),
                ):
                    return model(*args, source_valid=valid)

            output = forward(changes if version == "before" else {})
            captured = tensors(output)
            target_E = b["E"][:, v:] if known else output.extrinsics[:, :1]
            target_K = b["K"][:, v:] if known else output.intrinsics[:, :1]
            rendered = render_rgbt(output, target_E, target_K, *output.depth.shape[-2:])
            captured.update(
                {"render_" + k: value.cpu() for k, value in rendered.items()}
            )
            if version == "before":
                reference[name] = captured
            else:
                errors = {}
                for key, value in captured.items():
                    torch.testing.assert_close(
                        value, reference[name][key], rtol=3e-5, atol=3e-5
                    )
                    errors[key] = float((value - reference[name][key]).abs().max())
                report["cases"][name] = errors
            del output, rendered, captured
            # Warm source matching caches; synchronize around the entire model.
            forward({})
            times = []
            for _ in range(a.repeats):
                torch.cuda.synchronize()
                started = time.perf_counter()
                timed = forward({})
                torch.cuda.synchronize()
                times.append(time.perf_counter() - started)
                del timed
            report["runtime"].setdefault(name, {})[version] = {
                "median_seconds": float(np.median(times)),
                "seconds": times,
                "note": "Warm source feature cache, same device, complete forward, no target rendering.",
            }
            print(
                version,
                name,
                report["runtime"][name][version]["median_seconds"],
                flush=True,
            )
        del model, forward, args, valid, b
        gc.collect()
        torch.cuda.empty_cache()
    path = Path(a.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    print("VERIFIED", path)


if __name__ == "__main__":
    main()
