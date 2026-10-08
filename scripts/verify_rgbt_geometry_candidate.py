#!/usr/bin/env python3
"""Run the candidate checkpoint on fixed cached targets and save side-by-side evidence."""

import argparse, json
from pathlib import Path
import torch, numpy as np
from cache_rgbt_joint import move
from depth_anything_3.rgbt.checkpoint import load_rgbt_model
from depth_anything_3.rgbt.engine import batch_to, psnr, ssim
from depth_anything_3.rgbt.renderer import render_rgbt
from PIL import Image, ImageDraw


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    a = p.parse_args()
    torch.set_num_threads(4)
    torch.cuda.set_device(a.device)
    a.output.mkdir(parents=True, exist_ok=True)
    model, ck = load_rgbt_model(a.weights, a.device)
    del ck
    rows = []
    for record in json.loads((a.cache / "index.json").read_text()):
        data = torch.load(
            a.cache / record["file"], map_location="cpu", weights_only=False, mmap=True
        )
        b = batch_to(data["item"], a.device)
        v = record["views"]
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            o = model(
                *[b[k][:, :v] for k in ["rgb", "thermal", "E", "K"]],
                source_valid=b["valid"][:, :v],
            )
        r = render_rgbt(o, b["E"][:, v:], b["K"][:, v:], *b["rgb"].shape[-2:])
        old = move(data["previous"], a.device)
        m = b["valid"][:, v:]
        row = {"id": record["id"], "scene": record["scene"], "views": v}
        for name, obj in [("previous", old), ("candidate", r)]:
            row[name] = {}
            for mod in ["rgb", "thermal"]:
                row[name][mod + "_psnr"] = psnr(obj[mod].clamp(0, 1), b[mod][:, v:], m)
                row[name][mod + "_ssim"] = float(
                    ssim(obj[mod].clamp(0, 1) * m, b[mod][:, v:] * m)
                )
            row[name]["holes"] = float(((obj["alpha"] < 0.5) * m).sum() / m.sum())
        h, w = b["rgb"].shape[-2:]
        canvas = Image.new("RGB", (3 * w, 2 * (h + 22)), "white")
        d = ImageDraw.Draw(canvas)
        for ri, mod in enumerate(["rgb", "thermal"]):
            for ci, (label, obj) in enumerate(
                [("GT", None), ("Previous", old), ("Candidate", r)]
            ):
                x = b[mod][0, -1] if obj is None else obj[mod][0, 0]
                x = x.detach().float().cpu().clamp(0, 1)
                x = x.expand(3, -1, -1) if x.shape[0] == 1 else x
                im = Image.fromarray(
                    (x.permute(1, 2, 0).numpy() * 255 + 0.5).astype("uint8")
                )
                d.text((ci * w + 3, ri * (h + 22) + 3), label + " " + mod, fill="black")
                canvas.paste(im, (ci * w, ri * (h + 22) + 22))
        canvas.save(a.output / (record["file"] + ".png"))
        rows.append(row)
        print(json.dumps(row), flush=True)
    (a.output / "metrics.json").write_text(json.dumps(rows, indent=2))
    groups = sorted({(r["scene"], r["views"]) for r in rows})
    summary = {
        name: {
            k: float(
                np.mean(
                    [
                        np.mean(
                            [r[name][k] for r in rows if (r["scene"], r["views"]) == g]
                        )
                        for g in groups
                    ]
                )
            )
            for k in rows[0][name]
        }
        for name in ["previous", "candidate"]
    }
    print("SUMMARY", summary, flush=True)
    (a.output / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
