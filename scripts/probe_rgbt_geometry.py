#!/usr/bin/env python3
"""Render and measure the depth-corrector against fixed source/target baselines."""

import argparse, json
from pathlib import Path
import torch, numpy as np
from PIL import Image, ImageDraw
from cache_rgbt_joint import move
from depth_anything_3.rgbt.engine import batch_to, psnr, ssim
from depth_anything_3.rgbt.multiview_depth import MultiViewDepthCorrector
from depth_anything_3.rgbt.renderer import render_rgbt


def panel(path, b, methods):
    h, w = b["rgb"].shape[-2:]
    canvas = Image.new("RGB", ((len(methods) + 1) * w, 2 * (h + 25)), "white")
    d = ImageDraw.Draw(canvas)
    for row, mod in enumerate(("rgb", "thermal")):
        for col, (name, r) in enumerate([("GT", None), *methods.items()]):
            a = b[mod][0, -1] if r is None else r[mod][0, 0]
            a = a.detach().float().cpu().clamp(0, 1)
            if a.shape[0] == 1:
                a = a.expand(3, -1, -1)
            im = Image.fromarray(
                (a.permute(1, 2, 0).numpy() * 255 + 0.5).astype("uint8")
            )
            d.text((col * w, row * (h + 25) + 3), name + " " + mod, fill="black")
            canvas.paste(im, (col * w, row * (h + 25) + 25))
    canvas.save(path)


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--ids", nargs="+")
    p.add_argument("--weights", type=Path)
    p.add_argument("--prepare-only", action="store_true")
    p.add_argument("--device", default="cuda:0")
    a = p.parse_args()
    torch.set_num_threads(4)
    torch.cuda.set_device(a.device)
    a.output.mkdir(parents=True, exist_ok=True)
    model = MultiViewDepthCorrector().to(a.device).eval()
    if a.weights:
        ck = torch.load(a.weights, map_location="cpu", weights_only=False)
        model = MultiViewDepthCorrector(**ck["config"]).to(a.device).eval()
        model.load_state_dict(ck["state_dict"])
    rows = []
    paths = sorted(a.cache.glob("*.pt"))
    if a.ids:
        paths = [p for p in paths if any(s in p.stem for s in a.ids)]
    for path in paths:
        data = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        b = batch_to(data["item"], a.device)
        base = move(data["baseline"], a.device)
        v = base.depth.shape[1]
        dest = a.output / (path.stem + "_evidence.pt")
        if dest.exists():
            prepared = move(
                torch.load(dest, map_location="cpu", weights_only=False), a.device
            )
        else:
            prepared = model.prepare(b["rgb"][:, :v], base, b["valid"][:, :v])
            torch.save(move(prepared, "cpu"), dest)
        if a.prepare_only:
            print("PREPARED", path.stem, flush=True)
            continue
        output = model(b["rgb"][:, :v], base, b["valid"][:, :v], prepared)
        r = render_rgbt(output, b["E"][:, v:], b["K"][:, v:], *b["rgb"].shape[-2:])
        old = move(data["previous"], a.device)
        row = {"id": data["item"]["id"], "scene": data["item"]["scene"], "views": v}
        for name, obj in [("previous", old), ("corrected", r)]:
            row[name] = {}
            for mod in ["rgb", "thermal"]:
                row[name][mod + "_psnr"] = psnr(
                    obj[mod].clamp(0, 1), b[mod][:, v:], b["valid"][:, v:]
                )
                row[name][mod + "_ssim"] = float(
                    ssim(
                        obj[mod].clamp(0, 1) * b["valid"][:, v:],
                        b[mod][:, v:] * b["valid"][:, v:],
                    )
                )
            row[name]["holes"] = float(
                ((obj["alpha"] < 0.5) * b["valid"][:, v:]).sum()
                / b["valid"][:, v:].sum()
            )
        panel(a.output / (path.stem + ".png"), b, {"Previous": old, "Corrected": r})
        np.savez_compressed(
            a.output / (path.stem + ".npz"),
            depth=output.depth[0].cpu().numpy(),
            gate=model.last_audit["gate"][0].cpu().numpy(),
            rgb=r["rgb"][0, 0].cpu().numpy(),
            alpha=r["alpha"][0, 0].cpu().numpy(),
            means=output.gaussians.means[0].cpu().numpy(),
        )
        rows.append(row)
        print(json.dumps(row), flush=True)
    (a.output / "metrics.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
