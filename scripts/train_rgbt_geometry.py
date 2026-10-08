#!/usr/bin/env python3
"""Train confidence/residual prediction on scene-disjoint geometry evidence."""

import argparse, json, random, time
from pathlib import Path
from functools import lru_cache
import numpy as np, torch
from cache_rgbt_joint import move
from depth_anything_3.rgbt.engine import batch_to, psnr, ssim
from depth_anything_3.rgbt.multiview_depth import MultiViewDepthCorrector
from depth_anything_3.rgbt.renderer import render_rgbt
from probe_rgbt_geometry import panel


@lru_cache(maxsize=16)
def load(path):
    return torch.load(path, map_location="cpu", weights_only=False, mmap=True)


def edge(a, b, mask):
    result = 0
    for dim in [-1, -2]:
        m = mask.narrow(dim, 0, mask.shape[dim] - 1) * mask.narrow(
            dim, 1, mask.shape[dim] - 1
        )
        result += ((a.diff(dim=dim) - b.diff(dim=dim)).abs() * m).sum() / (
            m.sum() * a.shape[2]
        ).clamp_min(1)
    return result * 0.5


def predict(model, path, evidence, device):
    data = load(str(path))
    b = batch_to(data["item"], device)
    base = move(data["baseline"], device)
    v = base.depth.shape[1]
    p = move(load(str(evidence / (path.stem + "_evidence.pt"))), device)
    out = model(b["rgb"][:, :v], base, b["valid"][:, :v], p)
    rendered = render_rgbt(out, b["E"][:, v:], b["K"][:, v:], *out.depth.shape[-2:])
    return data, b, out, rendered, p


@torch.no_grad()
def evaluate(model, paths, evidence, device, dest):
    dest.mkdir(parents=True, exist_ok=True)
    rows = []
    model.eval()
    for path in paths:
        data, b, out, r, p = predict(model, path, evidence, device)
        v = out.depth.shape[1]
        m = b["valid"][:, v:]
        row = {"id": data["item"]["id"], "scene": data["item"]["scene"], "views": v}
        for name, obj in [("old", move(data["previous"], device)), ("new", r)]:
            row[name] = {}
            for mod in ["rgb", "thermal"]:
                pred = obj[mod].clamp(0, 1)
                truth = b[mod][:, v:]
                row[name][mod + "_psnr"] = psnr(pred, truth, m)
                row[name][mod + "_ssim"] = float(ssim(pred * m, truth * m))
                row[name][mod + "_edge"] = float(edge(pred, truth, m))
        rows.append(row)
        panel(
            dest / (path.stem + ".png"),
            b,
            {"Old": move(data["previous"], device), "New": r},
        )
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
        for name in ["old", "new"]
    }
    score = (
        summary["new"]["rgb_psnr"]
        + 0.5 * summary["new"]["thermal_psnr"]
        + 10 * summary["new"]["rgb_ssim"]
    )
    (dest / "metrics.json").write_text(
        json.dumps({"summary": summary, "rows": rows}, indent=2)
    )
    print("VALIDATION", dest.name, score, summary, flush=True)
    return score, summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--evidence", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--resume", type=Path)
    a = p.parse_args()
    torch.set_num_threads(4)
    torch.cuda.set_device(a.device)
    torch.manual_seed(20260923)
    random.seed(20260923)
    a.output.mkdir(parents=True, exist_ok=True)
    train = json.loads((a.cache / "train/index.json").read_text())
    val = json.loads((a.cache / "val/index.json").read_text())
    assert not {r["scene"] for r in train} & {r["scene"] for r in val}
    (a.output / "config.json").write_text(
        json.dumps(
            {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()},
            indent=2,
        )
    )
    model = MultiViewDepthCorrector(learned=True).to(a.device)
    if a.resume:
        ck = load(str(a.resume))
        cfg = dict(ck["config"])
        cfg["learned"] = True
        model = MultiViewDepthCorrector(**cfg).to(a.device)
        model.load_state_dict(ck["state_dict"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.001)
    paths = [a.cache / "val" / r["file"] for r in val]
    best, summary = evaluate(
        model, paths, a.evidence / "val", a.device, a.output / "step_0000"
    )

    def save(name, step, score):
        torch.save(
            dict(
                config=model.config,
                state_dict={k: v.detach().cpu() for k, v in model.state_dict().items()},
                step=step,
                validation_score=score,
                train_scenes=sorted({r["scene"] for r in train}),
                val_scenes=sorted({r["scene"] for r in val}),
            ),
            a.output / name,
        )

    save("best.pt", 0, best)
    groups = {}
    for r in train:
        groups.setdefault((r["scene"], r["views"]), []).append(r)
    started = time.time()
    for step in range(1, a.steps + 1):
        model.train()
        record = random.choice(groups[random.choice(list(groups))])
        path = a.cache / "train" / record["file"]
        optimizer.zero_grad(set_to_none=True)
        data, b, out, r, p = predict(model, path, a.evidence / "train", a.device)
        assert data["split"] == "train"
        v = out.depth.shape[1]
        mask = b["valid"][:, v:]
        old = move(data["previous"], a.device)
        loss = out.regularization * 0
        for mod, weight in [("rgb", 1.0), ("thermal", 0.4)]:
            pred = r[mod].clamp(0, 1)
            truth = b[mod][:, v:]
            diff = (pred - truth).abs()
            base = (old[mod].clamp(0, 1) - truth).abs()
            photometric = (diff * mask).sum() / (mask.sum() * pred.shape[2]).clamp_min(
                1
            )
            preserve = ((diff - base - 0.01).clamp_min(0) * mask).sum() / (
                mask.sum() * pred.shape[2]
            ).clamp_min(1)
            loss += weight * (
                photometric
                + 0.20 * (1 - ssim(pred * mask, truth * mask))
                + 0.5 * edge(pred, truth, mask)
                + 2 * preserve
            )
        # Keep independent, strongly supported surface matches while learning ambiguous regions.
        strong = (
            (p["confidence"] > 0.6) & (p["support"] > 0.8) & (p["improvement"] > 0.03)
        )
        loss += (
            0.05
            * ((out.depth / p["depth"].clamp_min(1e-6)).log().abs() * strong).sum()
            / strong.sum().clamp_min(1)
        )
        loss += 0.01 * ((old["alpha"] - r["alpha"]).clamp_min(0) * mask).mean()
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite training loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % 10 == 0:
            print(
                "TRAIN",
                step,
                "loss",
                float(loss.detach()),
                "seconds",
                round(time.time() - started, 1),
                flush=True,
            )
        if step % a.eval_every == 0 or step == a.steps:
            score, summary = evaluate(
                model, paths, a.evidence / "val", a.device, a.output / f"step_{step:04}"
            )
            save("last.pt", step, score)
            if score > best:
                best = score
                save("best.pt", step, score)
            save(f"step_{step:04}.pt", step, score)


if __name__ == "__main__":
    main()
