#!/usr/bin/env python3
"""Evaluate the unified RGB-T network without historical checkpoint dependencies."""

import argparse
import html
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import ConcatDataset

from depth_anything_3.rgbt.checkpoint import load_rgbt_model
from depth_anything_3.rgbt.data import SceneDataset
from depth_anything_3.rgbt.engine import batch_to, depth_loss, panel, psnr, ssim
from depth_anything_3.rgbt.export import export_scene
from depth_anything_3.rgbt.geometry import gaussian_depth
from depth_anything_3.rgbt.renderer import render_rgbt
from depth_anything_3.rgbt.surface_consensus import surface_metrics


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights", default="checkpoints/RGBT/model.pt")
    p.add_argument("--data", default="outputs/rgbt/data_336")
    p.add_argument("--split", default="val", choices=["train", "val", "test", "all"])
    p.add_argument("--datasets", nargs="+", choices=["RGBT-Scenes", "ThermoScenes"])
    p.add_argument("--scene", help="Evaluate only this exact scene ID from the manifest")
    p.add_argument("--target-id", help="Evaluate only this exact target frame ID")
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--targets",
        type=int,
        default=6,
        help="Targets per scene; 0 selects all reserved targets",
    )
    p.add_argument("--views", nargs="+", type=int, default=[3, 6])
    p.add_argument("--export", action="store_true")
    p.add_argument("--save-renders", action="store_true", help="Save target-view RGB/T PNGs and float predictions")
    p.add_argument(
        "--no-thermal-adapters",
        action="store_true",
        help="Skip thermal feature injection into the frozen DA3 backbone",
    )
    p.add_argument(
        "--limit", type=int, help="Maximum examples per protocol, for verification"
    )
    p.add_argument("--lpips", action="store_true")
    a = p.parse_args()
    if a.targets < 0 or (a.limit is not None and a.limit < 1):
        p.error("--targets must be nonnegative and --limit must be positive")
    if any(v < 2 for v in a.views):
        p.error("At least two source views are required")
    manifest = json.loads((Path(a.data) / "manifest.json").read_text())
    scene_info = {s["id"]: s for s in manifest["scenes"]}
    if a.scene and a.scene not in scene_info:
        p.error(f"Unknown scene: {a.scene}")
    targets = (
        max(len(s["target_pool"]) for s in manifest["scenes"])
        if a.targets == 0 or a.target_id
        else a.targets
    )
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "rows.jsonl").exists() or (out / "results.json").exists():
        p.error("Use a fresh output directory to preserve previous results")
    torch.set_num_threads(4)
    cv2.setNumThreads(4)
    torch.cuda.set_device(a.device)
    model, checkpoint = load_rgbt_model(a.weights, a.device)
    del checkpoint
    if a.no_thermal_adapters:
        model.reconstruction.adapters_enabled = False
    thermal_adapters_enabled = bool(model.reconstruction.adapter_layers) and model.reconstruction.adapters_enabled
    perceptual = None
    if a.lpips:
        import lpips

        perceptual = lpips.LPIPS(net="alex").to(a.device).eval()
    results, cards, exported = {}, [], set()
    started = time.time()
    for views in a.views:
        sampling = "nearest" if views == 3 else "wide"
        protocol = f"{sampling}{views}"
        folder = out / protocol
        folder.mkdir(exist_ok=True)
        datasets = []
        for split in ["train", "val", "test"] if a.split == "all" else [a.split]:
            ds = SceneDataset(
                a.data, split, views=views, sampling=sampling, targets_per_scene=targets
            )
            if a.datasets:
                ds.items = [
                    (si, ti)
                    for si, ti in ds.items
                    if ds.scenes[si]["domain"] in a.datasets
                ]
            if a.scene or a.target_id:
                ds.items = [
                    (si, ti)
                    for si, ti in ds.items
                    if (not a.scene or ds.scenes[si]["id"] == a.scene)
                    and (
                        not a.target_id
                        or ds.scenes[si]["frames"][ti]["id"] == a.target_id
                    )
                ]
            datasets.append(ds)
        ds = ConcatDataset(datasets)
        rows = []
        for i in range(len(ds) if a.limit is None else min(a.limit, len(ds))):
            item = ds[i]
            b = batch_to(item, a.device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = model(
                    *[b[k][:, :views] for k in ["rgb", "thermal", "E", "K"]],
                    source_valid=b["valid"][:, :views],
                )
            h, w = output.depth.shape[-2:]
            pred = render_rgbt(output, b["E"][:, views:], b["K"][:, views:], h, w)
            mask = b["valid"][:, views:]
            row = {
                "id": item["id"],
                "scene": item["scene"],
                "dataset": scene_info[item["scene"]]["domain"],
                "split": scene_info[item["scene"]]["split"],
                "context_ids": item["context_ids"],
            }
            for modality in ["rgb", "thermal"]:
                image = pred[modality].clamp(0, 1)
                target = b[modality][:, views:]
                row[modality + "_psnr"] = psnr(image, target, mask)
                row[modality + "_ssim"] = float(ssim(image * mask, target * mask))
            row["gaussian_depth_log_error"] = float(
                depth_loss(gaussian_depth(output), b["depth"][:, :views])
            )
            row["alpha_mean"] = float(
                (pred["alpha"] * mask).sum() / mask.sum().clamp_min(1)
            )
            row["alpha_coverage_05"] = float(
                ((pred["alpha"] > 0.5) * mask).sum() / mask.sum().clamp_min(1)
            )
            row["surface"] = (
                surface_metrics(output, model.last_association)
                if model.last_association is not None
                else None
            )
            if perceptual is not None:
                row["rgb_lpips"] = float(
                    perceptual(
                        (pred["rgb"].clamp(0, 1) * mask).flatten(0, 1),
                        (b["rgb"][:, views:] * mask).flatten(0, 1),
                        normalize=True,
                    ).mean()
                )
            if not all(np.isfinite(v) for v in row.values() if isinstance(v, float)):
                raise ValueError(f"Non-finite metrics for {item['id']}")
            panel(
                folder / (item["id"] + ".jpg"),
                b,
                pred,
                prediction_label=(
                    "SharedRefiner (no thermal adapters)"
                    if not thermal_adapters_enabled
                    else "SharedRefiner"
                ),
            )
            render_aliases = {}
            if a.save_renders:
                stem = folder / item["id"]
                target_name = None
                if a.target_id:
                    target_name = next(
                        frame["name"]
                        for frame in scene_info[item["scene"]]["frames"]
                        if frame["id"] == item["id"]
                    )
                for modality in ("rgb", "thermal"):
                    values = pred[modality][0, 0].detach().float().clamp(0, 1).cpu().numpy()
                    pixels = (np.moveaxis(values, 0, -1) * 255 + 0.5).astype(np.uint8)
                    if pixels.shape[-1] == 1:
                        pixels = pixels[..., 0]
                    image = Image.fromarray(pixels)
                    image.save(f"{stem}_{modality}.png")
                    if target_name:
                        original = Path(target_name)
                        alias = (
                            original
                            if modality == "rgb"
                            else original.with_name(f"{original.stem}_thermal{original.suffix}")
                        )
                        save_options = (
                            {"quality": 95, "subsampling": 0}
                            if alias.suffix.lower() in (".jpg", ".jpeg")
                            else {}
                        )
                        image.save(folder / alias, **save_options)
                        render_aliases[modality] = f"{protocol}/{alias.name}"
                np.savez_compressed(
                    f"{stem}_prediction.npz",
                    **{k: v.detach().float().cpu().numpy() for k, v in pred.items()},
                )
            rel = f"{protocol}/{item['id']}.jpg"
            row["comparison"] = rel
            if a.save_renders:
                row["rgb_render"] = f"{protocol}/{item['id']}_rgb.png"
                row["thermal_render"] = f"{protocol}/{item['id']}_thermal.png"
                row["prediction_npz"] = f"{protocol}/{item['id']}_prediction.npz"
                row["named_renders"] = render_aliases
            scene_export = Path("scenes") / protocol / item["scene"].replace("/", "_")
            export_key = (protocol, item["scene"])
            if a.export and export_key not in exported:
                export_scene(output, out / scene_export, (h, w), frames=12)
                (out / scene_export / "sources.json").write_text(
                    json.dumps(
                        {
                            "target_id": item["id"],
                            "context_ids": item["context_ids"],
                            "scene": item["scene"],
                            "protocol": protocol,
                            "cameras": "COLMAP provided",
                            "optimization_steps": 0,
                        },
                        indent=2,
                    )
                    + "\n"
                )
                exported.add(export_key)
            if a.export:
                row["scene_export"] = scene_export.as_posix()
            links = (
                f'<p><a href="{scene_export}/geometry.html">三维预览</a> · '
                f'<a href="{scene_export}/rgbt_trajectory.gif">视角动画</a> · '
                f'<a href="{scene_export}/rgb.ply">RGB PLY</a> · '
                f'<a href="{scene_export}/thermal.ply">Thermal PLY</a></p>'
                if a.export
                else ""
            )
            if a.save_renders:
                links += (
                    f'<p><a href="{row["rgb_render"]}">RGB 预测图</a> · '
                    f'<a href="{row["thermal_render"]}">热成像预测图</a> · '
                    f'<a href="{row["prediction_npz"]}">浮点预测数据</a></p>'
                )
            if render_aliases:
                links += (
                    f'<p><a href="{render_aliases["rgb"]}">按目标原文件名保存的 RGB 预测</a> · '
                    f'<a href="{render_aliases["thermal"]}">热成像预测</a></p>'
                )
            cards.append(
                f"<article><h3>{html.escape(protocol + ' / ' + item['id'])} ({row['split']})</h3>"
                f"<p>RGB PSNR {row['rgb_psnr']:.2f} dB · Thermal PSNR {row['thermal_psnr']:.2f} dB</p>"
                f'{links}<img loading="lazy" src="{html.escape(rel)}"></article>'
            )
            rows.append(row)
            with (out / "rows.jsonl").open("a") as stream:
                stream.write(json.dumps({"protocol": protocol, **row}) + "\n")
            print(
                protocol,
                f"{i + 1}/{len(ds) if a.limit is None else min(a.limit, len(ds))}",
                item["id"],
                json.dumps({k: v for k, v in row.items() if isinstance(v, float)}),
                flush=True,
            )
        if not rows:
            raise ValueError(f"No evaluation examples for {protocol}")
        keys = [k for k, v in rows[0].items() if isinstance(v, float)]
        scenes = {
            s: {
                k: float(np.mean([r[k] for r in rows if r["scene"] == s])) for k in keys
            }
            for s in sorted({r["scene"] for r in rows})
        }
        results[protocol] = {
            "summary": {
                k: float(np.mean([r[k] for r in scenes.values()])) for k in keys
            },
            "per_scene": scenes,
            "per_view": rows,
            "count": len(rows),
            "scene_splits": {s: scene_info[s]["split"] for s in scenes},
            "by_split": {
                split: {
                    k: float(
                        np.mean(
                            [
                                r[k]
                                for s, r in scenes.items()
                                if scene_info[s]["split"] == split
                            ]
                        )
                    )
                    for k in keys
                }
                for split in ("train", "val", "test")
                if any(scene_info[s]["split"] == split for s in scenes)
            },
        }
    report = {
        "weights": str(Path(a.weights).resolve()),
        "data": a.data,
        "split": a.split,
        "datasets": a.datasets,
        "scene": a.scene,
        "target_id": a.target_id,
        "targets_per_scene": "all reserved targets" if a.target_id or a.targets == 0 else a.targets,
        "cameras": "Provided COLMAP source and target cameras; no per-scene optimization",
        "thermal": "Dataset grayscale/display luminance, not calibrated temperature",
        "thermal_adapters_enabled": thermal_adapters_enabled,
        "elapsed_seconds": time.time() - started,
        "aggregation": "Equal scene weights within each view protocol",
        "results": results,
    }
    (out / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    (out / "index.html").write_text(
        '<!doctype html><html lang="zh"><meta charset="utf-8"><title>SharedRefiner evaluation</title><style>body{font:16px/1.6 system-ui;max-width:1200px;margin:30px auto;padding:20px}img{max-width:100%}article{margin:24px 0}</style><h1>RGB-T SharedRefiner 评价</h1>'
        + ('<p>热适配器：模型中未启用</p>' if not thermal_adapters_enabled else '')
        + '<p>每行左侧为真值，右侧为生成结果；上行为 RGB，下行为热图。全部选定样例均展示。</p><p>使用已知 COLMAP 相机。train 场景参与过训练，val/test 场景曾用于历史评价。三维预览是高斯中心点展示；PLY 包含高斯参数。每个场景／协议的三维导出对应第一个评价目标的源图组合。</p><pre>'
        + html.escape(
            json.dumps(
                {p: r["summary"] for p, r in results.items()},
                ensure_ascii=False,
                indent=2,
            )
        )
        + "</pre>"
        + "".join(cards)
        + "</html>"
    )


if __name__ == "__main__":
    main()
