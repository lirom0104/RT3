#!/usr/bin/env python3
"""Independently audit exported test PNGs with NumPy/SciPy, without engine metrics.

Original float predictions were not saved. PNG-derived scores are labelled as
such; PSNR intervals account for every possible pre-quantization prediction.
Optional rerendering verifies selected cases directly at float precision.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, minimum_filter, uniform_filter


EXAMPLES = [
    "RGBT-Scenes_Building_img_001",
    "RGBT-Scenes_Building_img_169",
    "ThermoScenes_raspberrypi_frame_eval_00002",
    "ThermoScenes_raspberrypi_frame_eval_00006",
]


def psnr_numpy(pred, target, mask):
    mse = float(np.square(pred - target)[mask].mean())
    return float(-10 * np.log10(max(mse, 1e-10)))


def ssim_map(pred, target, gaussian=False):
    # HWC, channel-independent population covariance. Range is fixed to [0,1].
    def filt(x):
        if gaussian:
            return gaussian_filter(x, (1.5, 1.5, 0), truncate=3.5, mode="reflect")
        return uniform_filter(x, (7, 7, 1), mode="constant", cval=0)

    px, py = filt(pred), filt(target)
    vx, vy = filt(pred * pred) - px * px, filt(target * target) - py * py
    cov = filt(pred * target) - px * py
    return ((2 * px * py + 0.01**2) * (2 * cov + 0.03**2)) / (
        (px * px + py * py + 0.01**2) * (vx + vy + 0.03**2)
    )


def read_png(path):
    a = np.asarray(Image.open(path), dtype=np.float64) / 255
    return a[..., None] if a.ndim == 2 else a


def get_rows(root):
    return [
        json.loads(line)
        for path in sorted(root.glob("rows_*.jsonl"))
        for line in path.read_text().splitlines()
    ]


def float_recheck(args, rows, dest):
    import cv2
    import torch

    from ablate_rgbt import VARIANTS, configured
    from depth_anything_3.rgbt.checkpoint import load_rgbt_model
    from depth_anything_3.rgbt.data import SceneDataset
    from depth_anything_3.rgbt.engine import batch_to
    from depth_anything_3.rgbt.renderer import render_rgbt

    torch.set_num_threads(4)
    cv2.setNumThreads(2)
    torch.cuda.set_device(args.device)
    model, checkpoint = load_rgbt_model(args.weights, args.device)
    del checkpoint
    ds = SceneDataset(args.data, "test", views=3, targets_per_scene=100000)
    ds.items = [
        (si, ti) for si, ti in ds.items if ds.scenes[si]["frames"][ti]["id"] in EXAMPLES
    ]
    reference = {r["id"]: r for r in rows if r["protocol"] == "nearest3"}
    results = []
    with torch.no_grad():
        for i in range(len(ds)):
            item = ds[i]
            row = reference[item["id"]]
            assert item["context_ids"] == row["context_ids"]
            batch = batch_to(item, args.device)
            inputs = [batch[k][:, :3] for k in ["rgb", "thermal", "E", "K"]]
            h, w = batch["rgb"].shape[-2:]
            target_E, target_K = batch["E"][:, 3:], batch["K"][:, 3:]
            for method in ["da3", "shared"]:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    if method == "da3":
                        with configured(model, VARIANTS["da3_rgbt_wrapper"]):
                            out = model.reconstruction(*inputs, teacher=True)
                    else:
                        out = model(*inputs, source_valid=batch["valid"][:, :3])
                rendered = render_rgbt(out, target_E, target_K, h, w)
                del out
                arrays = {k: v[0, 0].float().cpu().numpy() for k, v in rendered.items()}
                arrays["valid"] = item["valid"][-1].numpy()
                for modality in ["rgb", "thermal"]:
                    arrays[modality + "_gt"] = item[modality][-1].numpy()
                    pred = (
                        np.moveaxis(arrays[modality], 0, -1)
                        .astype(np.float64)
                        .clip(0, 1)
                    )
                    gt = np.moveaxis(arrays[modality + "_gt"], 0, -1).astype(np.float64)
                    mask = arrays["valid"][0].astype(bool)
                    psnr = psnr_numpy(pred, gt, mask)
                    ssim = float(
                        ssim_map(pred * mask[..., None], gt * mask[..., None]).mean()
                    )
                    original_psnr, original_ssim = (
                        row[method][modality + "_psnr"],
                        row[method][modality + "_ssim"],
                    )
                    results.append(
                        dict(
                            target=item["id"],
                            method=method,
                            modality=modality,
                            original_psnr=original_psnr,
                            numpy_float_psnr=psnr,
                            psnr_abs_difference=abs(psnr - original_psnr),
                            original_ssim=original_ssim,
                            scipy_float_legacy_ssim=ssim,
                            ssim_abs_difference=abs(ssim - original_ssim),
                            valid_fraction=float(mask.mean()),
                        )
                    )
                np.savez_compressed(dest / f"{item['id']}_{method}.npz", **arrays)
            print("FLOAT RECHECK", item["id"], flush=True)
    (dest / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    assert max(r["psnr_abs_difference"] for r in results) < 0.002
    assert max(r["ssim_abs_difference"] for r in results) < 0.001
    return results


def write_report(dest, summary, audit):
    """Make diagnostic figures directly from real rendered float arrays."""
    comparison_rows = {
        r["id"]: r for r in get_rows(dest.parent) if r["protocol"] == "nearest3"
    }
    examples = []
    selected = [
        ("ThermoScenes_raspberrypi_frame_eval_00006", "thermal"),
        ("RGBT-Scenes_Building_img_169", "rgb"),
    ]
    for target_id, modality in selected:
        files = [
            dest / "float_recheck" / f"{target_id}_{method}.npz"
            for method in ["da3", "shared"]
        ]
        if not all(path.is_file() for path in files):
            continue
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle

        arrays = []
        for path in files:
            with np.load(path) as data:
                arrays.append({k: v for k, v in data.items()})
        gt = np.moveaxis(arrays[0][modality + "_gt"], 0, -1).astype(np.float64)
        mask = arrays[0]["valid"][0].astype(bool)
        preds = [
            np.moveaxis(a[modality], 0, -1).astype(np.float64).clip(0, 1)
            for a in arrays
        ]
        record = next(
            r
            for r in audit
            if r["target"] == target_id
            and r["protocol"] == "nearest3"
            and r["method"] == "shared"
            and r["modality"] == modality
        )
        if modality == "thermal":
            region = mask & (gt[..., 0] >= 0.5)
            yy, xx = np.where(region)
            x0, x1 = max(0, xx.min() - 6), min(gt.shape[1], xx.max() + 7)
            y0, y1 = max(0, yy.min() - 6), min(gt.shape[0], yy.max() + 7)
            region_name = "GT luminance >= 0.5"
        else:
            x0, y0 = record["worst32_x"], record["worst32_y"]
            x1, y1 = x0 + 32, y0 + 32
            region = np.zeros_like(mask)
            region[y0:y1, x0:x1] = mask[y0:y1, x0:x1]
            region_name = "Selected 32x32 patch"
        psnrs = [psnr_numpy(p, gt, mask) for p in preds]
        local_psnrs = [psnr_numpy(p, gt, region) for p in preds]
        fig, axes = plt.subplots(3, 3, figsize=(11, 12), layout="constrained")
        for col, (name, im) in enumerate(
            zip(["Ground truth", "DA3 RGB-T wrapper", "SharedRefiner"], [gt, *preds])
        ):
            shown = im[..., 0] if modality == "thermal" else im
            axes[0, col].imshow(
                shown, cmap="gray", vmin=0, vmax=1, interpolation="nearest"
            )
            axes[0, col].add_patch(
                Rectangle(
                    (x0 - 0.5, y0 - 0.5),
                    x1 - x0,
                    y1 - y0,
                    fill=False,
                    edgecolor="#00ccff",
                    linewidth=1,
                )
            )
            title = name if col == 0 else f"{name}\nFull PSNR {psnrs[col - 1]:.2f} dB"
            if col:
                method = ["da3", "shared"][col - 1]
                lpips_value = comparison_rows[target_id][method].get(
                    modality + "_lpips"
                )
                if lpips_value is not None:
                    title += f"\nFull LPIPS {lpips_value:.4f} (lower is better)"
            axes[0, col].set_title(title)
            axes[1, col].imshow(
                shown[y0:y1, x0:x1],
                cmap="gray",
                vmin=0,
                vmax=1,
                interpolation="nearest",
            )
            axes[1, col].set_title(
                "Same crop (nearest display)"
                if col == 0
                else f"Region PSNR {local_psnrs[col - 1]:.2f} dB"
            )
        axes[2, 0].imshow(region, cmap="gray", vmin=0, vmax=1)
        axes[2, 0].set_title(
            f"{region_name}\n{100 * region.sum() / mask.sum():.2f}% of valid pixels"
        )
        for col, pred in enumerate(preds, start=1):
            errors = np.abs(pred - gt).mean(axis=-1)
            mapped = axes[2, col].imshow(
                np.ma.array(errors, mask=~mask), cmap="magma", vmin=0, vmax=0.15
            )
            axes[2, col].set_title("Absolute error, fixed 0 to 0.15")
        for ax in axes.flat:
            ax.set_xticks([])
            ax.set_yticks([])
        fig.colorbar(mapped, ax=axes[2, 1:].tolist(), shrink=0.7)
        fig.suptitle(
            f"{target_id} | nearest3 | {modality.upper()}\nFloat predictions; [0,1] display; cyan rectangle = crop"
        )
        image_name = f"{target_id}_{modality}_diagnostic.png"
        fig.savefig(dest / image_name, dpi=140)
        plt.close(fig)
        examples.append(
            dict(
                target=target_id,
                modality=modality,
                shared_full_psnr=psnrs[1],
                shared_region_psnr=local_psnrs[1],
                region=region_name,
                region_fraction=float(region.sum() / mask.sum()),
                image=image_name,
            )
        )
    (dest / "examples.json").write_text(json.dumps(examples, indent=2) + "\n")
    float_check = summary.get("float_recheck")
    checks_text = (
        f"另重新渲染 4 个目标、2 种方法、2 种模态，共 {float_check['pairs']} 组浮点结果。"
        f"独立重算 PSNR 最大差异 {float_check['max_psnr_abs_difference']:.9f} dB，"
        f"原口径 SSIM 最大差异 {float_check['max_legacy_ssim_abs_difference']:.8f}。"
        if float_check
        else "本次未重新渲染浮点结果，只核验已有 PNG。"
    )
    cases = "\n".join(
        f"- **{r['target']} / {r['modality']}**：SharedRefiner 全图 {r['shared_full_psnr']:.2f} dB；"
        f"诊断区域 {r['shared_region_psnr']:.2f} dB，占有效像素 {r['region_fraction'] * 100:.2f}%。"
        f"[局部放大及误差图]({r['image']})。"
        for r in examples
    )
    text = f"""# 测试图指标核验

结论：没有发现 PSNR 公式、通道归一化或图片对应错误；高分并不说明局部细节清楚。原 SSIM 可复现，但使用自定义窗口和掩码处理，不能不注明口径就与其他实现直接比较。

## 核验范围

- 全部 {summary["groups"]} 组对照、两种方法、两种模态，共 {summary["method_modality_pairs"]} 组：PNG 预测对缓存浮点真值独立重算，不调用项目的 PSNR/SSIM 函数。
- 全部原 PSNR 都在 PNG 量化允许区间内。PNG 重算最大差异 {summary["max_png_psnr_abs_difference"]:.4f} dB，原始预测转 8 位会引入误差；这不等于原指标算错。
- 全部导出 GT 与缓存真值逐像素对应。{checks_text}
- 有效区来自去畸变后的目标图像，不依赖模型，不按预测 alpha 或误差删像素。覆盖率 {100 * summary["valid_fraction_range"][0]:.2f}%–{100 * summary["valid_fraction_range"][1]:.2f}%。
- 前一版 verification.json 主要验证图片对应和重复运行一致；同一函数重复得到相同结果本身不足以证明公式正确。本次补充了独立计算。

## 为什么看着模糊、分数仍然高

PSNR = -10 log10(MSE)，MSE 在目标有效像素和全部通道上平均，信号范围固定 [0,1]，RGB 除以 3 个通道，热图除以 1 个通道。它衡量平均亮度误差，不直接衡量锐度、物体完整性或几何细节。

{cases}

热图诊断区域固定为 GT 显示灰度 >= 0.5；它不是语义分割或温度阈值。树莓派热图的大面积背景接近黑色，背景预测接近就会降低全图 MSE，少量主体误差被平均掉。RGB 诊断区域为误差最大的完整有效 32×32 窗口，仅用于定位问题，不作为新的公平排行榜指标。两个方法使用同一区域。

两例使用重新渲染的浮点结果，放大图只使用 nearest 显示，不做锐化或每图对比度拉伸，误差热图统一 [0,0.15] 色标。热图 GT 自身已有模糊；“更锐利”但偏离 GT 的结果不一定获得更高 PSNR。输入与评估分辨率只有 252×336 或 336×252，不能据此声称原分辨率细节重建良好。

## SSIM 口径问题

原实现是 7×7 均值窗口、总体协方差、零填充，先把预测和真值的无效区都置零，再把全图 SSIM 平均。这包含人为相同的黑区，也改变掩码边界窗口；影响可能向上也可能向下。应标为 `SSIM_box7_zero_mask`，而非不加说明的通用 SSIM。

用同一 PNG 比较“原均值”与“只平均完全有效的 7×7 窗口”，平均差 {summary["mean_legacy_minus_valid_box7_ssim"]:.4f}，最大绝对差 {summary["max_abs_legacy_minus_valid_box7_ssim"]:.4f}。掩码效应在本批数据中有限，不是高 PSNR 的来源。还提供 11×11 高斯窗口（sigma=1.5、data_range=1、总体协方差、只计完整有效窗口）作为明确标注的对照；更换窗口不保证得分下降。

原图、原 CSV 和原模型均未覆盖。`per_image.csv` 提供原指标、PNG 独立复算、量化区间、有效区域比例、两种有效窗口 SSIM、亮区和局部误差；`summary.json` 提供总核验；`float_recheck/` 保存 4 个目标的实际浮点预测和独立复算结果。原始热图只是数据集显示灰度，不是实际温度误差。没有训练或挑选新权重。

复现：在项目根目录执行 `bash scripts/rgbt_python.sh scripts/audit_rgbt_comparison.py --rerender-examples`。
"""
    (dest / "REPORT.md").write_text(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path("outputs/rgbt/test_comparison_20260922")
    )
    parser.add_argument("--data", type=Path, default=Path("outputs/rgbt/data_336"))
    parser.add_argument("--rerender-examples", action="store_true")
    parser.add_argument("--weights", default="checkpoints/RGBT/model.pt")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    rows = get_rows(args.root)
    dest = args.root / "metric_audit"
    dest.mkdir(exist_ok=True)
    manifest = json.loads((args.data / "manifest.json").read_text())
    frames = {
        f["id"]: f
        for s in manifest["scenes"]
        if s["split"] == "test"
        for f in s["frames"]
    }
    audit = []
    for row in rows:
        with np.load(frames[row["id"]]["cache"]) as cache:
            mask = cache["valid"][0].astype(bool)
            for modality in ["rgb", "thermal"]:
                target = np.moveaxis(cache[modality], 0, -1).astype(np.float64)
                gt_png = read_png(args.root / row["raw"] / f"{modality}_gt.png")
                expected_gt = (target.astype(np.float32).clip(0, 1) * 255 + 0.5).astype(
                    np.uint8
                )
                assert np.array_equal(np.round(gt_png * 255), expected_gt)
                for method in ["da3", "shared"]:
                    pred = read_png(args.root / row["raw"] / f"{modality}_{method}.png")
                    # Exact admissible interval for the original clipped float prediction.
                    lo, hi = (
                        (pred - 0.5 / 255).clip(0, 1),
                        (pred + 0.5 / 255).clip(0, 1),
                    )
                    dmin = np.maximum(np.maximum(lo - target, target - hi), 0)
                    dmax = np.maximum(abs(lo - target), abs(hi - target))
                    lower = -10 * np.log10(max(float((dmax**2)[mask].mean()), 1e-10))
                    upper = -10 * np.log10(max(float((dmin**2)[mask].mean()), 1e-10))
                    old_psnr = row[method][modality + "_psnr"]
                    old_ssim = row[method][modality + "_ssim"]
                    box_map = ssim_map(pred * mask[..., None], target * mask[..., None])
                    clean7 = minimum_filter(
                        mask.astype(np.uint8), 7, mode="constant", cval=0
                    ).astype(bool)
                    clean11 = minimum_filter(
                        mask.astype(np.uint8), 11, mode="constant", cval=0
                    ).astype(bool)
                    gaussian_map = ssim_map(pred, target, gaussian=True)
                    # Bright display-luminance region, not a temperature or object mask.
                    bright = mask & (target.mean(axis=-1) >= 0.5)
                    pixel_error = ((pred - target) ** 2).mean(axis=-1)
                    count = uniform_filter(mask.astype(float), 32, mode="constant")
                    patch_mse = uniform_filter(pixel_error, 32, mode="constant")
                    worst = np.where(count > 0.99999, patch_mse, -1)
                    py, px = np.unravel_index(worst.argmax(), worst.shape)
                    psnr_png = psnr_numpy(pred, target, mask)
                    record = dict(
                        scene=row["scene"],
                        protocol=row["protocol"],
                        target=row["id"],
                        method=method,
                        modality=modality,
                        valid_fraction=float(mask.mean()),
                        original_psnr=old_psnr,
                        png_prediction_psnr=psnr_png,
                        psnr_quantization_lower=float(lower),
                        psnr_quantization_upper=float(upper),
                        original_psnr_in_quantization_interval=bool(
                            lower - 0.0001 <= old_psnr <= upper + 0.0001
                        ),
                        original_ssim=old_ssim,
                        png_legacy_ssim=float(box_map.mean()),
                        png_box7_valid_windows_ssim=float(box_map[clean7].mean()),
                        png_gaussian11_valid_windows_ssim=float(
                            gaussian_map[clean11].mean()
                        ),
                        bright_gt_fraction=float(bright.sum() / mask.sum()),
                        bright_gt_psnr=psnr_numpy(pred, target, bright)
                        if bright.any()
                        else None,
                        worst32_psnr=float(-10 * np.log10(max(worst[py, px], 1e-10))),
                        worst32_x=int(px - 16),
                        worst32_y=int(py - 16),
                        valid_rmse_255=float(np.sqrt(pixel_error[mask].mean()) * 255),
                        comparison=row["comparison"],
                    )
                    audit.append(record)
    with (dest / "per_image.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(audit[0]))
        writer.writeheader()
        writer.writerows(audit)
    summary = dict(
        groups=len(rows),
        method_modality_pairs=len(audit),
        valid_fraction_range=[
            min(r["valid_fraction"] for r in audit),
            max(r["valid_fraction"] for r in audit),
        ],
        all_psnr_within_png_quantization_interval=all(
            r["original_psnr_in_quantization_interval"] for r in audit
        ),
        max_png_psnr_abs_difference=max(
            abs(r["original_psnr"] - r["png_prediction_psnr"]) for r in audit
        ),
        max_png_legacy_ssim_abs_difference=max(
            abs(r["original_ssim"] - r["png_legacy_ssim"]) for r in audit
        ),
        mean_legacy_minus_valid_box7_ssim=float(
            np.mean(
                [r["png_legacy_ssim"] - r["png_box7_valid_windows_ssim"] for r in audit]
            )
        ),
        max_abs_legacy_minus_valid_box7_ssim=max(
            abs(r["png_legacy_ssim"] - r["png_box7_valid_windows_ssim"]) for r in audit
        ),
        notes=[
            "All PNG computations use cached float GT and PNG predictions /255, data_range=1.",
            "Legacy SSIM: 7x7 uniform, zero padding, population covariance, invalid pixels zeroed then full mean.",
            "Diagnostic SSIM: 11x11 Gaussian sigma=1.5, population covariance, only fully valid windows.",
            "Bright region: GT display luminance >=0.5; not a semantic or temperature mask.",
            "Worst32 is a diagnostic region selected by error, not a replacement benchmark.",
            "Original metrics and images remain unchanged.",
        ],
    )
    float_dest = dest / "float_recheck"
    checks = None
    if args.rerender_examples:
        float_dest.mkdir(exist_ok=True)
        checks = float_recheck(args, rows, float_dest)
    elif (float_dest / "results.json").is_file():
        checks = json.loads((float_dest / "results.json").read_text())
    if checks:
        summary["float_recheck"] = dict(
            pairs=len(checks),
            max_psnr_abs_difference=max(r["psnr_abs_difference"] for r in checks),
            max_legacy_ssim_abs_difference=max(
                r["ssim_abs_difference"] for r in checks
            ),
        )
    assert summary["all_psnr_within_png_quantization_interval"]
    (dest / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_report(dest, summary, audit)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
