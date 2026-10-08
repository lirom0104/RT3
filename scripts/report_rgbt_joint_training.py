#!/usr/bin/env python3
"""Summarize completed shared-refiner experiments without hiding scene regressions."""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    root = Path(__file__).resolve().parents[1]
    work = root / "outputs/rgbt/joint_training_20260920"
    final = work / "full30_q"
    if not (final / "index_0.html").exists():
        raise RuntimeError("Full evaluation is incomplete; no final report generated")
    result = json.loads((final / "results_0.json").read_text())
    rows = [
        json.loads(line) for line in (final / "rows_0.jsonl").read_text().splitlines()
    ]
    assert result["cases"] == len(rows) == 766 and result["scenes"] == 30
    metrics = [
        "rgb_psnr",
        "thermal_psnr",
        "rgb_ssim",
        "thermal_ssim",
        "lpips",
        "depth_log_error",
    ]
    scenes = result["per_scene"]
    with (work / "scene_comparison_q1500.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=["scene", "views"] + metrics)
        writer.writeheader()
        writer.writerows(
            {
                "scene": row["scene"],
                "views": row["views"],
                **{key: row["delta"][key] for key in metrics},
            }
            for row in scenes
        )
    descriptive = {}
    rng = np.random.default_rng(20260920)
    for views in [3, 6]:
        selected = [row for row in scenes if row["views"] == views]
        descriptive[str(views)] = {}
        for metric in metrics:
            values = np.array([row["delta"][metric] for row in selected])
            boot = values[rng.integers(0, len(values), (10000, len(values)))].mean(1)
            minimum_better = metric in ["lpips", "depth_log_error"]
            tolerance = 0.001 if "psnr" in metric else 1e-5
            regressed = values > tolerance if minimum_better else values < -tolerance
            descriptive[str(views)][metric] = {
                "mean": float(values.mean()),
                "worst": float(values.max() if minimum_better else values.min()),
                "regressed_scenes": int(regressed.sum()),
                "scene_bootstrap_95pct": np.quantile(boot, [0.025, 0.975]).tolist(),
            }
    (work / "quality_summary_q1500.json").write_text(
        json.dumps(
            {
                "interpretation": "Bootstrap over these 30 scenes is descriptive; it does not prove non-regression on unseen scenes",
                "summary": descriptive,
            },
            indent=2,
        )
        + "\n"
    )
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    names = sorted({row["scene"] for row in scenes})
    for col, views in enumerate([3, 6]):
        for axis, metric in zip(axes[:, col], ["rgb_psnr", "thermal_psnr"]):
            mapping = {
                row["scene"]: row["delta"][metric]
                for row in scenes
                if row["views"] == views
            }
            values = [mapping[name] for name in names]
            axis.barh(
                np.arange(len(names)),
                values,
                color=["#287d59" if value >= 0 else "#b44536" for value in values],
            )
            axis.set_yticks(
                np.arange(len(names)),
                [name.split("/")[-1] for name in names],
                fontsize=6,
            )
            axis.axvline(0, color="black", linewidth=0.7)
            axis.set_title(f"{views} source views / {metric}")
            axis.set_xlabel("Delta from original model (dB); positive is better")
            axis.invert_yaxis()
            axis.grid(axis="x", alpha=0.2)
    fig.suptitle("Actual single shared refiner: all 30 scenes, no scene omitted")
    fig.tight_layout()
    fig.savefig(work / "scene_deltas_q1500.png", dpi=180)
    fig.savefig(work / "scene_deltas_q1500.pdf")
    trials = []
    for folder in sorted(work.glob("run_*")):
        if not folder.is_dir() or not (folder / "config.json").exists():
            continue
        validations = [
            (path, json.loads(path.read_text())) for path in folder.glob("val_*.json")
        ]
        if not validations:
            continue
        best_path, best = max(validations, key=lambda pair: pair[1]["score"])
        trials.append(
            {
                "run": folder.name,
                "best_validation": best_path.name,
                "delta": best["summary_delta"],
                "worst_scene_psnr_delta": best["worst_scene_psnr_delta"],
                "passed": best["passed"],
            }
        )
    (work / "trial_summary_final.json").write_text(json.dumps(trials, indent=2) + "\n")
    lines = [
        "# 三个细化模块合并训练结果",
        "",
        "目标是将邻域、跨视图、表面与外观三个独立学习模块合为一套共享网络，效果保持原版即可。",
        "",
        "**结构合并和训练已经实现；当前候选尚未通过严格无退化验收。默认 checkpoint 未替换。**",
        "",
        "共享候选只有 reconstruction 和 refiner.core。core 只有一个编码器、联合注意力和一个 15 通道输出头，按三个步骤复用权重；源几何、信任判定和外观反馈在内部执行。独立导出配置和权重均不包含旧三个 refiner。",
        "",
        "细化参数从 868,996 减为 503,215（约减少 42.1%）。第一步使用共享网络中固定的原邻域参数；后两步由该共享网络及步骤条件学习。训练后首步在 14 场景 × 3/6 视图的 28 组检查中与原版完全一致；这不等于整条网络无退化。",
        "",
        "## 验证协议",
        "",
        "14 个训练场景、840 组训练样本、250 个不同训练目标。模型选择用 4 个独立验证场景；全部验证集为 70 组。广泛评价覆盖 30 场景、383 个不同保留目标、3/6 源视图共 766 组，其中训练场景保留目标 356 组、验证 70 组、测试 340 组。训练与评价目标无重叠，训练与验证/测试场景无重叠。多个候选已访问测试集，因此它们是重复诊断评价，不能宣称最终一次完全未接触的测试。",
        "",
        "教师为任务开始时的默认模型，SHA256 为 `75742d3bdeff7c934f1e7135981fdfa69a6357f4696e422383076428390783c8`。骨干与重建器冻结，学生推理不调用教师。",
        "",
        "## 完整评价",
        "",
        "以下是 run_q/step_1500.pt，按验证集选出的质量保持候选。差值为每场景先平均、再对 30 场景等权汇总。LPIPS 和深度误差越低越好。",
        "",
        "|源视图|RGB PSNR Δ|Thermal PSNR Δ|RGB SSIM Δ|Thermal SSIM Δ|LPIPS Δ|深度 log 误差 Δ|",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for views, delta in result["summary_delta"].items():
        lines.append(
            f"|{views}|" + "|".join(f"{delta[key]:+.6f}" for key in metrics) + "|"
        )
    lines += [
        "",
        "仍出现退化的场景/源视图组合：",
        "",
        "|场景|源视图|RGB PSNR Δ|Thermal PSNR Δ|",
        "|---|---:|---:|---:|",
    ]
    for row in sorted(
        scenes,
        key=lambda record: min(
            record["delta"]["rgb_psnr"], record["delta"]["thermal_psnr"]
        ),
    )[:10]:
        lines.append(
            f"|{row['scene']}|{row['views']}|{row['delta']['rgb_psnr']:+.6f}|{row['delta']['thermal_psnr']:+.6f}|"
        )
    lines += [
        "",
        "[全部对照图](full30_q/index_0.html) · [逐场景 CSV](scene_comparison_q1500.csv) · [场景差值图](scene_deltas_q1500.png) · [所有试验](trial_summary_final.json)",
        "",
        "## 产物与限制",
        "",
        "实验候选：`shared_candidate_q1500.pt`，包含冻结重建器与共享 refiner，可独立加载；它没有覆盖正式模型。源码 `src/depth_anything_3/rgbt/learned_joint.py`、`ordered_joint.py`；训练与评价命令见各 run/config.json 和评价目录 config_0.json。",
        "",
        "修正了表面关联精度、几何/外观信任顺序及缓存丢失联合渲染标记的问题。保留了数值修正前的结果并记录于实验目录；最终结论依据真实完整前向的成对评价。测试、结构审计、完整推理/缓存等价检查均单独保存。",
        "",
        "严格门槛包括逐场景/逐目标的 RGB/T PSNR、SSIM、LPIPS 以及逐场景深度误差。没有因为结果接近而放宽门槛，也没有将均值改善当成局部退化不存在。当前不能将合并版宣布为无退化正式替代。",
    ]
    predicted = work / "predicted_q1500.json"
    if predicted.exists():
        data = json.loads(predicted.read_text())
        lines += [
            "",
            f"另完成预测相机模式 {data['scenes']} 场景 / {data['cases']} 组数值和源视图诊断。未提供相机时的这些源视图分数不等于新视角精度验证。",
        ]
    (work / "REPORT.md").write_text("\n".join(lines) + "\n")
    print("REPORT", work / "REPORT.md", "passed", result["passed"])


if __name__ == "__main__":
    main()
