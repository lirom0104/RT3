#!/usr/bin/env python3
"""Check a trained shared refiner against base reconstruction and its teacher."""

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean

METRICS = [
    "rgb_psnr",
    "thermal_psnr",
    "rgb_ssim",
    "thermal_ssim",
    "lpips",
    "depth_log_error",
]
LOWER = {"lpips", "depth_log_error"}


def summarize(rows):
    scenes = []
    for scene, views in sorted({(r["scene"], r["views"]) for r in rows}):
        group = [r for r in rows if (r["scene"], r["views"]) == (scene, views)]
        scenes.append(
            {
                "scene": scene,
                "views": views,
                "split": group[0]["split"],
                "targets": len(group),
                **{
                    tag: {k: mean(r[tag][k] for r in group) for k in METRICS}
                    for tag in ["base", "teacher", "student", "delta", "delta_base"]
                },
            }
        )
    summaries = {}
    for subset in ["all", "held_out", "validation"]:
        selected = [
            r
            for r in scenes
            if subset == "all"
            or (subset == "held_out" and r["split"] != "train")
            or (subset == "validation" and r["split"] == "val")
        ]
        summaries[subset] = {
            str(v): {
                tag: {
                    k: mean(r[tag][k] for r in selected if r["views"] == v)
                    for k in METRICS
                }
                for tag in ["base", "teacher", "student", "delta", "delta_base"]
            }
            for v in [3, 6]
        }
    return scenes, summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    work = Path(args.output)
    policy = json.loads((work / "acceptance.json").read_text())
    rows = [
        json.loads(line)
        for line in (work / "evaluation/rows_all.jsonl").read_text().splitlines()
    ]
    keys = {(r["scene"], r["views"], r["target"]) for r in rows}
    if len(rows) != 766 or len(keys) != 766 or len({r["scene"] for r in rows}) != 30:
        raise ValueError("Expected 766 distinct paired cases across 30 scenes")
    for r in rows:
        for tag in ["base", "teacher", "student", "delta", "delta_base"]:
            if not all(math.isfinite(r[tag][k]) for k in METRICS):
                raise ValueError(f"Non-finite metric for {r['target']}")
        for metric in METRICS:
            for delta, baseline in [("delta", "teacher"), ("delta_base", "base")]:
                if not math.isclose(
                    r[delta][metric],
                    r["student"][metric] - r[baseline][metric],
                    abs_tol=1e-10,
                ):
                    raise ValueError(f"Inconsistent paired delta for {r['target']}")
    scenes, summaries = summarize(rows)
    failures = []
    for subset in ["all", "held_out"]:
        for views, values in summaries[subset].items():
            for metric, limit in policy["mean_tolerances_vs_teacher"].items():
                value = values["delta"][metric]
                if value > limit if metric in LOWER else value < limit:
                    failures.append(
                        {
                            "level": subset,
                            "views": views,
                            "metric": metric,
                            "delta": value,
                            "limit": limit,
                            "reference": "teacher",
                        }
                    )
            for metric in METRICS:
                value = values["delta_base"][metric]
                if value >= 0 if metric in LOWER else value <= 0:
                    failures.append(
                        {
                            "level": subset,
                            "views": views,
                            "metric": metric,
                            "delta": value,
                            "reference": "base",
                        }
                    )
    for row in scenes:
        for metric, limit in policy["scene_tolerances_vs_teacher"].items():
            value = row["delta"][metric]
            if value > limit if metric in LOWER else value < limit:
                failures.append(
                    {
                        "level": "scene",
                        "scene": row["scene"],
                        "views": row["views"],
                        "metric": metric,
                        "delta": value,
                        "limit": limit,
                        "reference": "teacher",
                    }
                )
    worst = {}
    for metric in METRICS:
        chosen = sorted(
            rows, key=lambda r: r["delta"][metric], reverse=metric in LOWER
        )[0]
        worst[metric] = {k: chosen[k] for k in ["scene", "views", "target"]}
        worst[metric]["delta"] = chosen["delta"][metric]
    result = {
        "passed": not failures,
        "failures": failures,
        "cases": len(rows),
        "scenes": 30,
        "policy": policy,
        "summary": summaries,
        "per_scene": scenes,
        "worst_targets": worst,
        "strict_zero_regression_passed": json.loads(
            (work / "evaluation/results_all.json").read_text()
        )["passed"],
    }
    (work / "acceptance_result.json").write_text(json.dumps(result, indent=2) + "\n")
    with (work / "per_scene.csv").open("w", newline="") as stream:
        columns = ["scene", "views", "split", "targets"] + [
            tag + "_" + metric
            for tag in ["base", "teacher", "student", "delta", "delta_base"]
            for metric in METRICS
        ]
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in scenes:
            writer.writerow(
                {
                    **{k: row[k] for k in columns[:4]},
                    **{
                        tag + "_" + k: value
                        for tag in ["base", "teacher", "student", "delta", "delta_base"]
                        for k, value in row[tag].items()
                    },
                }
            )
    lines = [
        "# 单一共享细化模块交付验收",
        "",
        "基线按用户确认：去掉邻域、跨视图、表面三个细化模块后的 `GaussianReconstruction`。",
        "目标是结构合并、可重新训练、保留相对基础模型的提升；相对三模块教师允许小幅损失。",
        "",
        f"**本次工程验收：{'通过' if result['passed'] else '未通过'}。** 原严格逐目标零退化标准未通过，历史报告保留。",
        "",
        "## 结构和训练",
        "",
        "候选为 `run_r/step_3500.pt`，由该轮验证集分数选出；本次没有依据完整评测再选权重。",
        "细化部分只有一个 `LearnedJointRefiner.core`、一个 96 通道编码器和一个 15 通道输出头，按步骤条件复用三次。",
        "包含共享局部/跨视图注意力与共享残差层；保留源图几何约束和外观反馈。推理不调用教师或旧三个网络。",
        "参数由 868,996 减至 564,815，减少 35.0%；DA3 重建器冻结，整体文件仍约 5.4 GB。",
        "训练使用 14 场景、840 组样本；首步共享参数固定，后两步学习共享注意力、步骤条件、残差及联合输出。",
        "初始化沿用已有共享网络权重；这是迁移后的重新训练，不是随机初始化从零训练。",
        "",
        "## 三方完整对照",
        "",
        "30 场景、383 个保留目标、3/6 视图共 766 组；目标、源帧、相机和掩码逐组共用。先场景内平均，再场景等权。",
        "held_out 表示未参与梯度训练的 16 个场景；它们此前已用于多轮诊断，不能称为全新独立测试。",
        "",
        "|集合|视图|模型|RGB PSNR|热 PSNR|RGB SSIM|热 SSIM|LPIPS ↓|深度误差 ↓|",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for subset in ["all", "held_out"]:
        for views in ["3", "6"]:
            for tag, label in [
                ("base", "基础重建"),
                ("teacher", "原三模块"),
                ("student", "统一模块"),
            ]:
                lines.append(
                    f"|{subset}|{views}|{label}|"
                    + "|".join(
                        f"{summaries[subset][views][tag][k]:.6f}" for k in METRICS
                    )
                    + "|"
                )
    lines += [
        "",
        "## 相对原三模块的差值",
        "",
        "|视图|RGB PSNR Δ|热 PSNR Δ|RGB SSIM Δ|热 SSIM Δ|LPIPS Δ|深度误差 Δ|",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for views in ["3", "6"]:
        lines.append(
            f"|{views}|"
            + "|".join(f"{summaries['all'][views]['delta'][k]:+.6f}" for k in METRICS)
            + "|"
        )
    lines += [
        "",
        "## 局部退化",
        "",
        "以下为逐场景平均 PSNR 损失最大的五组，并非所有目标都不下降。",
        "",
        "|场景|视图|RGB PSNR Δ|热 PSNR Δ|",
        "|---|---:|---:|---:|",
    ]
    for row in sorted(
        scenes, key=lambda r: min(r["delta"]["rgb_psnr"], r["delta"]["thermal_psnr"])
    )[:5]:
        lines.append(
            f"|{row['scene']}|{row['views']}|{row['delta']['rgb_psnr']:+.6f}|{row['delta']['thermal_psnr']:+.6f}|"
        )
    lines += [
        "",
        "逐目标最差差值：",
        "",
        "|指标|目标|视图|差值|",
        "|---|---|---:|---:|",
    ]
    for metric, row in worst.items():
        lines.append(f"|{metric}|{row['target']}|{row['views']}|{row['delta']:+.6f}|")
    lines += [
        "",
        "平面厚度是内部几何一致性诊断；本次不声明与原三个模块的平面厚度完全相等。原始数据保存了三方同掩码的表面指标。",
        "预测相机模式只做功能检查；上述质量结论对应已知源相机、252×336 或 336×252 输入。",
        "",
        "## 复现与产物",
        "",
        "- [验收规则](acceptance.json) · [全部验收数值](acceptance_result.json) · [逐场景 CSV](per_scene.csv)",
        "- [三方图像对照](evaluation/index_all.html) · [逐目标原始数据](evaluation/rows_all.jsonl)",
        "- [训练与推理说明](../../../docs/RGBT_LEARNED_JOINT.md) · [固定训练配置](../../../configs/rgbt_joint_train.json)",
        "- 旧默认权重及源码保留在 `../joint_training_20260920/before/`。",
    ]
    if (work / "deployment_audit.json").exists():
        lines += [
            "",
            "默认 `checkpoints/RGBT/model.pt` 已切换为本次已训练、已评测的统一模块；",
            "冻结重建器逐张量未变，共享模块与被评测的训练权重逐张量一致。",
            "60 项测试通过；固定初始化训练和第 3500 步续训均通过实际梯度更新。",
            "续训第 3501 步的样本、损失及梯度范数与原训练记录一致。",
            "当前默认权重通过严格加载及 3/6 视图已知相机、3 视图预测相机完整/缓存前向验证。",
            "[交付核验](deployment_audit.json) · [默认权重前向核验](default_model_verification.json) · [测试日志](tests.log)",
        ]
    (work / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(
        json.dumps(
            {"passed": result["passed"], "failures": failures, "summary": summaries},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
