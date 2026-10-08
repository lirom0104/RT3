#!/usr/bin/env python3
"""Build the ablation report and a scene/target browser from completed results."""

import argparse
import json
from pathlib import Path

LABELS = {
    "full": "完整模型",
    "no_adapters": "移除全部热适配器",
    "no_geometry_residual": "移除基础几何残差",
    "no_rgb_source_blend": "移除源 RGB 混合",
    "no_neighborhood": "移除邻域细化",
    "no_crossview": "移除跨视图细化",
    "no_surface": "移除表面细化",
    "reconstruction_only": "只保留基础重建",
    "da3_rgbt_wrapper": "原始 DA3 权重 + RGB-T 包装",
    "anchored_da3": "DA3 + 深度锚定/覆盖修正",
    "no_neighborhood_feedback": "移除邻域源渲染反馈",
    "no_neighborhood_guard": "移除邻域源保护",
    "no_crossview_feedback": "移除跨视图渲染反馈",
    "no_crossview_calibration": "移除跨视图尺度校准",
    "no_crossview_trust": "移除跨视图可信度",
    "no_crossview_thermal_conditioning": "移除跨视图热条件",
    "no_plane_learning": "移除平面学习头",
    "no_plane_covariance": "移除平面协方差",
    "no_plane_anchoring": "移除平面三角化锚定",
    "no_surface_guard": "移除表面几何保护",
    "no_surface_fusion": "移除表面外观融合",
    "no_surface_appearance": "移除表面外观学习",
    "no_appearance_trust": "移除外观可信度",
    "no_dense_depth": "移除稠密深度修正",
    "no_source_anchors": "移除局部深度锚点",
    "no_rgb_feedback": "移除 RGB 残差反馈",
    "no_thermal_feedback": "移除热残差反馈",
    "no_thermal_consensus": "移除热反馈跨视图一致性",
    "no_thermal_blur": "移除热反馈平滑",
    "no_adapter_9": "移除第 9 层热适配器",
    "no_adapter_19": "移除第 19 层热适配器",
    "no_adapter_29": "移除第 29 层热适配器",
    "no_adapter_39": "移除第 39 层热适配器",
    "no_late_adapters": "移除第 29/39 层热适配器",
    "pruned": "最终精简组合",
}


def load(path):
    return json.loads(path.read_text())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default="outputs/rgbt/ablation_20260920")
    a = p.parse_args()
    root = Path(a.root)
    screen = load(root / "screen_summary/summary.json") + load(
        root / "adapters_summary/summary.json"
    )
    confirmation = load(root / "confirmation_summary/summary.json")
    coverage = load(root / "confirmation_summary/coverage.json")
    pruning = load(root / "pruned.pruning.json")
    verification = load(root / "pruning_equivalence.json")
    rows = [
        json.loads(line)
        for path in sorted((root / "confirmation").glob("rows_*.jsonl"))
        for line in path.read_text().splitlines()
    ]
    assert (
        coverage["scenes"] == 30
        and coverage["targets"] == 383
        and coverage["rows"] == 3064
    )

    def cell(data, variant, split, views, metric):
        return next(
            r
            for r in data
            if r["variant"] == variant
            and r["split"] == split
            and r["views"] == views
            and r["metric"] == metric
        )

    lines = [
        "# DA3 RGB-T 模块消融与精简结果",
        "",
        "默认模型已按本次消融精简。主干实际为 **DA3-Giant-1.1**，不是 DA3-Base；四个热适配器、邻域细化、跨视图细化、共享表面、深度锚定和可见性保护均保留。",
        "",
        "## 移除内容",
        "",
        "- 平面学习头：30 场景筛查和全部保留帧复核中收益接近零；移除该头及其触发的第二轮表面处理。共享平面关联、三角化锚定和几何拒绝仍保留。",
        "- 热反馈 7×7 平滑：总体上降低热图 PSNR/SSIM；默认直接使用受可见性与跨视图一致性约束的热残差。",
        "- 单次迭代下不可达的两个 `state_conv` 层、幅度固定为 0 的基础 RGB 残差头；这些分支的参数从部署权重实际删除。",
        "- 三个无调用入口的历史函数：`photometric_consensus`、`thermal_source_residual`、`angular_thermal`，共 214 行。",
        "",
        f"实际删除 {pruning['removed_parameters']:,} 个参数。相对 13.59 亿参数主干，权重体积变化很小；主要收益来自省去冗余表面计算。原有研究开关保留以便加载旧权重和复现消融，但当前部署权重不再包含已删除的头/状态层。",
        "",
        "## 评测方法与边界",
        "",
        "- 30 个场景：14 个训练场景用于回归，16 个非训练场景用于检查泛化；包括室外建筑、植被、工业物体、暗光、玻璃、冷热杯、人物和额外采集场景。",
        "- 首轮 29 配置 × 90 目标 × 2 协议 = 5,220 行；热适配器细分 6 配置 × 60 目标 × 2 协议 = 720 行。去重后共 34 种筛查配置。",
        "- 最终复核覆盖全部 383 个保留目标帧；近邻 3 视图与宽视角 6 视图共 766 组，完整模型、两个单项删减及组合共 3,064 行。各配置源图/目标图/相机逐项配对。",
        "- 所有结果为固定权重的推理消融，没有为每种架构重新训练。均值先按场景内目标平均，再对场景等权平均；95% 区间按场景均值 bootstrap 10,000 次。区间条件于当前权重，不代表多训练种子的置信度。",
        "- 历史测试场景此前已被查看，本次筛查也用于选择删减，不能把扩展复核称为全新独立测试。目标图像与稀疏深度只用于指标，模型反馈只读取源图。",
        "- 质量实验使用已知相机及 252×336 / 336×252 图像。预测相机只做功能及物理删减等价核验，未做全场景质量消融；不外推到其他分辨率。",
        "- 深度来自 COLMAP 稀疏点，不是激光扫描真值；表面厚度在各配置共用的源视图候选掩码/法向上计算，包含拒绝区且不按透明度删除，只衡量内部一致性。",
        "- 输电塔 RGB 为 640×480、热图为 635×477，缺少确认配准映射，因此排除；没有强行缩放后计入有效场景。",
        "",
        "## 全部保留帧：精简组合 − 完整模型",
        "",
        "PSNR/SSIM 越高越好，LPIPS/深度误差/厚度越低越好。",
        "",
        "| 集合 | 输入 | RGB PSNR | 热 PSNR | RGB SSIM | 热 SSIM | LPIPS | 深度 log 误差 | 固定区域厚度 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split, label in [
        ("all", "全部 30 场景"),
        ("nontrain", "非训练 16 场景"),
        ("train", "训练 14 场景"),
    ]:
        for views in [3, 6]:
            values = [
                cell(confirmation, "pruned", split, views, m)["delta"]
                for m in [
                    "rgb_psnr",
                    "thermal_psnr",
                    "rgb_ssim",
                    "thermal_ssim",
                    "rgb_lpips",
                    "depth_log_error",
                    "fixed_surface_thickness",
                ]
            ]
            lines.append(
                f"| {label} | {views} | "
                + " | ".join(f"{v:+.6f}" for v in values)
                + " |"
            )
    lines += [
        "",
        "## 不确定性及局部退化",
        "",
        "| 非训练场景 | 输入 | 平均差值 | 95% 场景 bootstrap 区间 | 改善/退化场景数 | 最差场景差值 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for views in [3, 6]:
        for metric in ["rgb_psnr", "thermal_psnr", "thermal_ssim", "depth_log_error"]:
            r = cell(confirmation, "pruned", "nontrain", views, metric)
            lines.append(
                f"| {metric} | {views} | {r['delta']:+.6f} | [{r['ci95_low']:+.6f}, {r['ci95_high']:+.6f}] | {r['improved_scenes']}/{r['worse_scenes']} | {r['worst_delta']:+.6f} |"
            )
    lines += [
        "",
        "删减并不保证每个场景、每个目标都改善。下表列出组合删减后热 PSNR 下降最多的场景，避免只报告平均收益。",
        "",
        "| 场景 | 输入 | 热 PSNR 差值 | RGB PSNR 差值 |",
        "|---|---:|---:|---:|",
    ]
    scene_rows = load(root / "confirmation_summary/per_scene.json")
    worst = sorted(
        (
            r
            for r in scene_rows
            if r["variant"] == "pruned" and r["metric"] == "thermal_psnr"
        ),
        key=lambda r: r["delta"],
    )[:8]
    for r in worst:
        rgb = next(
            x
            for x in scene_rows
            if x["variant"] == "pruned"
            and x["scene"] == r["scene"]
            and x["views"] == r["views"]
            and x["metric"] == "rgb_psnr"
        )
        lines.append(
            f"| {r['scene']} | {r['views']} | {r['delta']:+.5f} | {rgb['delta']:+.5f} |"
        )
    reference_rows = {(r["id"], r["views"]): r for r in rows if r["variant"] == "full"}
    lines += [
        "",
        "单目标仍有退化：",
        "",
        "| 输入 | 指标 | 最差目标 | 差值 |",
        "|---|---|---|---:|",
    ]
    for views in [3, 6]:
        for metric in ["rgb_psnr", "thermal_psnr"]:
            worst_delta, target = min(
                (r[metric] - reference_rows[r["id"], views][metric], r["id"])
                for r in rows
                if r["variant"] == "pruned" and r["views"] == views
            )
            lines.append(f"| {views} | {metric} | {target} | {worst_delta:+.5f} |")
    lines += [
        "",
        "## 全部模块筛查（非训练场景均值）",
        "",
        "表中数值为移除该分支后相对完整模型的变化。单项均值变好也可能伴随严重局部退化，因此没有仅按均值删除保护模块。",
        "",
        "| 配置 | 3视图 RGB dB | 3视图 热 dB | 6视图 RGB dB | 6视图 热 dB |",
        "|---|---:|---:|---:|---:|",
    ]
    for variant in dict.fromkeys(r["variant"] for r in screen):
        if variant == "full":
            continue
        values = [
            cell(screen, variant, "nontrain", v, m)["delta"]
            for v in [3, 6]
            for m in ["rgb_psnr", "thermal_psnr"]
        ]
        lines.append(
            f"| {LABELS.get(variant, variant)} | "
            + " | ".join(f"{v:+.5f}" for v in values)
            + " |"
        )
    lines += [
        "",
        "`da3_rgbt_wrapper` 使用与官方 checkpoint 逐张量一致的 971 个主干/预测头张量，关闭所有热适配器、学习残差和后级细化。原始 DA3 不原生输出热图，这里保留已知源相机尺度对齐及热采样包装；它不是未经修改的官方 API。`anchored_da3` 额外加入深度锚定与覆盖修正，避免把这些修正的收益误记到后级模块。",
        "",
        "## 部署核验与成本",
        "",
        "精简 checkpoint 已严格重载；真实权重的删减结果与已评测的开关组合逐项对照，包含 3/6 视图和已知/预测相机。完整/缓存前向一致；49 项测试通过，另完成一个真实训练梯度步骤，未保存该步骤更新。",
        "",
        "| 输入/相机 | 原模型秒 | 精简模型秒 | 时延降低 |",
        "|---|---:|---:|---:|",
    ]
    for name, timing in verification["runtime"].items():
        before, after = [timing[k]["median_seconds"] for k in ["before", "after"]]
        lines.append(
            f"| {name} | {before:.4f} | {after:.4f} | {(1 - after / before) * 100:.1f}% |"
        )
    maximum = max(v for case in verification["cases"].values() for v in case.values())
    lines += [
        "",
        f"上述时延为单 GPU、同一 DailyStuff 样例、源特征缓存预热后 5 次完整前向的中位数，不含目标渲染，不代表全部场景。物理删减与开关消融的输出最大绝对差为 `{maximum:.8g}`。",
        "",
        "## 复现与回退",
        "",
        "- [逐场景 JSON/CSV](confirmation_summary/per_scene.csv) · [汇总及置信区间](confirmation_summary/summary.csv) · [所有筛查结果](screen_summary/summary.csv)",
        "- [完整图像对照浏览器](index.html) · [默认模型前向核验](pruned_verification.json) · [物理删减等价与时延](pruning_equivalence.json)",
        "- 原 checkpoint 和改动前源码保留在 `before/`；原权重 SHA256 为 `4cccf1541196ec5adf54641e922edc088d0eb02962c2d0718405143a17d863ce`。",
        f"- 精简权重 SHA256：`{pruning['checkpoint_sha256']}`。",
        "",
        "```bash",
        "# 两张 GPU 分别使用 --shard 0/1 --device cuda:0/1",
        "OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \\",
        "bash scripts/rgbt_python.sh scripts/ablate_rgbt.py \\",
        "  --output outputs/rgbt/recheck --targets 100000 --shards 2 --shard 0 \\",
        "  --device cuda:0 --lpips --surface-metrics \\",
        "  --variants-file outputs/rgbt/ablation_20260920/selected_variants.json \\",
        "  --variants full no_plane_learning no_thermal_blur pruned",
        "```",
        "",
    ]
    (root / "REPORT.md").write_text("\n".join(lines))

    base = {(r["id"], r["views"]): r for r in rows if r["variant"] == "full"}
    browser = []
    for r in rows:
        if r["variant"] != "pruned":
            continue
        ref = base[r["id"], r["views"]]
        browser.append(
            {
                "scene": r["scene"],
                "id": r["id"],
                "views": r["views"],
                "split": r["split"],
                "rgb": r["rgb_psnr"] - ref["rgb_psnr"],
                "thermal": r["thermal_psnr"] - ref["thermal_psnr"],
            }
        )
    document = """<!doctype html><html lang="zh"><meta charset="utf-8"><title>DA3 RGB-T 消融</title>
<style>body{font:16px/1.7 system-ui;max-width:1450px;margin:24px auto;padding:20px;background:#f5f7fa;color:#172538}select{font:inherit;max-width:100%;padding:8px;margin:8px}img{width:100%;background:white}main{display:grid;grid-template-columns:1fr 1fr;gap:16px}.card{background:white;padding:16px;border-radius:8px}a{color:#1665ab}pre{white-space:pre-wrap}@media(max-width:800px){main{grid-template-columns:1fr}}</style>
<h1>DA3 RGB-T 消融与精简</h1><p>30 场景 · 383 目标 · 3/6 输入视图 · 固定权重消融</p>
<p><a href="REPORT.md">完整报告与局部失败</a> · <a href="confirmation_summary/summary.csv">统计与置信区间 CSV</a> · <a href="confirmation_summary/per_scene.csv">逐场景 CSV</a></p>
<p>已精简：移除平面学习头及热反馈平滑，清理无效头和状态层。大模块和四个热适配器保留。每张图各行左侧是真值，右侧是输出。</p>
<label>场景<select id="scene"></select></label><label>输入<select id="views"><option>3</option><option>6</option></select></label><label>目标<select id="target"></select></label><p id="metrics"></p>
<main><div class="card"><h2>完整模型</h2><img id="before"></div><div class="card"><h2>精简模型</h2><img id="after"></div></main>
<script>const rows=DATA;const scene=document.querySelector('#scene'),views=document.querySelector('#views'),target=document.querySelector('#target');
for(const s of [...new Set(rows.map(r=>r.scene))].sort()){scene.add(new Option(s,s));}
function targets(){target.replaceChildren();for(const r of rows.filter(r=>r.scene===scene.value && r.views===Number(views.value))){target.add(new Option(r.id,r.id));}show();}
function show(){const r=rows.find(r=>r.scene===scene.value && r.views===Number(views.value) && r.id===target.value);if(!r)return;document.querySelector('#metrics').textContent=`精简 − 完整：RGB PSNR ${r.rgb.toFixed(5)} dB；热 PSNR ${r.thermal.toFixed(5)} dB。数据分组：${r.split}。`;document.querySelector('#before').src=`confirmation/panels/${r.views}/full/${r.id}.jpg`;document.querySelector('#after').src=`confirmation/panels/${r.views}/pruned/${r.id}.jpg`;}
scene.onchange=targets;views.onchange=targets;target.onchange=show;targets();</script></html>"""
    (root / "index.html").write_text(
        document.replace(
            "DATA", json.dumps(browser, ensure_ascii=False).replace("</", "<\\/")
        )
    )
    for row in browser:
        for name in ["full", "pruned"]:
            assert (
                root
                / "confirmation/panels"
                / str(row["views"])
                / name
                / (row["id"] + ".jpg")
            ).is_file()
    print("REPORT", root / "REPORT.md", "panels", len(browser) * 2)


if __name__ == "__main__":
    main()
