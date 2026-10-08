import argparse
import csv
import html
import json
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser(
    description="Report all single-forward regression results, including failures"
)
parser.add_argument("--output", type=Path, required=True)
root = parser.parse_args().output
data = json.loads((root / "test_evidence/eval/metrics.json").read_text())
rows = data["rows"]
names = ["native", "previous", "single"]


def summary(rows):
    scenes = sorted({r["scene"] for r in rows})
    keys = list(rows[0]["single"])
    return {
        name: {
            k: float(
                np.mean(
                    [
                        np.mean([r[name][k] for r in rows if r["scene"] == s])
                        for s in scenes
                    ]
                )
            )
            for k in keys
        }
        for name in names
    }


protocols = {str(v): summary([r for r in rows if r["views"] == v]) for v in [3, 6]}
flat = []
for r in rows:
    for name in names:
        flat.append(
            {**{k: r[k] for k in ["id", "scene", "views"]}, "method": name, **r[name]}
        )
with (root / "test_results.csv").open("w") as f:
    w = csv.DictWriter(f, fieldnames=list(flat[0]))
    w.writeheader()
    w.writerows(flat)
notes = [
    "# 单次前向实验：未通过完整视觉验收",
    "",
    "候选权重：`checkpoints/RGBT_single_forward/model.pt`。原默认 `checkpoints/RGBT/model.pt` 未替换。",
    "采用同一组 nearest3 / wide6 源图，14 个 train 场景 168 个样本训练，2 个 val 场景 16 组样本选择权重。69 个 test 目标 × 2 协议共 138 组回归对照。该测试集已被反复检查，不属于未触碰的新测试。",
    "",
    "## 结果",
    "",
    "下表按测试场景等权汇总。低覆盖比例用目标有效区域内 alpha < 0.5 定义；它不同于几乎全空的 alpha < 0.01。LPIPS 越低越好。",
    "",
    "|源视角|方法|低覆盖 %|近全空 %|RGB PSNR|RGB LPIPS|T PSNR|T LPIPS|",
    "|---|---|---:|---:|---:|---:|---:|---:|",
]
for v, summary_ in protocols.items():
    for name, x in summary_.items():
        notes.append(
            f"|{v}|{name}|{100 * x['holes_0.5']:.3f}|{100 * x['holes_0.01']:.5f}|{x['rgb_psnr']:.3f}|{x['rgb_lpips']:.4f}|{x['thermal_psnr']:.3f}|{x['thermal_lpips']:.4f}|"
        )
notes += [
    "",
    "native = 原始 DA3 高斯 + 源热图采样包装；previous = 原 SharedRefiner；single = 此次单次前向候选。",
    "",
    "**结论：减少空洞这一项有改善，整体 RGB 视觉质量这一项未通过。不能称为问题已完全解决。**",
    "Building img_025 的细杆分叉明显减轻，但部分纹理仍发虚。3 视角的 RGB/T LPIPS 均不如原 SharedRefiner；6 视角热图改善，RGB LPIPS略退化。未用平均指标掩盖这些问题，也没有把候选设为默认。",
    "",
    "## 实现",
    "",
    "一次 DA3 backbone + 一次 Gaussian head + 一次共享残差网络，输出一套 RGB/T 共用的高斯。推理不读目标图、不优化参数、不渲染源图、不调用第二个 DA3，也不拼接两张渲染结果。RGB/T/alpha 在一次目标光栅化中产生。",
    "恢复原生高斯偏移与各向异性协方差，避免原锚定模式对细结构与覆盖范围的硬限制。训练使用图像、感知、多尺度边缘、alpha 覆盖和局部 RGB 保持损失；RGB 残差沿源相机实际投影颜色证据预测。",
    "所有导出高斯均保留，RGB/T PLY 使用同一几何；取消默认裁掉边缘高斯，避免导出时再丢覆盖。",
    "",
    "## 审计与文件",
    "",
    "- [完整 138 张对照画廊](index.html)，图中依次是真值、原生 DA3、原 SharedRefiner、单前向候选。",
    "- [全部逐图指标](test_results.csv)；[原始 JSON](test_evidence/eval/metrics.json)。",
    "- [真实模型调用次数与缓存一致性](inference_audit/audit.json)。",
    "- `inference_audit/` 含实际单前向导出的 PLY、RGB/T 图、alpha、相机轨迹。",
    "- 验证集选中的权重是 `run_evidence/step_0500.pt`；其他尝试均保留在 `run*`，未混合模型输出。",
    "",
    "## 下一步需要解决的实质问题",
    "",
    "原生 DA3 与锚定表示的收益尚不能兼得：前者保留细结构与覆盖，但当前轻量残差训练没有保留/改善所有场景的 RGB 纹理；后者背景误差低，却存在明显细杆几何错误。不能靠回退、换源图或把模糊当作补洞来宣布完成。",
]
(root / "REPORT.md").write_text("\n".join(notes) + "\n")
(root / "summary.json").write_text(
    json.dumps(
        {
            "status": "experimental_not_accepted",
            "protocols": protocols,
            "overall": data["summary"],
            "test_targets": 69,
            "comparisons": 138,
            "default_replaced": False,
        },
        indent=2,
    )
)
cards = []
for r in rows:
    p = f"test_evidence/eval/{r['id']}_{r['views']}.png"
    s = r["single"]
    old = r["previous"]
    cards.append(
        f'<article><h3>{html.escape(r["id"])} / {r["views"]} views</h3><p>RGB LPIPS {old["rgb_lpips"]:.4f} → {s["rgb_lpips"]:.4f}; T LPIPS {old["thermal_lpips"]:.4f} → {s["thermal_lpips"]:.4f}; alpha&lt;0.5 {100 * old["holes_0.5"]:.2f}% → {100 * s["holes_0.5"]:.2f}%</p><a href="{p}"><img loading="lazy" src="{p}"></a></article>'
    )
(root / "index.html").write_text(
    '<!doctype html><meta charset="utf-8"><title>单次前向候选：未通过完整验收</title><style>body{font:16px sans-serif;margin:24px;background:#f3f3f3}article{background:white;padding:12px;margin-bottom:20px}img{width:100%;max-width:1344px}p{line-height:1.6}</style><h1>单次前向候选：未通过完整视觉验收</h1><p>空洞减少，但整体 RGB 仍退化。原默认权重未替换。列顺序：真值 / 原生 DA3 / 原 SharedRefiner / 单前向候选。</p><p><a href="REPORT.md">完整报告</a> · <a href="test_results.csv">逐图指标</a></p>'
    + "".join(cards)
)
