#!/usr/bin/env python3
"""Build an offline gallery and scene-weighted tables from dataset evaluations."""

import argparse
import csv
import html
import json
import statistics
from pathlib import Path


METRICS = [
    "rgb_psnr",
    "rgb_ssim",
    "thermal_psnr",
    "thermal_ssim",
    "rgb_lpips",
    "gaussian_depth_log_error",
    "alpha_mean",
]


def write_csv(path, rows, fields):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def table(rows, fields):
    def cell(value):
        return f"{value:.4f}" if isinstance(value, float) else html.escape(str(value))

    return (
        '<div class="table"><table><thead><tr>'
        + "".join(f"<th>{html.escape(f)}</th>" for f in fields)
        + "</tr></thead><tbody>"
        + "".join(
            "<tr>" + "".join(f"<td>{cell(r[f])}</td>" for f in fields) + "</tr>"
            for r in rows
        )
        + "</tbody></table></div>"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    root = args.output
    rows, per_scene, summaries, reports = [], [], [], {}
    for domain in ["RGBT-Scenes", "ThermoScenes"]:
        report = json.loads((root / domain / "results.json").read_text())
        reports[domain] = report
        manifest = json.loads((Path(report["data"]) / "manifest.json").read_text())
        scenes = {
            s["id"]: s
            for s in manifest["scenes"]
            if s["domain"] == domain
            and (report["split"] == "all" or s["split"] == report["split"])
        }
        for protocol, result in report["results"].items():
            if report["targets_per_scene"] == "all reserved targets":
                expected = {
                    s["frames"][i]["id"]
                    for s in scenes.values()
                    for i in s["target_pool"]
                }
                actual = [r["id"] for r in result["per_view"]]
                if len(actual) != len(set(actual)) or set(actual) != expected:
                    raise ValueError(
                        f"Incomplete or duplicate targets in {domain}/{protocol}"
                    )
            for row in result["per_view"]:
                if row["id"] in row["context_ids"]:
                    raise ValueError("Target included among source views")
                if not (root / domain / row["comparison"]).is_file():
                    raise FileNotFoundError(row["comparison"])
                rows.append({**row, "protocol": protocol})
            for scene, metrics in result["per_scene"].items():
                per_scene.append(
                    {
                        "dataset": domain,
                        "scene": scene,
                        "split": scenes[scene]["split"],
                        "protocol": protocol,
                        "targets": sum(r["scene"] == scene for r in result["per_view"]),
                        **metrics,
                    }
                )
            for split in ["all", "train", "val", "test", "val+test"]:
                selected = [
                    r
                    for r in per_scene
                    if r["dataset"] == domain
                    and r["protocol"] == protocol
                    and (
                        split == "all"
                        or r["split"] == split
                        or (split == "val+test" and r["split"] in ("val", "test"))
                    )
                ]
                if selected:
                    summaries.append(
                        {
                            "dataset": domain,
                            "protocol": protocol,
                            "split": split,
                            "scenes": len(selected),
                            "targets": sum(r["targets"] for r in selected),
                            **{
                                key: statistics.mean(r[key] for r in selected)
                                for key in METRICS
                            },
                        }
                    )

    write_csv(
        root / "summary.csv",
        summaries,
        ["dataset", "protocol", "split", "scenes", "targets", *METRICS],
    )
    write_csv(
        root / "per_scene.csv",
        per_scene,
        ["dataset", "scene", "split", "protocol", "targets", *METRICS],
    )
    write_csv(
        root / "per_target.csv",
        rows,
        [
            "dataset",
            "scene",
            "split",
            "protocol",
            "id",
            *METRICS,
            "context_ids",
            "comparison",
        ],
    )
    (root / "summary.json").write_text(
        json.dumps(
            {
                "aggregation": "Mean within each scene, then equal scene weights",
                "evaluation_groups": len(rows),
                "summary": summaries,
                "weights": {d: r["weights"] for d, r in reports.items()},
            },
            indent=2,
        )
        + "\n"
    )

    overall = [r for r in summaries if r["split"] == "all"]
    heldout = [r for r in summaries if r["split"] == "val+test"]
    scene_count = len({r["scene"] for r in rows})
    target_count = len({r["id"] for r in rows})
    columns = ["dataset", "protocol", "scenes", "targets", *METRICS[:5]]
    note = (
        "使用当前 checkpoints/RGBT/model.pt，COLMAP 已知相机，前馈推理，无逐场景优化。"
        "RGBT-Scenes 分辨率为 252×336（高×宽），ThermoScenes 为 336×252。"
        "nearest3 选择邻近的 3 张源图；wide6 选择范围更广的 6 张源图，源图组合不同。"
        "目标图像不作为模型输入；目标来自原有 target_pool。"
        "指标先在场景内平均，再对场景等权平均。"
        "PSNR、SSIM 越高越好，LPIPS 越低越好。SSIM 使用现有项目实现。"
        "热图指标对应数据集灰度／显示亮度，并非温度精度。"
        "train 场景参与过训练，val/test 场景未参与梯度训练，但曾用于历史评价。"
        "COLMAP 标定本身可能使用过目标图像，当前结果不能代表未知相机条件。"
    )
    md = [
        "# SharedRefiner：RGBT-Scenes 与 ThermoScenes",
        "",
        note,
        "",
        f"共 {scene_count} 个场景、{target_count} 个保留目标、{len(rows)} 组推理。",
        "",
        "[打开离线结果画廊](index.html) · [逐场景 CSV](per_scene.csv) · [逐目标 CSV](per_target.csv)",
        "",
        "| 数据集 | 源视图 | RGB PSNR | RGB SSIM | 热 PSNR | 热 SSIM | RGB LPIPS |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in overall:
        md.append(
            f"| {r['dataset']} | {r['protocol']} | {r['rgb_psnr']:.3f} | {r['rgb_ssim']:.4f} | {r['thermal_psnr']:.3f} | {r['thermal_ssim']:.4f} | {r['rgb_lpips']:.4f} |"
        )
    md.extend(
        [
            "",
            "三维导出位于 `<数据集>/scenes/<协议>/<场景>/`，每个场景／协议使用第一个评价目标对应的源图组合，未按指标挑选。",
            "",
            "- `rgb.ply` / `thermal.ply`：RGB 和热外观高斯参数。",
            "- `geometry.html`：可离线旋转的高斯中心点预览。",
            "- `rgbt_trajectory.gif`：源相机之间插值的渲染动画。",
            "- `sources.json`：目标 ID 与源图 ID。",
            "",
            f"全部 {len(rows)} 张对照图保留，包括覆盖不足、空洞和局部重影的目标。",
            "",
        ]
    )
    if (root / "run.sh").exists():
        md.extend(["复现命令见 [run.sh](run.sh)。", ""])
    if (root / "run_manifest.json").exists():
        md.extend(
            ["运行与文件完整性记录见 [run_manifest.json](run_manifest.json)。", ""]
        )
    (root / "REPORT.md").write_text("\n".join(md))

    scene_cards, target_cards = [], []
    for scene_row in per_scene:
        first = next(
            r
            for r in rows
            if r["scene"] == scene_row["scene"]
            and r["protocol"] == scene_row["protocol"]
        )
        domain = first["dataset"]
        base = f"{domain}/{first['scene_export']}"
        for filename in [
            "rgb.ply",
            "thermal.ply",
            "geometry.html",
            "rgbt_trajectory.gif",
            "sources.json",
        ]:
            if not (root / base / filename).is_file():
                raise FileNotFoundError(f"{base}/{filename}")
        scene_cards.append(
            f'<article class="scene-card" data-domain="{domain}" data-protocol="{first["protocol"]}" data-split="{first["split"]}" data-scene="{html.escape(first["scene"])}">'
            f"<h3>{html.escape(first['scene'])} · {first['protocol']}</h3><p>{first['split']} · {scene_row['targets']} 个目标 · "
            f"场景均值 RGB {scene_row['rgb_psnr']:.2f} / 热 {scene_row['thermal_psnr']:.2f} dB</p>"
            f'<a href="{domain}/{first["comparison"]}"><img loading="lazy" src="{domain}/{first["comparison"]}" alt="首个保留目标对照"></a>'
            f'<p><a href="{base}/geometry.html">旋转三维预览</a> · <a href="{base}/rgbt_trajectory.gif">视角动画</a> · '
            f'<a href="{base}/rgb.ply">RGB PLY</a> · <a href="{base}/thermal.ply">热 PLY</a></p></article>'
        )
    for row in rows:
        target_cards.append(
            f'<article class="target-card" data-domain="{row["dataset"]}" data-protocol="{row["protocol"]}" data-split="{row["split"]}" data-scene="{html.escape(row["scene"])}">'
            f"<h3>{html.escape(row['id'])} · {row['protocol']}</h3><p>RGB {row['rgb_psnr']:.2f} dB / SSIM {row['rgb_ssim']:.3f} · "
            f"热 {row['thermal_psnr']:.2f} dB / SSIM {row['thermal_ssim']:.3f}</p>"
            f'<a href="{row["dataset"]}/{row["comparison"]}"><img loading="lazy" src="{row["dataset"]}/{row["comparison"]}" alt="真值与预测"></a></article>'
        )
    document = """<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SharedRefiner · 双数据集结果</title><style>
body{font:15px/1.7 system-ui,sans-serif;color:#dfe7f2;background:#101722;margin:0;padding:32px}main{max-width:1400px;margin:auto}
h1{font-size:32px}h2{margin-top:40px}h3{font-size:15px;overflow-wrap:anywhere}p{color:#b9c8db}a{color:#8ac9ff}
.table{overflow:auto}table{border-collapse:collapse;width:100%;font-size:13px}td,th{padding:10px;text-align:left;white-space:nowrap;border-bottom:1px solid #334255}
th{background:#1b2838}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(370px,1fr));gap:20px}
article{background:#192535;border:1px solid #314358;padding:16px;border-radius:8px}img{width:100%;height:auto;border-radius:4px}
.filters{position:sticky;top:0;background:#101722ee;padding:16px 0;z-index:1;display:flex;flex-wrap:wrap;gap:12px}
select,input{background:#24344a;color:white;border:1px solid #526782;padding:8px;border-radius:4px;max-width:100%}
[hidden]{display:none!important}.badge{color:#9ce6c1}@media(max-width:500px){body{padding:16px}.grid{grid-template-columns:1fr}}
</style></head><body><main><p class="badge">DA3 RGB-T / SharedRefiner / 2026-09-22</p><h1>两个数据集的重建结果</h1>
"""
    document += f"<p>{scene_count} 个场景 · {target_count} 个保留目标 · {len(rows)} 组评测 · {len(per_scene)} 组高斯导出</p><p>{note}</p>"
    document += '<p><a href="REPORT.md">文字报告</a> · <a href="summary.csv">汇总 CSV</a> · <a href="per_scene.csv">逐场景 CSV</a> · <a href="per_target.csv">逐目标 CSV</a></p>'
    document += "<h2>全部场景</h2>" + table(overall, columns)
    document += "<h2>未参与梯度训练的场景（val + test）</h2>" + table(heldout, columns)
    document += (
        "<p>这是历史数据上的复评，各子集场景数见表格。</p><h2>场景与三维预览</h2>"
    )
    document += "<p>图像左侧是真值，右侧是 SharedRefiner；上行是 RGB，下行是热图。每个场景的缩略图和三维导出采用首个评价目标，全部目标见下方画廊。三维页面展示高斯中心点，动画展示高斯渲染。</p>"
    document += '<div class="filters"><label>数据集 <select id="domain"><option value="">全部</option><option>RGBT-Scenes</option><option>ThermoScenes</option></select></label>'
    document += '<label>视图 <select id="protocol"><option value="">全部</option><option>nearest3</option><option>wide6</option></select></label>'
    document += '<label>划分 <select id="split"><option value="">全部</option><option>train</option><option>val</option><option>test</option></select></label><label>场景 <input id="scene" placeholder="输入场景名称"></label></div>'
    document += (
        '<div class="grid">'
        + "".join(scene_cards)
        + '</div><h2>全部目标对照</h2><p id="count"></p><div class="grid">'
        + "".join(target_cards)
        + "</div>"
    )
    document += """<script>
const controls=['domain','protocol','split','scene'].map(x=>document.getElementById(x));
function filter(){let n=0;document.querySelectorAll('article').forEach(a=>{const show=controls.every(c=>!c.value||(c.id==='scene'?a.dataset.scene.toLowerCase().includes(c.value.toLowerCase()):a.dataset[c.id]===c.value));a.hidden=!show;if(show&&a.classList.contains('target-card'))n++;});document.getElementById('count').textContent=`当前显示 ${n} 组目标对照`;}
controls.forEach(c=>c.addEventListener('input',filter));filter();
</script></main></body></html>"""
    (root / "index.html").write_text(document)
    print(
        json.dumps(
            {
                "groups": len(rows),
                "scene_protocols": len(per_scene),
                "summary": overall,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
