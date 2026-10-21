# SEAR 全场景：官方 DA3 与最佳热红外适配器

> 当前结果已整理至 `outputs/depth_pointcloud/`，每组包含原来的 60 个场景。2026-10-19 已重新推理官方联合输入及两组单模态结果，见下一段。完整导航见 [outputs/README.md](../outputs/README.md)，旧路径对照见 [PATH_MAPPING.csv](../outputs/PATH_MAPPING.csv)。后文保留原实验说明和当时的复现命令，原 artifacts 记录保留历史路径；2026-10-13 的官方单模态导出已在前次清理中删除。

2026-10-19 官方三组结果分别位于 `outputs/depth_pointcloud/official/`（12 RGB + 12 红外）、`official_rgb_only/`（仅 12 RGB）、`official_thermal_only/`（仅 12 红外）。三组使用原来相同的 12 对帧索引，均重新进行独立模型推理，设置保持 BF16、518、`saddle_balanced`。记录、原始预测和完整验证见 [official_three_modalities_20261019](../artifacts/official_three_modalities_20261019/README.md)。被替换的旧联合结果也在该记录目录归档。

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
/home/lf/miniconda3/envs/da3/bin/python -u scripts/rerun_official_modalities.py \
  --record-dir artifacts/official_three_modalities_20261019
```

已有运行记录时加 `--resume`，仅验证已完成结果时加 `--verify-only`；新一轮完整推理使用新的记录目录。

本次覆盖 `/home/lf/feedforward/data/SEAR` 中全部 60 个场景。
每场景均匀取 12 对 RGB／热红外图像，包含首尾帧；两次推理使用完全相同的图像、顺序和预处理。
推理读取配对图像路径，无需数据中的深度或相机标签，因此也包含训练时因缺少标签而排除的 3 个场景。

- `official`：本地官方 `DA3-GIANT-1.1` 底座，严格加载预训练权重。
- `new`：相同底座，加 `checkpoints/thermal_giant_sear/adapter_best.pt`，第 16 轮、4784 步。

使用 GPU 1，BF16，DA3 原生 `saddle_balanced` 参考视图策略。
最长边 518，按官方 InputProcessor 预处理；RGB 和热红外联合推理。
适配器推理接收热红外模态标记，原始模型使用普通图像序列。
未提供数据集的深度或相机内外参，未使用 Gaussian 分支。

## 结果目录

```text
outputs/comparison/
  official/<数据集>/<场景>/
  new/<数据集>/<场景>/
```

每个场景目录只有：

- `rgb.ply`、`thermal.ply`、`combined.ply`：RGB、热红外及合并点云。
- `depth/rgb/*.png`、`depth/thermal/*.png`：24 张彩色深度图，文件名包含源帧索引和图像名称。
- `depth.png`：4 对代表帧的 RGB／热红外深度汇总图。

PLY 为标准二进制 XYZ／RGB 点云，采用各次模型预测的原始坐标。
深度为相对深度，PNG 颜色在单个场景的单次运行内使用统一范围；范围和原始坐标尺度由各次预测决定。
点云过滤规则一致：置信度不低于每图中位数、剔除最远 1% 深度与 2% 边缘，每图最多 15000 点。
深度 PNG 使用完整预测，没有按点云规则过滤。

结果目录不保存网页、原始输入副本、NPZ／NPY、JSON 或日志。
输入清单、权重哈希、运行时间、逐文件哈希及可续跑进度另存于
`artifacts/comparison_all_scenes_20261010/`。

## 运行与续跑

在仓库根目录执行：

```bash
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=8 /home/lf/miniconda3/envs/da3/bin/python -u \
  scripts/infer_sear_comparison.py \
  --record-dir artifacts/comparison_all_scenes_20261010 --pairs 12
```

已有运行记录时在相同命令末尾加 `--resume`。
脚本先跑完官方模型的全部场景，再加载适配器跑第二轮；逐场景原子发布结果，核对两轮输入哈希。
续跑会校验已完成文件的大小与 SHA-256，拒绝覆盖未经本次运行记录确认的已有场景。

日志：`artifacts/comparison_all_scenes_20261010/run.log`。
进度：`artifacts/comparison_all_scenes_20261010/progress.json`。

检查场景清单：

```bash
/home/lf/miniconda3/envs/da3/bin/python scripts/infer_sear_comparison.py \
  --record-dir artifacts/comparison_all_scenes_20261010 --dry-run
```

基础检查：

```bash
/home/lf/miniconda3/envs/da3/bin/python -m unittest discover \
  -s tests -p 'test_sear_comparison.py' -v
```

## 官方模型的单模态对照

`outputs/comparison` 另含两组独立推理结果，均覆盖原场景清单中的 60 个场景：

- `official_rgb_only/<数据集>/<场景>/`：只输入原选定的 12 张 RGB。
- `official_thermal_only/<数据集>/<场景>/`：只输入相同帧索引的 12 张热红外。

两组均重新运行官方 `DA3-GIANT-1.1`，使用 `saddle_balanced`、BF16、处理分辨率 518
和原随机种子。每次模型调用只收到 12 张同模态图片，不加载热红外 adapter，
不提供数据集相机参数，不使用 Gaussian 分支。预处理结果逐场景核对为原联合输入的对应子集。
点云使用原置信度、深度和边缘过滤规则，以及原视图索引对应的采样种子，每图最多 15000 点。

每个单模态场景目录含对应的 `rgb.ply` 或 `thermal.ply`、`depth.png`
以及 `depth/rgb/*.png` 或 `depth/thermal/*.png` 中的 12 张逐帧深度图。
点云保留各次独立预测的原始坐标与尺度；深度颜色范围按各次预测分别计算。

原始预测、导出像素索引、输入清单、官方权重哈希、代码快照、运行日志和完整核验记录位于
`artifacts/comparison_official_single_modality_20261013/`。
核验逐点检查点云坐标和颜色、逐像素检查深度 PNG，并校验原 `official` 和 `new` 的所有文件。

```bash
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
/home/lf/miniconda3/envs/da3/bin/python -u \
  scripts/infer_sear_official_single_modality.py
```

续跑时加 `--resume`；已有结果的完整核验使用 `--verify-only`。
