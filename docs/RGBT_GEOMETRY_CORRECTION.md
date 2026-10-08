# RGB-T 多视图几何修正（实验候选）

这次修改针对细杆弯曲、不同源视图的轮廓重影。当前候选能改善这些问题，但**尚未消除全部伪影，也不能保证所有视角、所有 RGB/T 指标都提高**。默认 `checkpoints/RGBT/model.pt` 保留；新增候选通过标准 `load_rgbt_model` 加载并执行完整网络前向。

完整结果和逐场景指标见 [实验报告](../outputs/rgbt/geometry_fix_20260923/REPORT.md)。

## 模块与接入位置

`src/depth_anything_3/rgbt/multiview_depth.py` 实现 `MultiViewDepthCorrector`：

1. 沿每个源高斯的实际相机射线搜索原深度的 0.4～2.5 倍，建立 RGB/局部对比度匹配代价体。不会先用原深度的 10% 误差门限排除正确匹配。
2. 使用匹配峰的区分度、相对原深度的改善和独立视角深度一致性筛选。使用单个深度模式及局部亚采样修正，避免把前景和背景两个深度模式平均到一起。
3. 对可靠深度做颜色和深度边界约束下的逆深度平滑。
4. `line_consensus.py` 检测源 RGB 中的直线段，只用有多视图证据的样本鲁棒拟合沿线逆深度，填补直线段内部的局部错误。颜色断层和无效像素限制传播；不假定世界竖直方向，不使用场景名或目标图。
5. 三视角采用保守门控；超过三视角时，要求至少两个其他源相机的深度一致，减少重复纹理造成的偶然匹配。
6. 同时修改 Gaussian means、Gaussian scales 和返回的 depth，保持源相机射线和高斯角尺寸一致。选中的候选保留原不透明度。

最终选中的位置是 **SharedRefiner 之后、目标渲染之前**。将大幅校正后的几何直接交给现有冻结 refiner，在独立验证场景上退化，因此未选择该位置。新模块没有目标相机、目标 RGB、目标热图输入。目标图只用于训练损失或离线评估。

不配置 `geometry_correction` 的旧 checkpoint 不创建新参数，继续使用原路径。现有 `prepared` 是旧几何下的 refiner 缓存，与启用几何校正的调用一起传入会明确报错，避免静默使用失效缓存。

## 显卡约束和运行

所有 CUDA 工作使用物理显卡 1。统一包装脚本按 `nvidia-smi -i 1` 获取 UUID 并设置 `CUDA_VISIBLE_DEVICES`。进程内的 `cuda:0` 因此对应**物理显卡 1**。脚本识别显卡失败时退出，不回退到显卡 0。

```bash
# 在项目根目录执行；读取最终候选并重跑已有 138 个测试组合。
bash scripts/rgbt_gpu1_python.sh scripts/verify_rgbt_geometry_candidate.py \
  --weights outputs/rgbt/geometry_fix_20260923/RGBT_geometry_candidate.pt \
  --cache outputs/rgbt/geometry_fix_20260923/cache/test \
  --output outputs/rgbt/geometry_fix_20260923/reproduced_test

# 单元回归，包括独立合成几何和原有 RGB-T 测试。
bash scripts/rgbt_gpu1_python.sh -m unittest discover -s tests -p 'test_rgbt*.py'
```

正常推理也可以直接把现有脚本的 `--weights` 换成上述候选路径，并用相同的 GPU 1 包装脚本启动。输入大小、单场景 batch 和相机约定沿用原网络。本次真实场景回归使用已知相机和 336 尺寸数据；没有验证未知相机下的可靠性。

## 复现实验数据与训练

缓存脚本保存固定的源视图、原网络几何和基线渲染。训练采样 14 个场景、112 组；验证采样另外 2 个场景、16 组；测试为另外 4 个场景、138 组。

```bash
bash scripts/rgbt_gpu1_python.sh scripts/cache_rgbt_geometry.py \
  --output outputs/rgbt/geometry_fix_20260923/cache

# 分别为 train、val 生成 source-only 匹配证据。
bash scripts/rgbt_gpu1_python.sh scripts/probe_rgbt_geometry.py \
  --cache outputs/rgbt/geometry_fix_20260923/cache/train \
  --output outputs/rgbt/geometry_fix_20260923/evidence/train --prepare-only
bash scripts/rgbt_gpu1_python.sh scripts/probe_rgbt_geometry.py \
  --cache outputs/rgbt/geometry_fix_20260923/cache/val \
  --output outputs/rgbt/geometry_fix_20260923/evidence/val --prepare-only

# 可选置信度/残差头训练；这不是当前候选所使用的已接受权重。
bash scripts/rgbt_gpu1_python.sh scripts/train_rgbt_geometry.py \
  --cache outputs/rgbt/geometry_fix_20260923/cache \
  --evidence outputs/rgbt/geometry_fix_20260923/evidence \
  --output outputs/rgbt/geometry_fix_20260923/new_training \
  --resume outputs/rgbt/geometry_fix_20260923/analytic_wide_consensus.pt \
  --lr 0.0003 --steps 600

# 从基础模型和独立修正模块生成可直接加载的完整 checkpoint。
bash scripts/rgbt_python.sh scripts/export_rgbt_geometry_candidate.py \
  --corrector outputs/rgbt/geometry_fix_20260923/analytic_wide_consensus.pt \
  --output outputs/rgbt/geometry_fix_20260923/reexported_candidate.pt
```

已执行两轮各 600 步的修正头训练，验证最优均为训练前的 step 0；因此当前候选采用解析匹配与门控，`learned=false`。没有把训练后退化的结果替换进去，也没有重训 DA3 基础模型。`trainable_modules` 的原 refiner 训练协议不变，独立训练脚本负责新头的优化。

## 限制

- 强遮挡、低纹理、细物体混合像素或错误相机仍可能没有足够的匹配证据；保守回退会留下原错误几何。
- 单个源像素对应一个高斯，不能完整表示同一像素内的前景/背景层。仅改中心深度无法解决全部边缘混合和新露出的背景空洞。
- 灯杆主体改善不等于灯头、边缘和整图都修复。当前仍有失败样例，热图 SSIM 的总体均值仍小幅降低。
- 代价体增加运行成本，线段拟合目前在 CPU 上执行。未做部署性能优化。
- `source_transfer`、`conflict_strength` 和训练头是实验选项，未纳入本次选中的配置。局部补全、强制抑制冲突高斯、冻结 refiner 前插入校正等方案均有验证退化。
- 测试图用于问题诊断和反复回归，不应把当前统计当作一次完全未见测试集上的泛化结论；推理本身不读取目标图。
