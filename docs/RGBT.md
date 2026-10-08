# RGB-T 统一高斯重建模型

默认 `checkpoints/RGBT/model.pt` 使用单一共享细化模块，顶层为 `reconstruction -> refiner -> renderer`。已从默认模型中删除四个向 ViT 注入热特征的适配器及其专用热特征层；热图仍进入重建 neck 和细化模块。完整权重包含 DA3-Giant 主干，推理不需要教师或实验目录。删除前的完整模型保存在 `checkpoints/RGBT/model_with_adapters.pt`。

细化部分使用一个共享编码器、局部/跨视图注意力和一个 15 通道联合输出头，按步骤条件复用三次。细化参数仍为 564,815；删除热适配器后总模型约 13.57 亿参数。[结构与训练细节](RGBT_LEARNED_JOINT.md)。

此前带适配器版本在 30 场景、766 组完整对照中，3/6 视图相对无细化基础重建的 RGB PSNR 提升 1.47/1.54 dB，热 PSNR 提升 1.94/1.89 dB。[原版验收报告及全部图像](../outputs/rgbt/joint_release_20260920/REPORT.md)不能作为当前删减版的完整数据集指标。当前删减版已在 Building img_025 的 3/6 视图上确认与关闭适配器的推理逐像素一致；[对应结果](../outputs/test-noadapter/results.json)。

## 推理

在仓库根目录执行：

```bash
bash scripts/rgbt_python.sh scripts/infer_rgbt.py \
  --rgb rgb_1.png rgb_2.png rgb_3.png \
  --thermal thermal_1.png thermal_2.png thermal_3.png \
  --cameras source_cameras.npz \
  --output outputs/rgbt/new_scene --device cuda:0
```

`--weights` 默认是 `checkpoints/RGBT/model.pt`，省略 `--cameras` 使用预测相机。相机 NPZ 包含原图像素内参 `intrinsics` 与 world-to-camera 外参 `extrinsics`。要求至少两对配准图像，默认长边 336；输出 RGB/T 高斯 PLY 与可视化。

不同传感器网格可通过 `--thermal-registration` 提供 `map_x/map_y`，程序不自行估计配准。`--radiometric` 支持已标定热值编码；未标定灰度不能当作真实温度。

```python
import torch
from depth_anything_3.rgbt import load_rgbt_model
from depth_anything_3.rgbt.renderer import render_rgbt

model, checkpoint = load_rgbt_model("checkpoints/RGBT/model.pt", "cuda:0")
del checkpoint
# rgb [1,V,3,H,W], thermal [1,V,1,H,W]，取值 [0,1]
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    output = model(rgb, thermal, extrinsics, intrinsics, source_valid=valid)
prediction = render_rgbt(output, target_extrinsics, target_intrinsics, H, W)
```

RGB/T 共享高斯中心、尺度、旋转与不透明度。RGB 使用二阶 SH，热外观为单通道值；保留 `V×H×W` 个高斯，`H/W` 是 14 的倍数，当前使用 `B=1`。输入只包含源图和可选源相机；目标图像只用于训练或评价，推理无逐场景梯度优化。

## 训练、续训与验证

```bash
# 从固定共享网络初始化重新训练
bash scripts/rgbt_python.sh scripts/train_rgbt_joint.py \
  --config configs/rgbt_joint_train.json \
  --output outputs/rgbt/retrain_shared --device cuda:0

# 从交付训练状态继续训练
bash scripts/rgbt_python.sh scripts/train_rgbt_joint.py \
  --resume checkpoints/RGBT/training_state.pt \
  --output outputs/rgbt/continue_shared --device cuda:0 --steps 5000

# 完整前向、缓存前向、冻结主干、已知/预测相机检查
bash scripts/rgbt_python.sh scripts/verify_rgbt.py

# 正式模型的独立评价
bash scripts/rgbt_python.sh scripts/evaluate_rgbt.py \
  --output outputs/rgbt/evaluation --split val --views 3 6 --lpips
```

### 删除适配器前：RGBT-Scenes 与 ThermoScenes 的完整结果

2026-09-22 使用带适配器权重运行的结果：[离线画廊](../outputs/rgbt/evaluation_20260922/index.html) · [指标报告](../outputs/rgbt/evaluation_20260922/REPORT.md)。覆盖两个数据集的全部 20 个场景、267 个保留目标，3/6 源视图共 534 组。每个场景、每种协议导出 RGB/T 高斯 PLY、可旋转的高斯中心预览和视角动画。

`evaluate_rgbt.py` 支持 `--split all`、`--datasets RGBT-Scenes ThermoScenes` 和 `--targets 0`（全部保留目标）。例如：

```bash
bash scripts/rgbt_python.sh scripts/evaluate_rgbt.py \
  --data outputs/rgbt/data_336 --split all --datasets RGBT-Scenes \
  --targets 0 --views 3 6 --lpips --export --device cuda:0 \
  --output outputs/rgbt/new_evaluation/RGBT-Scenes
```

将 `--datasets` 换为 `ThermoScenes`，输出到同一父目录下的 `ThermoScenes`，即可用 `scripts/report_rgbt_datasets.py --output outputs/rgbt/new_evaluation` 生成汇总画廊。评测需使用新输出目录，逐目标指标同时写入 `rows.jsonl`。完整复现入口为 [run.sh](../outputs/rgbt/evaluation_20260922/run.sh)。

指标按场景等权汇总，并单独报告 train/val/test。这里使用已知 COLMAP 相机；nearest3 与 wide6 的源图组合不同。热指标衡量灰度／显示亮度。全部场景均值包含曾参与训练的场景，不能当作纯测试集性能。

测试场景的三方 PNG 对照单独保存在 `outputs/rgbt/test_comparison_20260922/`，仅含 Building、Parterre、raspberrypi、trees。全部 69 个测试目标 × 3/6 视图共 138 张图片，列顺序为真值、原始 DA3 权重 + RGB-T 包装、SharedRefiner；另存无标题 PNG 和场景总览，不生成网页。复现入口为 `scripts/visualize_rgbt_test.py --dataset RGBT-Scenes --output <新目录>`，另一数据集将 `--dataset` 换为 `ThermoScenes`。

测试对照图同时标注 PSNR、SSIM 和 LPIPS-AlexNet v0.1（越低越好）。LPIPS 在浮点预测及缓存真值上计算，使用 `normalize=True` 将 [0,1] 转为 [-1,1]；采用 float32 和 eval 模式。沿用现有评测口径，预测和真值均按同一个目标有效掩码将无效像素置零，再对特征距离取平均。热图复制为三通道后计算，衡量显示灰度图的感知差异，不是温度误差。逐图 CSV、场景汇总和 8 张总览均包含 LPIPS。

该脚本逐张量核对官方 DA3-GIANT-1.1 的 971 个张量，采用已有 `da3_rgbt_wrapper` 开关；关闭热适配器、学习残差、几何锚定、源 RGB 混合和全部细化。原始 DA3 没有原生热图输出，此处热图通过 DA3 几何采样源热图获得，保留源相机尺度对齐及原路径的前方深度约束。图片标题和目录说明均明确这一基线定义。

完整的训练范围、缓存、导出和重新评价命令见[共享模块说明](RGBT_LEARNED_JOINT.md)。`scripts/legacy/train_rgbt.py` 是旧三模块的表面微调入口，不用于当前共享网络。训练文件 `training_state.pt` 与完整推理文件 `model.pt` 用途不同。

测试场景的深度 PNG 单独保存在 [depth_test_20260922](../outputs/rgbt/depth_test_20260922/README.md)：138 张目标渲染深度对照、621 张源图深度／最终高斯中心深度对照，以及明确标注独立色标的结构预览和场景总览。参考来自数值 COLMAP geometric MVS，**不是传感器真值**；按相机内参对齐到缓存图像，撤销源基线归一化后使用 COLMAP 单位（非米）。主图统一场景色标；`*_relative` 使用各深度块独立色标，仅帮助观察结构。原始浮点深度和不透明度保存为 NPZ。复现脚本是 `scripts/visualize_rgbt_depth.py` 与 `scripts/report_rgbt_depth.py`，两数据集运行入口见该结果目录的 `run.sh`。

## 验证边界与历史

原三模块权重保存在 `outputs/rgbt/joint_training_20260920/before/model.pt`，旧代码继续支持严格加载，便于对照和回退。此前的[消融精简](../outputs/rgbt/ablation_20260920/REPORT.md)及[计算融合](RGBT_JOINT_REFINER.md)是历史结果；其“输出完全相等”结论不适用于本次重新训练的结构合并。

上述历史质量结论对应带适配器权重、已知相机及 252×336 / 336×252 输入。历史测试已多次访问，不能称为全新测试；单场景和单目标仍会退化。几何深度使用 COLMAP 稀疏点对照，平面厚度仅反映内部一致性。遮挡、非静态物体、配准误差和预测相机精度仍会影响重建。


单次前向视觉实验（空洞改善，但整体 RGB 尚未通过验收，默认权重未替换）见 [RGBT_VISUAL_QUALITY.md](RGBT_VISUAL_QUALITY.md)。
