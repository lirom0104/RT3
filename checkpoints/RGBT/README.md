# RGB-T 单一共享细化模型

`model.pt` 是当前默认的完整推理权重，包含冻结的 DA3-Giant 重建器与训练后的 `LearnedJointRefiner`。四个 ViT 热适配器及其专用热特征层已从模型参数和 checkpoint 中移除；热图仍用于其他重建和细化路径。直接由 `load_rgbt_model` 严格加载。删除前的完整权重保存在 `model_with_adapters.pt`。

此次删除了 1,595,528 个参数，模型现有 1,357,177,512 个参数；共享细化器的 564,815 个参数未改动。

- `config.json`：嵌入模型配置的可读副本；加载以 `model.pt` 内部配置为准。
- `manifest.json`：SHA-256、参数量、验收和回退路径。
- `selection.json`：验证集选取 run_r/step_3500 的记录。
- `training_seed.pt`：该轮训练开始时的共享网络初始化，用于重新训练。
- `training_state.pt`：已交付模块第 3500 步的状态，含优化器与随机数状态，用于续训。
- `model_with_adapters.config.json` / `model_with_adapters.manifest.json`：删除前权重的配置和校验信息。

两个训练权重文件不是完整推理权重。细化网络共有 564,815 个参数，较原三个网络减少 35.0%；三个状态更新共用一个编码器与一个输出头。

在仓库根目录重新训练：

```bash
bash scripts/rgbt_python.sh scripts/train_rgbt_joint.py \
  --config configs/rgbt_joint_train.json \
  --output outputs/rgbt/retrain_shared --device cuda:0
```

[结构与训练说明](../../docs/RGBT_LEARNED_JOINT.md) · [删除前模型的 30 场景验收](../../outputs/rgbt/joint_release_20260920/REPORT.md) · [关闭适配器的对应推理](../../outputs/test-noadapter/results.json)

删除前的带适配器模型相比未加入三个细化模块的基础重建有明确平均收益，相比原三模块总体接近；这些完整评测指标尚未对删减版重跑。删减版通过严格加载、3/6 已知相机与 3 视图预测相机验证，在 Building img_025 上与关闭适配器的推理逐像素一致。更早的三模块教师权重保存在 `outputs/rgbt/joint_training_20260920/before/model.pt`。
