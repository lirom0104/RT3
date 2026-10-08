# 单次前向视觉质量实验

**状态：空洞减少，整体 RGB 质量仍有退化，未通过完整验收。原默认权重没有替换。**

本次实现使用一次 DA3 backbone、一次 Gaussian head、一次 RGB-T 残差网络，输出一套高斯。恢复原生高斯的位置偏移和各向异性协方差，在高斯参数上预测有界残差。网络不读取目标图，不调用备用模型，不做图像拼接，也不在推理时优化。实际调用次数已在 3/6 视角下检查，RGB/T/alpha 由一次目标光栅化生成。

[完整报告](../outputs/rgbt/native_single_20260922/REPORT.md) · [138 张固定源图对照](../outputs/rgbt/native_single_20260922/index.html) · [调用次数审计](../outputs/rgbt/native_single_20260922/inference_audit/audit.json)

候选权重：`checkpoints/RGBT_single_forward/model.pt`。原默认模型仍为 `checkpoints/RGBT/model.pt`。候选通过统一加载器加载，推理脚本可显式传入 `--weights checkpoints/RGBT_single_forward/model.pt`。目前只验证了已知源相机、252×336 / 336×252 输入。

## 复现

```bash
# 缓存冻结的单次重建；train / val / test 场景严格分离
bash scripts/rgbt_python.sh scripts/cache_rgbt_native.py --split train --targets 6 --output <train_cache>
bash scripts/rgbt_python.sh scripts/cache_rgbt_native.py --split val --targets 4 --output <val_cache>

# 验证集选中的候选来自 run_evidence 第 500 步，不使用 test 选择权重
bash scripts/rgbt_python.sh scripts/train_rgbt_native.py \
  --train-cache <train_cache> --val-cache <val_cache> --output <run> \
  --steps 2000 --eval-every 500 --rgb-weight 2 --preservation-weight 2 \
  --bounds outputs/rgbt/native_single_20260922/evidence_bounds.json

bash scripts/rgbt_python.sh scripts/export_rgbt_native.py \
  --refiner <run>/best.pt --output <standalone_model.pt>

bash scripts/rgbt_python.sh scripts/verify_rgbt_single_forward.py \
  --weights <standalone_model.pt> \
  --cache outputs/rgbt/native_single_20260922/cache_test --output <audit>
```

训练损失直接监督保留的新视角图像，包含 RGB/T 图像损失、LPIPS、多尺度边缘、alpha 覆盖和局部 RGB 误差保持项。数据仅取训练场景，验证按场景和视角数等权选择。历史测试已多次检查，本轮属于回归评测。

旧版双前向渲染回退已经撤回，相关代码不再位于当前推理路径。其结果与说明仅存档在 `outputs/rgbt/quality_fix_20260922/`，不属于单次前向结果。
