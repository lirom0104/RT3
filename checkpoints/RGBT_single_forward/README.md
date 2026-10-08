# 单次前向实验候选：未通过完整视觉验收

此权重只调用一次 DA3 与一次共享残差网络。六视角低覆盖比例下降，但整体 RGB LPIPS 仍退化，不能作为已完成的修复。默认推理权重未切换。

选择：`outputs/rgbt/native_single_20260922/run_evidence/step_0500.pt`，依据两个 val 场景的场景/协议等权选择分数。训练使用 14 个 train 场景；未使用 test 目标参与梯度更新。

见 `outputs/rgbt/native_single_20260922/REPORT.md` 和 `docs/RGBT_VISUAL_QUALITY.md`。
