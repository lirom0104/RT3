"""Optional frozen-base RGB regularization for noisy/incomplete supervision."""
from contextlib import contextmanager
import torch
import torch.nn.functional as F
from depth_anything_3.thermal.adapter import LoRALinear


@contextmanager
def frozen_base_forward(model):
    """Temporarily bypass trainable updates; do this before the student forward."""
    base = model.module if hasattr(model, "module") else model
    modules = [(m, m.scale) for m in base.modules() if isinstance(m, LoRALinear)]
    training = base.training
    try:
        base.eval()
        for module, _ in modules:
            module.scale = 0.0
        yield base
    finally:
        for module, scale in modules:
            module.scale = scale
        base.train(training)


def rgb_preservation_loss(student, teacher_depth, valid, thermal_mask, valid_weight=.1):
    """One positive scale per sequence; emphasize RGB holes lacking geometry labels."""
    pred = student["depth"].float().clamp_min(1e-6)
    target = teacher_depth.float().clamp_min(1e-6)
    rgb = (~thermal_mask)[..., None, None].expand_as(pred)
    finite = torch.isfinite(pred) & torch.isfinite(target)
    mask = rgb & finite
    losses = []
    for b in range(pred.shape[0]):
        if not mask[b].any():
            continue
        residual = pred[b].log() - target[b].log()
        reliable = mask[b] & valid[b]
        align = reliable if reliable.sum() >= 100 else mask[b]
        offset = residual[align].detach().median()
        residual = residual - offset
        weight = torch.where(valid[b], float(valid_weight), 1.0) * mask[b]
        loss = F.smooth_l1_loss(residual, torch.zeros_like(residual), beta=.1, reduction="none")
        losses.append((loss * weight).sum() / weight.sum().clamp_min(1))
    return torch.stack(losses).mean() if losses else pred.sum() * 0
