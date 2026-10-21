"""Quality-gated RGB teacher boundaries; labels retain the scene geometry."""
import torch
import torch.nn.functional as F


def rgb_teacher_inputs(batch):
    """Gather RGB views without treating thermal images as RGB teacher inputs."""
    if batch["images"].shape[0] != 1:
        raise ValueError("Variable RGB view counts require batch size one")
    indices = (~batch["thermal_mask"][0]).nonzero(as_tuple=True)[0]
    return indices, batch["images"][:, indices]


def scatter_rgb_teacher(batch, indices, output):
    depth = torch.ones_like(batch["depth"], dtype=torch.float32)
    confidence = torch.zeros_like(depth)
    depth[:, indices] = output["depth"].detach().float()
    confidence[:, indices] = output["depth_conf"].detach().float()
    return depth, confidence


def _dilate(mask, radius):
    shape = mask.shape
    return F.max_pool2d(mask.float().reshape(-1, 1, *shape[-2:]),
                        radius * 2 + 1, stride=1, padding=radius).reshape(shape).bool()


def rgb_boundary_preservation(prediction, teacher_depth, teacher_conf, batch,
                              edge_threshold=.03, radius=5, confidence_quantile=.1,
                              colmap_tolerance=.2, thin_weight=2.0, return_focus=False):
    """Match log-depth gradients near reliable teacher edges, at two scales.

    Confidence is an uncalibrated filter, not a correctness probability. Reject
    teacher pixels that conflict with valid COLMAP after ONE sequence scale fit.
    Weak VGGT labels are not used as a hard veto for fine teacher structures.
    A small local foreground trough is a thin-structure proxy, not semantic GT.
    """
    pred = prediction["depth"].float()
    target = teacher_depth.detach().float()
    conf = teacher_conf.detach().float()
    if pred.shape != target.shape or conf.shape != target.shape:
        raise ValueError("Teacher/student depth or confidence shapes disagree")
    if not 0 <= confidence_quantile < 1 or radius < 0:
        raise ValueError("Invalid confidence quantile or boundary radius")
    rgb = (~batch["thermal_mask"])[..., None, None].expand_as(pred)
    finite = torch.isfinite(target) & (target > 1e-6) & torch.isfinite(conf)
    safe = torch.where(finite, target, 1.0).clamp_min(1e-6)
    log_target = safe.log()
    reliable = rgb & finite
    for b in range(len(pred)):
        for v in range(pred.shape[1]):
            mask = reliable[b, v]
            if mask.any():
                cutoff = torch.quantile(conf[b, v][mask], confidence_quantile)
                reliable[b, v] &= conf[b, v] >= cutoff
        hard = batch["colmap_mask"][b, :, None, None] & batch["valid"][b] & reliable[b]
        if hard.sum() >= 100:
            gt = batch["depth"][b].float().clamp_min(1e-6).log()
            offset = (gt[hard] - log_target[b][hard]).median()
            conflict = hard & ((log_target[b] + offset - gt).abs() > colmap_tolerance)
            reliable[b] &= ~conflict
    # Mark both endpoints; finite differences must never cross invalid pixels.
    edges = torch.zeros_like(reliable)
    gx = (log_target[..., :, 1:] - log_target[..., :, :-1]).abs() > edge_threshold
    gy = (log_target[..., 1:, :] - log_target[..., :-1, :]).abs() > edge_threshold
    gx &= reliable[..., :, 1:] & reliable[..., :, :-1]
    gy &= reliable[..., 1:, :] & reliable[..., :-1, :]
    edges[..., :, 1:] |= gx
    edges[..., :, :-1] |= gx
    edges[..., 1:, :] |= gy
    edges[..., :-1, :] |= gy
    focus = _dilate(edges, radius) & reliable
    # A foreground dip against neighbors on opposite sides captures narrow rods.
    thin = torch.zeros_like(reliable)
    for distance in (2, 4):
        center = log_target[..., :, distance:-distance]
        thin[..., :, distance:-distance] |= (
            (log_target[..., :, :-2*distance] - center > edge_threshold) &
            (log_target[..., :, 2*distance:] - center > edge_threshold))
        center = log_target[..., distance:-distance, :]
        thin[..., distance:-distance, :] |= (
            (log_target[..., :-2*distance, :] - center > edge_threshold) &
            (log_target[..., 2*distance:, :] - center > edge_threshold))
    thin = _dilate(thin & reliable, 2) & focus
    weight = focus.float() * (1 + thin_weight * thin.float())
    log_pred = pred.clamp_min(1e-6).log()
    losses = []
    for stride in (1, 2):
        r = log_pred[..., ::stride, ::stride] - log_target[..., ::stride, ::stride]
        m = focus[..., ::stride, ::stride]
        w = weight[..., ::stride, ::stride]
        for axis in (-1, -2):
            if axis == -1:
                residual = r[..., :, 1:] - r[..., :, :-1]
                valid = m[..., :, 1:] & m[..., :, :-1]
                pair_weight = (w[..., :, 1:] + w[..., :, :-1]) * .5
            else:
                residual = r[..., 1:, :] - r[..., :-1, :]
                valid = m[..., 1:, :] & m[..., :-1, :]
                pair_weight = (w[..., 1:, :] + w[..., :-1, :]) * .5
            pair_weight = pair_weight * valid
            losses.append((residual.abs() * pair_weight).sum() / pair_weight.sum().clamp_min(1))
    loss = torch.stack(losses).mean()
    count = rgb.sum().clamp_min(1)
    metrics = {"rgb_boundary": loss.detach(),
               "teacher_boundary_fraction": focus.sum().float() / count,
               "teacher_thin_fraction": thin.sum().float() / count}
    return (loss, metrics, focus) if return_focus else (loss, metrics)


def relax_pseudo_edges(batch, focus):
    """Remove conflicting hard pseudo supervision near trusted RGB teacher edges.

    COLMAP valid pixels and thermal supervision are never removed. Retain enough
    geometry pixels to fit the original common depth/translation sequence scale.
    Validation continues to use the original supervision without this relaxation.
    """
    rgb = (~batch["thermal_mask"])[..., None, None]
    weak = ~batch["colmap_mask"][..., None, None]
    relaxed = focus & rgb & weak & batch["valid"]
    valid = batch["valid"] & ~relaxed
    for b in range(len(valid)):
        if valid[b].sum() < 100:
            relaxed[b] = False
            valid[b] = batch["valid"][b]
    return dict(batch, valid=valid), relaxed


def local_rgb_depth_preservation(prediction, teacher_depth, focus, batch):
    """Keep teacher depths at uncertain fine structures after one RGB scale fit."""
    pred = prediction["depth"].float().clamp_min(1e-6)
    teacher = teacher_depth.detach().float().clamp_min(1e-6)
    rgb = (~batch["thermal_mask"])[..., None, None].expand_as(pred)
    residual = pred.log() - teacher.log()
    losses = []
    for b in range(len(pred)):
        mask = focus[b] & rgb[b]
        if not mask.any():
            continue
        alignment = rgb[b] & batch["valid"][b] & ~focus[b]
        if alignment.sum() < 100:
            alignment = rgb[b]
        offset = residual[b][alignment].detach().median()
        value = F.smooth_l1_loss(residual[b] - offset, torch.zeros_like(residual[b]), beta=.1, reduction="none")
        losses.append(value[mask].mean())
    return torch.stack(losses).mean() if losses else pred.sum() * 0
