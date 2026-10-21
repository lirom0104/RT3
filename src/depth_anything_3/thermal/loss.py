"""Joint depth/pose supervision in one common scale and first-camera coordinates.

SEAR labels are VGGT-derived pseudo labels. Their arbitrary scene scale differs
from DA3, so fit ONE detached positive scale per sequence for depth AND translation.
Do not independently align each frame: that would discard multi-view consistency.
"""

import torch
import torch.nn.functional as F
from depth_anything_3.utils.geometry import affine_inverse, as_homogeneous


def masked_mean(value, mask):
    return (value * mask).sum() / mask.sum().clamp_min(1)


def geometry_loss(prediction, batch):
    pred = prediction["depth"].float()
    gt, valid = batch["depth"].float(), batch["valid"]
    if pred.shape != gt.shape:
        raise ValueError(f"Depth shape mismatch: {pred.shape} vs {gt.shape}")
    if not torch.isfinite(pred).all() or not torch.isfinite(prediction["extrinsics"]).all():
        raise FloatingPointError("Non-finite model prediction")
    log_pred, log_gt = pred.clamp_min(1e-6).log(), gt.clamp_min(1e-6).log()
    scales = []
    for b in range(pred.shape[0]):
        scales.append((log_gt[b][valid[b]] - log_pred[b][valid[b]]).detach().median().exp())
    scale = torch.stack(scales)
    aligned_depth = pred * scale[:, None, None, None]
    residual = aligned_depth.clamp_min(1e-6).log() - log_gt
    depth_loss = masked_mean(F.smooth_l1_loss(residual, torch.zeros_like(residual), reduction="none", beta=.1), valid)
    gradient_loss = pred.new_zeros(())
    for stride in (1, 2, 4):
        r, m = residual[..., ::stride, ::stride], valid[..., ::stride, ::stride]
        gradient_loss = gradient_loss + masked_mean((r[..., 1:, :] - r[..., :-1, :]).abs(), m[..., 1:, :] & m[..., :-1, :])
        gradient_loss = gradient_loss + masked_mean((r[..., :, 1:] - r[..., :, :-1]).abs(), m[..., :, 1:] & m[..., :, :-1])
    gradient_loss = gradient_loss / 6

    # Both public DA3 extrinsics and dataset labels are w2c. Avoid comparing
    # DA3's raw c2w pose encoding against VGGT's w2c pose encoding.
    ext = as_homogeneous(prediction["extrinsics"].float())
    ext = ext @ affine_inverse(ext[:, :1])
    gt_ext = batch["extrinsics"].float()
    rotation = F.l1_loss(ext[..., :3, :3], gt_ext[..., :3, :3])
    translation = F.smooth_l1_loss(ext[..., :3, 3] * scale[:, None, None], gt_ext[..., :3, 3], beta=.1)
    h, w = gt.shape[-2:]
    sizes = pred.new_tensor([w, h])
    pred_focal = torch.stack([prediction["intrinsics"][..., 0, 0], prediction["intrinsics"][..., 1, 1]], -1)
    gt_focal = torch.stack([batch["intrinsics"][..., 0, 0], batch["intrinsics"][..., 1, 1]], -1)
    focal = F.l1_loss(2 * torch.atan(sizes / (2 * pred_focal.clamp_min(1e-6))), 2 * torch.atan(sizes / (2 * gt_focal)))
    camera = rotation + translation + .5 * focal
    loss = depth_loss + .5 * gradient_loss + 5 * camera
    with torch.no_grad():
        abs_rel = masked_mean((aligned_depth - gt).abs() / gt.clamp_min(1e-6), valid)
        relative_rotation = ext[..., :3, :3] @ gt_ext[..., :3, :3].mT
        cosine = ((relative_rotation.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(-1, 1)
        rot_deg = cosine[:, 1:].acos().rad2deg().mean()
    return loss, {"loss": loss.detach(), "depth": depth_loss.detach(), "depth_grad": gradient_loss.detach(),
                  "rotation": rotation.detach(), "translation": translation.detach(), "fov": focal.detach(),
                  "abs_rel": abs_rel, "rotation_deg": rot_deg, "scale": scale.detach().mean()}
