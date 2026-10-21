"""Hard geometric constraints and separate, weak, quality-gated pseudo targets."""
from contextlib import contextmanager
import torch
import torch.nn.functional as F
from depth_anything_3.utils.geometry import affine_inverse, as_homogeneous


def masked_mean(value, mask):
    return torch.where(mask, value, 0).sum() / mask.sum().clamp_min(1)


def target_loss(log_prediction, target, valid):
    residual=log_prediction-target.clamp_min(1e-6).log()
    value=F.smooth_l1_loss(residual, torch.zeros_like(residual), beta=.1, reduction='none')
    depth=masked_mean(value,valid)
    grad=depth*0
    # Each source is differentiated separately; no artificial hard/soft seam targets.
    for stride in (1,2,4):
        r,m=residual[...,::stride,::stride],valid[...,::stride,::stride]
        grad+=masked_mean((r[...,1:,:]-r[...,:-1,:]).abs(),m[...,1:,:]&m[...,:-1,:])
        grad+=masked_mean((r[...,:,1:]-r[...,:,:-1]).abs(),m[...,:,1:]&m[...,:,:-1])
    return depth,grad/6


def geometry_loss(prediction,batch):
    pred=prediction['depth'].float();gt=batch['depth'].float()
    hard=batch['hard_valid'];soft=batch['soft_depth'].float()
    soft_valid=torch.isfinite(soft)&(soft>1e-6)
    if pred.shape!=gt.shape or soft.shape!=gt.shape:raise ValueError('Hybrid target shape mismatch')
    if not torch.isfinite(pred).all() or not torch.isfinite(prediction['extrinsics']).all():
        raise FloatingPointError('Non-finite geometry prediction')
    logp=pred.clamp_min(1e-6).log();scales=[]
    for b in range(len(pred)):
        mask=hard[b] if hard[b].sum()>=100 else soft_valid[b]
        target=gt[b] if hard[b].sum()>=100 else soft[b]
        if mask.sum()<100:raise ValueError('No reliable hybrid labels')
        scales.append((target[mask].log()-logp[b][mask]).detach().median().exp())
    scale=torch.stack(scales);aligned=pred*scale[:,None,None,None]
    log_aligned=aligned.clamp_min(1e-6).log()
    hd,hg=target_loss(log_aligned,gt,hard)
    sd,sg=target_loss(log_aligned,soft,soft_valid)
    depth=hd+.2*sd;gradient=hg+.2*sg
    ext=as_homogeneous(prediction['extrinsics'].float());ext=ext@affine_inverse(ext[:,:1])
    ge=batch['extrinsics'].float()
    rotation=F.l1_loss(ext[...,:3,:3],ge[...,:3,:3])
    translation=F.smooth_l1_loss(ext[...,:3,3]*scale[:,None,None],ge[...,:3,3],beta=.1)
    h,w=gt.shape[-2:];sizes=pred.new_tensor([w,h])
    pf=torch.stack([prediction['intrinsics'][...,0,0],prediction['intrinsics'][...,1,1]],-1)
    gf=torch.stack([batch['intrinsics'][...,0,0],batch['intrinsics'][...,1,1]],-1)
    focal=F.l1_loss(2*torch.atan(sizes/(2*pf.clamp_min(1e-6))),2*torch.atan(sizes/(2*gf)))
    camera=rotation+translation+.5*focal
    # Depth is the selection objective; low camera regularization keeps geometry coherent.
    camera_weight=1.0 if hard.any() else .2
    loss=depth+.5*gradient+camera_weight*camera
    with torch.no_grad():
        hard_rel=masked_mean((aligned-gt).abs()/gt.clamp_min(1e-6),hard)
        soft_rel=masked_mean((aligned-soft).abs()/soft.clamp_min(1e-6),soft_valid)
        rel=(hard_rel+.2*soft_rel)/(float(hard.any())+.2*float(soft_valid.any()))
        rr=ext[...,:3,:3]@ge[...,:3,:3].mT
        angle=((rr.diagonal(dim1=-2,dim2=-1).sum(-1)-1)/2).clamp(-1,1)[:,1:].acos().rad2deg().mean()
    return loss,{'loss':loss.detach(),'depth':depth.detach(),'depth_grad':gradient.detach(),
        'hard_depth':hd.detach(),'soft_depth':sd.detach(),'hard_abs_rel':hard_rel,'soft_abs_rel':soft_rel,
        'abs_rel':rel,'rotation':rotation.detach(),'translation':translation.detach(),'fov':focal.detach(),
        'rotation_deg':angle,'scale':scale.mean().detach()}


@contextmanager
def frozen_adapter_forward(model,teacher_weights):
    """Share frozen base weights; restore student parameters before its forward graph."""
    base=model.module if hasattr(model,'module') else model
    parameters=dict(base.named_parameters());was_training=base.training
    saved={n:parameters[n].detach().clone() for n in teacher_weights}
    try:
        with torch.no_grad():
            for n,v in teacher_weights.items():parameters[n].copy_(v)
        base.eval()
        yield base
    finally:
        with torch.no_grad():
            for n,v in saved.items():parameters[n].copy_(v)
        base.train(was_training)


def confidence_preservation_loss(prediction,teacher_conf):
    # The raw DA3 confidence is positive, uncalibrated, and has a lower floor at 1.
    # Preserve its spatial response rather than pretending it is a probability.
    return F.smooth_l1_loss(prediction['depth_conf'].float().clamp_min(1e-6).log(),
                            teacher_conf.clamp_min(1e-6).log(),beta=.1)


def rgb_preservation_loss(student,teacher_depth,valid,thermal_mask,valid_weight=.1):
    """Keep the absolute downweighting on labeled pixels, including dense VGGT scenes."""
    pred=student['depth'].float().clamp_min(1e-6)
    target=teacher_depth.float().clamp_min(1e-6)
    rgb=(~thermal_mask)[...,None,None].expand_as(pred)
    mask=rgb&torch.isfinite(pred)&torch.isfinite(target)
    losses=[]
    for b in range(len(pred)):
        if not mask[b].any():continue
        residual=pred[b].log()-target[b].log()
        reliable=mask[b]&valid[b]
        align=reliable if reliable.sum()>=100 else mask[b]
        residual=residual-residual[align].detach().median()
        weight=torch.where(valid[b],float(valid_weight),1.)*mask[b]
        value=F.smooth_l1_loss(residual,torch.zeros_like(residual),beta=.1,reduction='none')
        # Dividing by rgb count preserves .1 strength on dense, trusted target images.
        losses.append((value*weight).sum()/mask[b].sum().clamp_min(1))
    return torch.stack(losses).mean() if losses else pred.sum()*0
