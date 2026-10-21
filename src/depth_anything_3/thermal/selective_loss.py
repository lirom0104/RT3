"""One-sided sky order constraints, geometry supervision and old-model replay."""
import torch
import torch.nn.functional as F
from .hybrid_loss import geometry_loss as hybrid_geometry,masked_mean,target_loss,confidence_preservation_loss
from .loss import geometry_loss as original_geometry

def sky_order_loss(pred,batch,scale):
    aligned=pred.float().clamp_min(1e-6)*scale[:,None,None,None]
    residual=(batch['sky_lower'].clamp_min(1e-6).log()[...,None,None]-aligned.log()).relu()
    return masked_mean(residual,batch['sky_mask'])

def repair_loss(pred,batch):
    loss,m=hybrid_geometry(pred,batch)
    scales=[]
    for p,g,valid in zip(pred['depth'].float(),batch['depth'],batch['hard_valid']):
        scales.append((g[valid].log()-p[valid].clamp_min(1e-6).log()).detach().median().exp())
    scale=torch.stack(scales);order=sky_order_loss(pred['depth'],batch,scale)
    loss=loss+2*order;m.update(sky_order=order.detach(),loss=loss.detach())
    return loss,m

def replay_loss(pred,batch,teacher):
    geometry,metrics=original_geometry(pred,batch)
    p=pred['depth'].float().clamp_min(1e-6);t=teacher['depth'].float().clamp_min(1e-6)
    scales=torch.stack([(tt.log()-pp.log()).detach().median().exp() for pp,tt in zip(p,t)])
    d,g=target_loss((p*scales[:,None,None,None]).log(),t,torch.isfinite(t)&(t>0))
    confidence=confidence_preservation_loss(pred,teacher['depth_conf'].float())
    loss=.2*geometry+4*(d+.5*g)+.2*confidence
    metrics.update(preservation=d.detach(),loss=loss.detach())
    return loss,metrics
