import sys
from pathlib import Path
import unittest
import torch
from torch import nn
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from depth_anything_3.thermal.hybrid_loss import geometry_loss,target_loss,frozen_adapter_forward,rgb_preservation_loss

class HybridSupervisionTest(unittest.TestCase):
    def sample(self):
        depth=torch.ones(1,2,8,8)
        extr=torch.eye(4)[None,None].repeat(1,2,1,1)
        extr[:,1,0,3]=.2
        intr=torch.eye(3)[None,None].repeat(1,2,1,1);intr[...,0,0]=intr[...,1,1]=8
        pred={'depth':depth.clone().requires_grad_(),'extrinsics':extr.clone(),'intrinsics':intr.clone()}
        batch={'depth':depth.clone(),'soft_depth':depth.clone(),'hard_valid':torch.ones_like(depth,dtype=torch.bool),
               'extrinsics':extr.clone(),'intrinsics':intr.clone()}
        return pred,batch
    def test_joint_scene_scale(self):
        p,b=self.sample();p['depth']=p['depth']*7;p['extrinsics'][...,:3,3]*=7
        loss,m=geometry_loss(p,b);self.assertLess(float(loss.detach()),1e-6)
    def test_holes_have_no_gradient(self):
        p,b=self.sample();b['hard_valid'][...,:1,:]=False;b['soft_depth'][...,:1,:]=0
        with torch.no_grad():p['depth'][...,:1,:]=500
        loss,m=geometry_loss(p,b);loss.backward()
        self.assertEqual(float(p['depth'].grad[...,:1,:].abs().sum()),0)
    def test_bad_soft_does_not_change_hard_scale(self):
        p,b=self.sample();b['soft_depth']*=100
        _,m=geometry_loss(p,b);self.assertEqual(float(m['scale']),1.)
        self.assertEqual(float(m['hard_abs_rel']),0.)
    def test_soft_weight_is_not_normalized_away(self):
        p,b=self.sample();p['depth']=torch.cat((torch.ones(1,1,8,8),torch.full((1,1,8,8),2.)),1)
        _,both=geometry_loss(p,b);b['hard_valid'][:]=False
        _,soft=geometry_loss(p,b)
        self.assertAlmostEqual(float(soft['depth'])*6,float(both['depth']),places=6)
    def test_frozen_teacher_restores_before_student_backward(self):
        m=nn.Linear(2,1,bias=False);before=m.weight.detach().clone()
        for _ in range(2):
            with frozen_adapter_forward(m,{'weight':torch.ones_like(m.weight)}) as teacher,torch.no_grad():
                self.assertEqual(float(teacher(torch.ones(1,2))),2.)
            self.assertTrue(torch.equal(m.weight,before));m(torch.ones(1,2)).sum().backward()
        self.assertTrue(torch.equal(m.weight.grad,torch.full_like(m.weight,2.)))
    def test_dense_teacher_downweight_does_not_cancel(self):
        p,b=self.sample();target=b['depth'].clone();target[:,1]*=2
        mask=torch.zeros(1,2,dtype=torch.bool)
        a=rgb_preservation_loss(p,target,b['hard_valid'],mask,.1)
        z=rgb_preservation_loss(p,target,b['hard_valid'],mask,1.)
        self.assertAlmostEqual(float(a.detach())*10,float(z.detach()),places=6)

if __name__=='__main__':unittest.main()
