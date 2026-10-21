import unittest
import torch
from depth_anything_3.thermal.selective_loss import sky_order_loss
class SkyConstraintTests(unittest.TestCase):
 def test_wrong_sky_has_upward_gradient_only_on_mask(self):
  p=torch.tensor([[[[.5,1.],[2.,3.]]]],requires_grad=True)
  b={'sky_lower':torch.tensor([[2.]]),'sky_mask':torch.tensor([[[[True,False],[False,False]]]])}
  l=sky_order_loss(p,b,torch.ones(1));l.backward()
  self.assertGreater(float(l),0);self.assertLess(float(p.grad[0,0,0,0]),0);self.assertEqual(int((p.grad!=0).sum()),1)
 def test_far_sky_unconstrained_and_common_scale_invariant(self):
  b={'sky_lower':torch.tensor([[2.]]),'sky_mask':torch.ones(1,1,2,2,dtype=torch.bool)}
  self.assertEqual(float(sky_order_loss(torch.full((1,1,2,2),3.),b,torch.ones(1))),0)
  p=torch.full((1,1,2,2),.5)
  self.assertAlmostEqual(float(sky_order_loss(p,b,torch.ones(1))),float(sky_order_loss(p*7,b,torch.ones(1)/7)),places=6)
if __name__=='__main__':unittest.main()
