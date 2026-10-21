import unittest
import torch
from torch import nn
from depth_anything_3.thermal.preservation import rgb_preservation_loss, frozen_base_forward
from depth_anything_3.thermal.adapter import LoRALinear, AdapterConfig


class PreservationTests(unittest.TestCase):
    def test_scene_scale_does_not_create_penalty(self):
        teacher = torch.linspace(1, 5, 400).reshape(1, 2, 10, 20)
        student = (teacher * 7).requires_grad_()
        loss = rgb_preservation_loss({"depth": student}, teacher,
                                    torch.ones_like(teacher, dtype=torch.bool),
                                    torch.tensor([[False, False]]))
        self.assertLess(float(loss), 1e-10)

    def test_inverted_invalid_region_is_penalized_and_thermal_excluded(self):
        teacher = torch.ones(1, 2, 20, 20)
        teacher[:, :, :5] = 5
        pred = teacher.clone();pred[:, :, :5] = .5;pred.requires_grad_()
        valid = torch.ones_like(teacher, dtype=torch.bool);valid[:, :, :5] = False
        loss = rgb_preservation_loss({"depth": pred}, teacher, valid, torch.tensor([[False, True]]))
        self.assertGreater(float(loss), 1)
        loss.backward()
        self.assertGreater(float(pred.grad[:, 0, :5].abs().sum()), 0)
        self.assertEqual(float(pred.grad[:, 1].abs().sum()), 0)

    def test_bypass_restored_after_exception(self):
        model = nn.Sequential(LoRALinear(nn.Linear(3, 2), AdapterConfig(rank=2)))
        model.train();scale = model[0].scale
        with self.assertRaises(RuntimeError):
            with frozen_base_forward(model):
                self.assertFalse(model.training)
                self.assertEqual(model[0].scale, 0)
                raise RuntimeError("probe")
        self.assertTrue(model.training);self.assertEqual(model[0].scale, scale)


if __name__ == '__main__':
    unittest.main()
