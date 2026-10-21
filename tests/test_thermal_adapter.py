import copy
import unittest

import torch
from torch import nn

from depth_anything_3.model.dinov2.vision_transformer import DinoVisionTransformer
from depth_anything_3.thermal.adapter import AdapterConfig, adapter_state_dict, install_adapter, load_adapter
from depth_anything_3.thermal.loss import geometry_loss


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Module()
        self.backbone.pretrained = DinoVisionTransformer(
            img_size=28, patch_size=14, embed_dim=24, depth=40, num_heads=3,
            ffn_layer="swiglufused", alt_start=13, rope_start=13, qknorm_start=13,
        )

    def forward(self, images, mask=None, strategy="first", cam_token=None):
        return self.backbone.pretrained.get_intermediate_layers(
            images, [19, 27, 33, 39], thermal_mask=mask, ref_view_strategy=strategy,
            cam_token=cam_token,
        )[0]


class AdapterTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        torch.set_num_threads(2)
        self.x = torch.randn(1, 3, 3, 28, 28)
        self.model = TinyModel()
        self.base_state = copy.deepcopy(self.model.state_dict())

    def test_initial_equivalence_and_roundtrip(self):
        self.model.eval()
        with torch.no_grad():
            before = self.model(self.x, strategy="middle")
        config = AdapterConfig(rank=4, alpha=8, dropout=0)
        install_adapter(self.model, config)
        self.model.eval()
        mask = torch.tensor([[True, False, True]])
        with torch.no_grad():
            after = self.model(self.x, mask, "middle")
        for p, q in zip(before, after):
            for a, b in zip(p, q):
                torch.testing.assert_close(a, b)
        with torch.no_grad():
            for n, p in self.model.named_parameters():
                if n.endswith("lora_B"):
                    p.normal_(std=.01)
        checkpoint = {"adapter_config": self.model.thermal_adapter_config,
                      "adapter": adapter_state_dict(self.model)}
        restored = TinyModel()
        restored.load_state_dict(self.base_state)
        load_adapter(restored, checkpoint)
        restored.eval()
        for p, q in zip(self.model(self.x, mask), restored(self.x, mask)):
            for a, b in zip(p, q):
                torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_reorder_and_backward_with_checkpointing(self):
        install_adapter(self.model, AdapterConfig(rank=4, alpha=8, dropout=0))
        backbone = self.model.backbone.pretrained
        mask = torch.tensor([[True, True, False]])
        with torch.no_grad():
            backbone.thermal_camera_token.add_(2)
        captured = []
        handle = backbone.blocks[13].register_forward_pre_hook(lambda m, args: captured.append(args[0].detach().clone()))
        outputs = self.model(self.x, mask, "middle")
        handle.remove()
        # middle strategy orders original views [1, 0, 2]. The first two are thermal.
        tokens = captured[0].reshape(1, 3, 5, 24)[:, :, 0]
        expected = torch.stack([backbone.thermal_camera_token[0, 0],
                                backbone.thermal_camera_token[0, 1], backbone.camera_token[0, 1]])[None]
        torch.testing.assert_close(tokens, expected)
        sum(p.square().mean() + c.square().mean() for p, c in outputs).backward()
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                self.assertIsNotNone(p.grad, n)
                self.assertTrue(torch.isfinite(p.grad).all(), n)
                if n.endswith(("lora_B", "thermal_camera_token")):
                    self.assertGreater(p.grad.norm().item(), 0, n)
            else:
                self.assertIsNone(p.grad, n)

    def test_initial_adapter_preserves_provided_camera_conditioning(self):
        self.model.eval()
        cameras = torch.randn(1, 3, 24)
        mask = torch.tensor([[True, False, True]])
        with torch.no_grad():
            before = self.model(self.x, cam_token=cameras)
        install_adapter(self.model, AdapterConfig(rank=4, dropout=0))
        self.model.eval()
        with torch.no_grad():
            after = self.model(self.x, mask, cam_token=cameras)
        for p, q in zip(before, after):
            for a, b in zip(p, q):
                torch.testing.assert_close(a, b)

    def test_thermal_offset_keeps_calibrated_tokens_and_rgb_unchanged(self):
        install_adapter(self.model, AdapterConfig(rank=4, dropout=0))
        self.model.eval()
        backbone = self.model.backbone.pretrained
        cameras = torch.randn(1, 3, 24)
        original = cameras.clone()
        mask = torch.tensor([[True, False, True]])
        with torch.no_grad():
            backbone.thermal_camera_token.add_(2)
        captured = []
        handle = backbone.blocks[13].register_forward_pre_hook(
            lambda m, args: captured.append(args[0].detach().clone()))
        with torch.no_grad():
            self.model(self.x, mask, cam_token=cameras)
        handle.remove()
        tokens = captured[0].reshape(1, 3, 5, 24)[:, :, 0]
        torch.testing.assert_close(tokens, cameras + mask[..., None] * 2)
        torch.testing.assert_close(cameras, original, rtol=0, atol=0)

    def test_invalid_mask(self):
        with self.assertRaises(ValueError):
            self.model(self.x, torch.zeros(1, 3, dtype=torch.bool))
        install_adapter(self.model, AdapterConfig(rank=4))
        with self.assertRaises(ValueError):
            self.model(self.x, torch.zeros(1, 2, dtype=torch.bool))


class LossTests(unittest.TestCase):
    def test_common_scale_and_reference_invariance(self):
        depth = torch.rand(1, 3, 14, 14) + 1
        ext = torch.eye(4).repeat(1, 3, 1, 1)
        ext[0, :, 0, 3] = torch.tensor([0., .2, .4])
        intr = torch.tensor([[20., 0, 7], [0, 20., 7], [0, 0, 1]]).repeat(1, 3, 1, 1)
        batch = {"depth": depth, "valid": torch.ones_like(depth, dtype=torch.bool), "extrinsics": ext, "intrinsics": intr}
        scaled = ext.clone()
        scaled[..., :3, 3] /= 3
        scaled[..., :3, 3] += torch.tensor([.7, -.2, .3])
        prediction = {"depth": (depth / 3).requires_grad_(), "extrinsics": scaled.requires_grad_(), "intrinsics": intr.clone().requires_grad_()}
        loss, metrics = geometry_loss(prediction, batch)
        self.assertLess(loss.item(), 1e-5)
        self.assertLess(metrics["abs_rel"].item(), 1e-6)
        loss.backward()
        for p in prediction.values():
            self.assertTrue(torch.isfinite(p.grad).all())
        # An inconsistent depth scale between views must be penalized.
        wrong = dict(prediction)
        wrong["depth"] = prediction["depth"] * torch.tensor([1., 2., 1.])[None, :, None, None]
        wrong_loss, _ = geometry_loss(wrong, batch)
        self.assertGreater(wrong_loss.item(), .1)


if __name__ == "__main__":
    unittest.main()
