import copy
import unittest

import torch
from torch import nn

from depth_anything_3.thermal.adapter import (
    AdapterConfig, LoRALinear, adapter_state_dict, install_adapter, load_adapter,
)
from test_thermal_adapter import TinyModel


class ThermalOnlyLoRATests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        torch.set_num_threads(2)
        self.images = torch.randn(2, 3, 3, 28, 28)
        self.mask = torch.tensor([[True, False, True], [False, True, False]])

    def adapted(self, *, dropout=0):
        model = TinyModel()
        install_adapter(model, AdapterConfig(rank=4, alpha=8, dropout=dropout,
                                             lora_scope="thermal_only"))
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if name.endswith("lora_B"):
                    parameter.normal_(std=.03)
            model.backbone.pretrained.thermal_camera_token.add_(.3)
        return model

    @staticmethod
    def objective(outputs):
        return sum(p[..., 0].square().mean() + c.square().mean() for p, c in outputs)

    def test_rgb_has_no_linear_update_or_adapter_gradient(self):
        layer = LoRALinear(nn.Linear(5, 7), AdapterConfig(rank=2, dropout=0,
                                                       lora_scope="thermal_only"))
        with torch.no_grad():
            layer.lora_B.normal_()
        inputs = torch.randn(2, 3, 5)
        mask = self.mask
        result = layer(inputs, lora_mask=mask)
        base = layer.base(inputs)
        torch.testing.assert_close(result[~mask], base[~mask], rtol=0, atol=0)
        self.assertGreater((result[mask] - base[mask]).abs().max().item(), 0)
        result[~mask].square().sum().backward()
        for parameter in (layer.lora_A, layer.lora_B):
            self.assertIsNotNone(parameter.grad)
            self.assertEqual(parameter.grad.count_nonzero().item(), 0)
        with self.assertRaises(ValueError):
            layer(inputs)

    def test_trained_adapter_preserves_rgb_only_backbone(self):
        model = self.adapted().eval()
        base = TinyModel().eval()
        # Recover the frozen pretrained weights from the adapted model.
        state = {name.replace(".base.", "."): value for name, value in model.state_dict().items()
                 if not name.endswith(("lora_A", "lora_B", "thermal_camera_token"))}
        base.load_state_dict(state, strict=True)
        with torch.no_grad():
            expected = base(self.images, strategy="middle")
            for mask in (None, torch.zeros_like(self.mask)):
                actual = model(self.images, mask, "middle")
                for before, after in zip(expected, actual):
                    for p, q in zip(before, after):
                        torch.testing.assert_close(p, q, rtol=0, atol=0)

    def test_local_and_global_routes_follow_reordered_views(self):
        model = self.adapted().eval()
        blocks = model.backbone.pretrained.blocks
        captured = {}
        handles = []
        for index in (13, 14):
            modules = (blocks[index].attn.qkv, blocks[index].attn.proj,
                       blocks[index].mlp.w12, blocks[index].mlp.w3)
            for number, module in enumerate(modules):
                def capture(module, args, kwargs, key=(index, number)):
                    captured[key] = kwargs["lora_mask"].clone()
                handles.append(module.register_forward_pre_hook(capture, with_kwargs=True))
        try:
            with torch.no_grad():
                model(self.images, self.mask, "middle")
        finally:
            for handle in handles:
                handle.remove()
        reordered = self.mask[:, [1, 0, 2], None].expand(2, 3, 5)
        self.assertEqual(len(captured), 8)
        for (index, _), actual in captured.items():
            shape = (2, 15) if index == 13 else (6, 5)
            torch.testing.assert_close(actual, reordered.reshape(shape))

    def test_checkpoint_recomputation_keeps_each_forward_route(self):
        checkpointed = self.adapted(dropout=.1).train()
        ordinary = copy.deepcopy(checkpointed)
        ordinary.backbone.pretrained.gradient_checkpointing = False
        for model in (checkpointed, ordinary):
            torch.manual_seed(19)
            first = model(self.images, self.mask, "middle")
            # A second forward must not overwrite the first checkpoint's route.
            second = model(self.images * .8, ~self.mask, "first")
            (self.objective(first) + self.objective(second)).backward()
        other_parameters = dict(ordinary.named_parameters())
        for name, parameter in checkpointed.named_parameters():
            if not parameter.requires_grad:
                self.assertIsNone(parameter.grad, name)
                continue
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            torch.testing.assert_close(parameter.grad, other_parameters[name].grad,
                                       rtol=2e-4, atol=1e-6, msg=name)

    def test_rgb_loss_in_mixed_input_can_train_thermal_lora(self):
        model = self.adapted().train()
        outputs = model(self.images, self.mask)
        patches, cameras = outputs[-1]
        (patches[~self.mask][..., 0].square().mean()
         + cameras[~self.mask].square().mean()).backward()
        gradient = model.backbone.pretrained.blocks[13].attn.qkv.lora_B.grad
        self.assertGreater(gradient.norm().item(), 0)

    def test_roundtrip_and_legacy_shared_checkpoint(self):
        model = self.adapted().eval()
        checkpoint = {"adapter_config": model.thermal_adapter_config,
                      "adapter": adapter_state_dict(model)}
        restored = TinyModel()
        base_state = {name.replace(".base.", "."): value
                      for name, value in model.state_dict().items()
                      if not name.endswith(("lora_A", "lora_B", "thermal_camera_token"))}
        restored.load_state_dict(base_state)
        load_adapter(restored, checkpoint)
        restored.eval()
        for before, after in zip(model(self.images, self.mask), restored(self.images, self.mask)):
            for p, q in zip(before, after):
                torch.testing.assert_close(p, q, rtol=0, atol=0)
        shared = TinyModel()
        install_adapter(shared, AdapterConfig(rank=4))
        legacy_config = dict(shared.thermal_adapter_config)
        del legacy_config["lora_scope"]
        load_adapter(TinyModel(), {"adapter_config": legacy_config,
                                   "adapter": adapter_state_dict(shared)})
        with self.assertRaisesRegex(ValueError, "configuration"):
            load_adapter(shared, checkpoint)


if __name__ == "__main__":
    unittest.main()
