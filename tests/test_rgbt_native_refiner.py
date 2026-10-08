"""The single-forward path preserves the native representation and learns it."""

import unittest
from dataclasses import replace
from unittest.mock import patch

import torch
from test_rgbt_refinement import scene
from torch import nn

from depth_anything_3.rgbt.model import RGBTGaussianNetwork
from depth_anything_3.rgbt.native_refiner import NativeGaussianRefiner


class NativeRefinerTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(3)
        self.initial = scene()
        self.rgb = torch.rand(1, 2, 3, 28, 28)
        self.thermal = torch.rand(1, 2, 1, 28, 28)
        self.refiner = NativeGaussianRefiner(channels=16, attention_pairs=1)

    def test_zero_update_preserves_offset_and_anisotropic_covariance(self):
        gs = self.initial.gaussians
        scales = gs.scales.clone()
        scales[..., 0], scales[..., 1], scales[..., 2] = 1e-5, 0.03, 2.0
        # Keep deliberately displaced, thin splats; a pixel clamp would fail.
        initial = replace(
            self.initial, gaussians=replace(gs, scales=scales, means=gs.means + 0.04)
        )
        output = self.refiner(self.rgb, self.thermal, initial=initial)
        for name in ["means", "scales", "rotations", "harmonics", "opacities"]:
            torch.testing.assert_close(
                getattr(output.gaussians, name),
                getattr(initial.gaussians, name),
                atol=1e-6,
                rtol=1e-5,
            )
        torch.testing.assert_close(output.thermal_colors, initial.thermal_colors)

    def test_network_calls_reconstruction_and_residual_exactly_once_without_render(
        self,
    ):
        class Reconstruction(nn.Module):
            def forward(inner, *args):
                inner.calls += 1
                return self.initial

        model = RGBTGaussianNetwork.__new__(RGBTGaussianNetwork)
        nn.Module.__init__(model)
        model.reconstruction, model.refiner = Reconstruction(), self.refiner
        model.reconstruction.calls = 0
        calls = []
        handle = self.refiner.register_forward_hook(lambda *args: calls.append(1))
        with patch(
            "depth_anything_3.rgbt.gaussian_update.render_rgbt",
            side_effect=AssertionError("Inference must not render"),
        ):
            result = model(
                self.rgb, self.thermal, self.initial.extrinsics, self.initial.intrinsics
            )
        handle.remove()
        self.assertEqual(model.reconstruction.calls, 1)
        self.assertEqual(len(calls), 1)
        self.assertEqual(
            result.gaussians.means.shape, self.initial.gaussians.means.shape
        )
        self.assertFalse(hasattr(result, "render_reference"))

    def test_coverage_and_appearance_parameters_have_gradients(self):
        output = self.refiner(self.rgb, self.thermal, initial=self.initial)
        loss = (
            output.gaussians.scales.mean()
            + output.gaussians.opacities.mean()
            + output.gaussians.harmonics[..., 0].mean()
            + output.thermal_colors.mean()
        )
        loss.backward()
        grad = self.refiner.core.head[-1].bias.grad
        for indices in [[3, 4, 5], [10], [11, 12, 13], [14]]:
            self.assertGreater(float(grad[indices].abs().sum()), 0)


if __name__ == "__main__":
    unittest.main()
