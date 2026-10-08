"""Removing unreachable recurrent parameters preserves learned inference."""

import sys
import unittest
from dataclasses import fields
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import torch
from test_rgbt_refinement import scene

from depth_anything_3.rgbt.legacy.crossview import CrossViewRefiner
from depth_anything_3.rgbt.legacy.refinement import NeighborhoodRefiner


class CompactTests(unittest.TestCase):
    def test_one_iteration_removes_state_without_changing_trained_update(self):
        torch.manual_seed(102)
        config = {
            "channels": 16,
            "blocks": 1,
            "iterations": 1,
            "render_feedback": False,
        }
        original = NeighborhoodRefiner(**config).eval()
        compact = NeighborhoodRefiner(**config, compact=True).eval()
        with torch.no_grad():
            original.head[-1].weight.normal_(0, 0.01)
            original.head[-1].bias.normal_(0, 0.01)
        compact.load_state_dict(
            {
                k: v
                for k, v in original.state_dict().items()
                if not k.startswith("state_conv.")
            },
            strict=True,
        )
        self.assertIsNone(compact.state_conv)
        self.assertTrue(compact.config["compact"])
        rgb, thermal = torch.rand(1, 2, 3, 28, 28), torch.rand(1, 2, 1, 28, 28)
        initial = scene()
        with torch.no_grad():
            before = original(rgb, thermal, initial=initial)
            after = compact(rgb, thermal, initial=initial)
        for field in fields(before.gaussians):
            x, y = (
                getattr(before.gaussians, field.name),
                getattr(after.gaussians, field.name),
            )
            if isinstance(x, torch.Tensor):
                torch.testing.assert_close(x, y, rtol=0, atol=0)
        torch.testing.assert_close(
            before.thermal_colors, after.thermal_colors, rtol=0, atol=0
        )

    def test_recurrent_configuration_retains_its_state_parameters(self):
        model = NeighborhoodRefiner(channels=16, blocks=1, iterations=2, compact=True)
        self.assertIsNotNone(model.state_conv)
        self.assertIn("state_conv.weight", model.state_dict())

    def test_crossview_serializes_compact_configuration(self):
        model = CrossViewRefiner(channels=16, blocks=1, compact=True)
        self.assertIsNone(model.state_conv)
        reconstructed = CrossViewRefiner(**model.config)
        reconstructed.load_state_dict(model.state_dict(), strict=True)
        self.assertIsNone(reconstructed.state_conv)


if __name__ == "__main__":
    unittest.main()
