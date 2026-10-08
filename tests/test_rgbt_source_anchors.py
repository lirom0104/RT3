"""Local triangulation must preserve unsupported geometry and depth boundaries."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import torch

from depth_anything_3.rgbt.source_anchors import interpolate_anchor_field


class SourceAnchorTests(unittest.TestCase):
    def inputs(self):
        depth = torch.ones(1, 32, 32) * 2
        rgb = torch.ones(1, 3, 32, 32) * 0.5
        valid = torch.ones(1, 1, 32, 32)
        observations = [
            torch.tensor(
                [
                    [x + 0.5, y + 0.5, 2.2]
                    for x in range(6, 26, 4)
                    for y in range(6, 26, 4)
                ]
            )
        ]
        return depth, rgb, valid, observations

    def test_no_observations_is_identity(self):
        d, r, v, _ = self.inputs()
        field, count = interpolate_anchor_field(d, r, v, [[]])
        self.assertEqual(count, [0])
        self.assertEqual(float(field.abs().max()), 0)

    def test_consistent_anchors_correct_scale_with_bounded_shift(self):
        d, r, v, p = self.inputs()
        field, _ = interpolate_anchor_field(d, r, v, p)
        self.assertLess(float((d * field.exp() - 2.2).abs().mean()), 0.01)
        self.assertLessEqual(float(field.abs().max()), 0.12)

    def test_invalid_observations_and_pixels_cannot_move(self):
        d, r, v, p = self.inputs()
        v.zero_()
        field, count = interpolate_anchor_field(d, r, v, p)
        self.assertEqual(count, [0])
        self.assertEqual(float(field.abs().max()), 0)

    def test_depth_edge_blocks_cross_surface_propagation(self):
        d, r, v, p = self.inputs()
        d[:, :, 16:] = 5
        p = [p[0][p[0][:, 0] < 14]]
        field, _ = interpolate_anchor_field(d, r, v, p, radius=32)
        self.assertGreater(float(field[0, :, 4:12].mean()), 0.04)
        self.assertLess(float(field[0, :, 22:].abs().max()), 1e-6)

    def test_conflicting_depths_reduce_confidence(self):
        d, r, v, p = self.inputs()
        good, _ = interpolate_anchor_field(d, r, v, p)
        bad = p[0].clone()
        bad[::2, 2] = 1.8
        conflicted, _ = interpolate_anchor_field(d, r, v, [bad])
        self.assertLess(float(conflicted.abs().mean()), float(good.abs().mean()) * 0.1)


if __name__ == "__main__":
    unittest.main()
