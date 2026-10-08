"""Geometric regression tests independent of the real scenes/checkpoints."""

import sys, unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import torch
from depth_anything_3.specs import Gaussians
from depth_anything_3.rgbt.types import RGBTOutput
from depth_anything_3.rgbt.multiview_depth import MultiViewDepthCorrector
from depth_anything_3.rgbt.geometry import gaussian_depth, pixel_grid


def synthetic():
    h, w = 42, 70
    v = 3
    K = torch.tensor([[65.0, 0, w / 2], [0, 65.0, h / 2], [0, 0, 1.0]]).repeat(v, 1, 1)
    E = torch.eye(4).repeat(v, 1, 1)
    E[:, 0, 3] = torch.tensor([0.0, -0.16, 0.16])
    xy = pixel_grid(h, w, "cpu")
    hom = torch.cat([xy, torch.ones(h, w, 1)], -1)
    rays = torch.einsum("vij,hwj->vhwi", torch.linalg.inv(K), hom)
    centers = torch.linalg.inv(E)[:, :3, 3]
    world = centers[:, None, None] + rays * 2
    x, y = world[..., 0], world[..., 1]
    rgb = torch.stack(
        [
            0.5 + 0.24 * torch.sin(x * 24 + y * 5),
            0.5 + 0.23 * torch.sin(x * 9 - y * 22),
            0.5 + 0.2 * torch.sin(x * 31 + y * 15),
        ],
        1,
    )[None]
    prior = torch.full((1, v, h, w), 3.1)
    means = (centers[:, None, None] + rays * prior[0, ..., None]).reshape(1, -1, 3)
    n = v * h * w
    rotation = torch.zeros(1, n, 4)
    rotation[..., 0] = 1
    gs = Gaussians(
        means,
        torch.full((1, n, 3), 0.02),
        rotation,
        torch.zeros(1, n, 3, 9),
        torch.full((1, n), 0.5),
    )
    out = RGBTOutput(
        gs,
        torch.zeros(1, n, 1),
        prior,
        E[None],
        K[None],
        torch.ones_like(prior),
        torch.tensor(0.0),
    )
    return rgb, out


class MultiViewDepthTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_search_recovers_depth_outside_old_ten_percent_gate(self):
        rgb, out = synthetic()
        m = MultiViewDepthCorrector(hypotheses=81, line_support=False)
        p = m.prepare(rgb, out)
        roi = p["depth"][..., 8:-8, 15:-15]
        self.assertLess(float((roi - 2).abs().median()), 0.07)
        fixed = m(rgb, out, prepared=p)
        self.assertLess(
            float((gaussian_depth(fixed)[..., 8:-8, 15:-15] - 2).abs().median()), 0.15
        )
        torch.testing.assert_close(
            fixed.depth, gaussian_depth(fixed), atol=2e-6, rtol=1e-6
        )

    def test_invalid_source_evidence_cannot_move_gaussians(self):
        rgb, out = synthetic()
        m = MultiViewDepthCorrector(hypotheses=17, line_support=False)
        fixed = m(rgb, out, torch.zeros_like(rgb[:, :, :1]))
        torch.testing.assert_close(fixed.gaussians.means, out.gaussians.means)

    def test_camera_coordinate_change_preserves_depth_decision(self):
        from dataclasses import replace

        rgb, out = synthetic()
        m = MultiViewDepthCorrector(hypotheses=33, line_support=False)
        p = m.prepare(rgb, out)
        # A shared world translation must not affect the corrected source camera Z.
        translation = torch.tensor([1.3, -2.1, 0.7])
        E = out.extrinsics.clone()
        E[..., :3, 3] -= translation
        moved = replace(
            out,
            extrinsics=E,
            gaussians=replace(out.gaussians, means=out.gaussians.means + translation),
        )
        q = m.prepare(rgb, moved)
        torch.testing.assert_close(p["depth"], q["depth"], atol=3e-4, rtol=1e-4)

    def test_learning_has_finite_geometry_gradients(self):
        rgb, out = synthetic()
        m = MultiViewDepthCorrector(hypotheses=17, learned=True, line_support=False)
        p = m.prepare(rgb, out)
        fixed = m(rgb, out, prepared=p)
        loss = fixed.gaussians.means.square().mean() + fixed.gaussians.opacities.mean()
        loss.backward()
        grad = m.decoder[-1].weight.grad
        self.assertTrue(torch.isfinite(grad).all())
        self.assertGreater(float(grad.abs().sum()), 0)

    def test_correction_preserves_source_ray_and_angular_footprint(self):
        rgb, out = synthetic()
        m = MultiViewDepthCorrector(hypotheses=33, line_support=False)
        fixed = m(rgb, out)
        centers = torch.linalg.inv(out.extrinsics)[..., :3, 3][:, :, None, None]
        shape = (*out.depth.shape, 3)
        before = (out.gaussians.means.reshape(shape) - centers) / gaussian_depth(out)[
            ..., None
        ]
        after = (fixed.gaussians.means.reshape(shape) - centers) / gaussian_depth(
            fixed
        )[..., None]
        torch.testing.assert_close(before, after, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(
            out.gaussians.scales.reshape(shape) / gaussian_depth(out)[..., None],
            fixed.gaussians.scales.reshape(shape) / gaussian_depth(fixed)[..., None],
        )
        torch.testing.assert_close(out.gaussians.opacities, fixed.gaussians.opacities)

    def test_consistency_requires_two_valid_other_cameras(self):
        rgb, out = synthetic()
        m = MultiViewDepthCorrector(hypotheses=17, line_support=False)
        p = m.prepare(rgb, out)
        p["depth"] = p["prior"].clone()
        p["valid"][:, 2] = 0
        single = m.independent_support(p, out, 1)
        two = m.independent_support(p, out, 2)
        self.assertGreater(float(single[:, 0, 8:-8, 15:-15].mean()), 0.99)
        self.assertEqual(float(two[:, 0].sum()), 0.0)

    def test_line_completion_respects_color_boundaries_and_invalid_pixels(self):
        from depth_anything_3.rgbt.line_consensus import refine_lines

        h, w = 90, 40
        rgb = torch.zeros(1, 3, h, w)
        rgb[:, :, :, 17:23] = 0.8
        depth = torch.full((1, h, w), 5.0)
        depth[:, :, 17:23] = 2.0
        conf = torch.zeros_like(depth)
        conf[:, 5:85, 17:23] = 0.9
        support = torch.ones_like(depth)
        valid = torch.ones_like(depth)
        # Missing matches inside a genuine straight object must be filled from its line.
        depth[:, 35:42, 17:23] = 3.0
        conf[:, 35:42, 17:23] = 0
        # Invalid source pixels must never become new supported points.
        valid[:, 55:59] = 0
        proposed, weight = refine_lines(rgb, depth, conf, support, valid)
        repaired = weight[:, 35:42, 17:23] > 0
        self.assertGreater(int(repaired.sum()), 12)
        torch.testing.assert_close(
            proposed[:, 35:42, 17:23][repaired],
            torch.full_like(proposed[:, 35:42, 17:23][repaired], 2.0),
            atol=1e-5,
            rtol=1e-5,
        )
        self.assertEqual(float(weight[:, 55:59].sum()), 0.0)
        self.assertEqual(float(weight[:, :, :14].sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
