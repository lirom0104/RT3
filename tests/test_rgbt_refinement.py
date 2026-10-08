"""Geometric correctness and cross-view visibility for the learned refiner."""

import sys
import unittest
from dataclasses import fields, replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import torch

from depth_anything_3.rgbt.geometry import gaussian_depth, pixel_grid
from depth_anything_3.rgbt.model import RGBTOutput, render_rgbt
from depth_anything_3.rgbt.legacy.refinement import NeighborhoodRefiner
from depth_anything_3.rgbt.attention import ProjectedNeighborhood
from depth_anything_3.specs import Gaussians


def scene():
    b, v, h, w = 1, 2, 28, 28
    E = torch.eye(4)[None, None].repeat(b, v, 1, 1)
    E[:, 1, 0, 3] = -0.1
    K = torch.tensor([[35.0, 0.0, 12.0], [0.0, 34.0, 16.0], [0.0, 0.0, 1.0]])[
        None, None
    ].repeat(b, v, 1, 1)
    xy = pixel_grid(h, w, "cpu")[None, None].expand(b, v, -1, -1, -1)
    ray = torch.einsum(
        "bvij,bvhwj->bvhwi",
        K.inverse(),
        torch.cat([xy, torch.ones_like(xy[..., :1])], -1),
    )
    points = ray * 2 - E[..., :3, 3][:, :, None, None]
    n = v * h * w
    gs = Gaussians(
        means=points.reshape(b, n, 3),
        scales=torch.full((b, n, 3), 0.03),
        rotations=torch.tensor([1.0, 0.0, 0.0, 0.0]).expand(b, n, 4).clone(),
        harmonics=torch.zeros(b, n, 3, 9),
        opacities=torch.full((b, n), 0.5),
    )
    return RGBTOutput(
        gs,
        torch.full((b, n, 1), 0.5),
        torch.full((b, v, h, w), 2.0),
        E,
        K,
        torch.ones(b, v, h, w),
        torch.zeros(()),
        torch.randn(b, v, 64, h // 2, w // 2),
    )


class RefinementTests(unittest.TestCase):
    def build(self, **kwargs):
        o = scene()
        model = NeighborhoodRefiner(channels=16, blocks=1, **kwargs)
        return model, o

    def test_zero_update_preserves_geometry_and_learns(self):
        model, o = self.build()
        rgb = torch.rand(1, 2, 3, 28, 28)
        t = torch.rand(1, 2, 1, 28, 28)
        result = model(rgb, t, o.extrinsics, o.intrinsics, initial=o)
        for name in ["means", "scales", "rotations", "harmonics", "opacities"]:
            torch.testing.assert_close(
                getattr(result.gaussians, name),
                getattr(o.gaussians, name),
                atol=1e-6,
                rtol=1e-5,
            )
        torch.testing.assert_close(result.thermal_colors, o.thermal_colors)
        result.gaussians.scales.mean().backward()
        self.assertGreater(float(model.head[-1].weight.grad.abs().sum()), 0)

    def test_total_depth_bound_across_iterations(self):
        model, o = self.build(iterations=3, depth_limit=0.1)
        with torch.no_grad():
            model.head[-1].bias[2] = 10
        result = model(
            torch.rand(1, 2, 3, 28, 28), torch.rand(1, 2, 1, 28, 28), initial=o
        )
        ratio = gaussian_depth(result) / gaussian_depth(o)
        self.assertLessEqual(float(ratio.max().detach()), 1.10001)
        self.assertGreater(float(ratio.min().detach()), 1.09)

    def test_fixed_positions_allow_coverage_updates(self):
        model, o = self.build(
            support_context=True, depth_limit=0.0, position_strength=0.0
        )
        with torch.no_grad():
            model.head[-1].bias.fill_(0.5)
        result = model(
            torch.rand(1, 2, 3, 28, 28), torch.rand(1, 2, 1, 28, 28), initial=o
        )
        torch.testing.assert_close(
            result.gaussians.means, o.gaussians.means, atol=1e-6, rtol=1e-5
        )
        torch.testing.assert_close(result.gaussians.rotations, o.gaussians.rotations)
        self.assertGreater(
            float(result.gaussians.scales.mean().detach()),
            float(o.gaussians.scales.mean()),
        )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA source trust rendering")
    def test_source_guard_recovers_observed_opacity_without_target(self):
        model, o = self.build(
            depth_limit=0.0, position_strength=0.0, rgb_limit=0.0, source_guard=True
        )
        gs = replace(
            o.gaussians,
            **{
                f.name: getattr(o.gaussians, f.name).cuda()
                for f in fields(o.gaussians)
                if isinstance(getattr(o.gaussians, f.name), torch.Tensor)
            },
        )
        o = replace(
            o,
            **{
                f.name: getattr(o, f.name).cuda()
                for f in fields(o)
                if isinstance(getattr(o, f.name), torch.Tensor)
            },
            gaussians=gs,
        )
        updated = replace(
            o, gaussians=replace(gs, opacities=torch.full_like(gs.opacities, 0.9))
        )
        truth = replace(
            o, gaussians=replace(gs, opacities=torch.full_like(gs.opacities, 0.6))
        )
        observed = render_rgbt(truth, o.extrinsics, o.intrinsics, 28, 28)["rgb"]
        selected = model.guard_update(o, updated, observed)
        torch.testing.assert_close(
            selected.refinement_strength, torch.tensor([0.25], device="cuda")
        )
        torch.testing.assert_close(
            selected.gaussians.opacities, truth.gaussians.opacities
        )
        torch.testing.assert_close(selected.gaussians.means, o.gaussians.means)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA mixed precision")
    def test_mixed_precision_zero_update_does_not_move_points(self):
        model, o = self.build()
        gs = replace(
            o.gaussians,
            **{
                f.name: getattr(o.gaussians, f.name).cuda()
                for f in fields(o.gaussians)
                if isinstance(getattr(o.gaussians, f.name), torch.Tensor)
            },
        )
        o = replace(
            o,
            **{
                f.name: getattr(o, f.name).cuda()
                for f in fields(o)
                if isinstance(getattr(o, f.name), torch.Tensor)
            },
            gaussians=gs,
        )
        model = model.cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            result = model(
                torch.rand(1, 2, 3, 28, 28, device="cuda"),
                torch.rand(1, 2, 1, 28, 28, device="cuda"),
                initial=o,
            )
        torch.testing.assert_close(
            result.gaussians.means, gs.means, atol=1e-6, rtol=1e-5
        )

    def test_occluded_view_features_do_not_cross_surfaces(self):
        torch.manual_seed(3)
        module = ProjectedNeighborhood(16)
        x = torch.randn(1, 2, 16, 4, 4)
        E = torch.eye(4)[None, None].repeat(1, 2, 1, 1)
        K = torch.tensor([[4.0, 0.0, 2.0], [0.0, 4.0, 2.0], [0.0, 0.0, 1.0]])[
            None, None
        ].repeat(1, 2, 1, 1)
        xy = pixel_grid(4, 4, "cpu")[None, None].expand(1, 2, -1, -1, -1)
        ray = torch.einsum(
            "bvij,bvhwj->bvhwi",
            K.inverse(),
            torch.cat([xy, torch.ones_like(xy[..., :1])], -1),
        )
        points = ray * torch.tensor([2.0, 10.0])[None, :, None, None, None]
        a = module(x, points, E, K, (4, 4))
        other = x.clone()
        other[:, 1] = torch.randn_like(other[:, 1]) * 5
        b = module(other, points, E, K, (4, 4))
        torch.testing.assert_close(a[:, 0], b[:, 0])
        points = ray * 2
        a = module(x, points, E, K, (4, 4))
        b = module(other, points, E, K, (4, 4))
        self.assertGreater(float((a[:, 0] - b[:, 0]).abs().max().detach()), 0.01)


if __name__ == "__main__":
    unittest.main()
