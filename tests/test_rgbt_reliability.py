"""Surface rejection must protect the complete geometry and observation path."""

import sys
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import test_rgbt_surface as surface_tests
import torch
from test_rgbt_refinement import scene

from depth_anything_3.rgbt.geometry import gaussian_depth
from depth_anything_3.rgbt.legacy.surface import SurfaceRefiner
from depth_anything_3.rgbt.surface_reliability import (
    guard_surface,
    projection_association,
    select_geometry,
    source_color_residual,
    visible_appearance,
)


class ReliabilityTests(unittest.TestCase):
    def model(self, output, **kwargs):
        return SurfaceRefiner(channels=16, **kwargs)

    def test_rejected_projection_preserves_complete_surface(self):
        o, a = surface_tests.SurfaceTests().setup_scene()
        a["offsets"] = torch.tensor([5.0])
        model = self.model(
            o, geometry_guard=False, plane_cov=True, thermal_fusion=1, rgb_fusion=0.5
        )
        p, used = model.surface(
            o, a, torch.full((1, 2, 3, 28, 28), 0.8), torch.ones(1, 2, 1, 28, 28), None
        )
        self.assertFalse((used["ids"] >= 0).any())
        for k in ["means", "scales", "rotations", "harmonics", "opacities"]:
            torch.testing.assert_close(getattr(p.gaussians, k), getattr(o.gaussians, k))
        torch.testing.assert_close(p.thermal_colors, o.thermal_colors)

    def test_dense_source_invalid_mask_is_respected(self):
        o, a = surface_tests.SurfaceTests().setup_scene()
        valid = torch.ones(1, 2, 1, 28, 28)
        valid[..., 10:12, 10:12] = 0
        a = projection_association(o, a, valid)
        self.assertTrue((a["ids"][..., 10:12, 10:12] == -1).all())

    def test_same_label_does_not_override_depth_occlusion(self):
        o = scene()
        a = {"ids": torch.zeros(1, 2, 28, 28, dtype=torch.long)}
        rgb = torch.full((1, 2, 3, 28, 28), 0.5)
        thermal = torch.full((1, 2, 1, 28, 28), 0.6)
        clear = visible_appearance(o, rgb, thermal, a, thermal_sigma=1)
        self.assertGreater(
            float((clear.thermal_colors - o.thermal_colors).abs().max()), 0.02
        )
        means = o.gaussians.means.reshape(1, 2, 28, 28, 3).clone()
        means[:, 1, ..., 2] = 1
        occluded = replace(
            o, gaussians=replace(o.gaussians, means=means.reshape(1, -1, 3))
        )
        fused = visible_appearance(occluded, rgb, thermal, a, thermal_sigma=1)
        torch.testing.assert_close(fused.thermal_colors, o.thermal_colors)

    def test_thermal_disagreement_is_independent_of_rgb(self):
        o = scene()
        a = {"ids": torch.zeros(1, 2, 28, 28, dtype=torch.long)}
        fused = visible_appearance(
            o,
            torch.full((1, 2, 3, 28, 28), 0.5),
            torch.ones(1, 2, 1, 28, 28),
            a,
            thermal_sigma=0.05,
        )
        torch.testing.assert_close(fused.thermal_colors, o.thermal_colors)

    def test_failed_geometry_guard_restores_original_centers(self):
        o, a = surface_tests.SurfaceTests().setup_scene()
        p = replace(o, gaussians=replace(o.gaussians, means=o.gaussians.means * 1.1))
        rgb = torch.zeros(1, 2, 3, 28, 28)
        thermal = torch.zeros(1, 2, 1, 28, 28)

        def fake_render(out, *args):
            return {
                "rgb": rgb + (0 if out is o else 0.5),
                "thermal": thermal,
                "alpha": torch.ones_like(thermal),
            }

        with patch(
            "depth_anything_3.rgbt.surface_reliability.render_rgbt",
            side_effect=fake_render,
        ):
            filtered, audit = guard_surface(o, p, a, rgb, thermal)
        self.assertFalse(audit[0]["accepted"])
        restored = select_geometry(o, p, filtered["ids"])
        torch.testing.assert_close(restored.gaussians.means, o.gaussians.means)

    def test_nonplanar_depth_receives_gradient_and_obeys_bound(self):
        o = scene()
        a = {
            "ids": torch.full((1, 2, 28, 28), -1, dtype=torch.long),
            "normals": torch.empty(0, 3),
            "offsets": torch.empty(0),
            "groups": [],
        }
        model = self.model(
            o, geometry_guard=False, learn_planes=False, dense_depth_limit=0.02
        )
        rgb = torch.rand(1, 2, 3, 28, 28)
        thermal = torch.rand(1, 2, 1, 28, 28)
        result = model(rgb, thermal, initial=o, association=a)
        gaussian_depth(result).mean().backward()
        self.assertGreater(float(model.depth_head[-1].bias.grad.abs().sum()), 1e-5)
        with torch.no_grad():
            model.depth_head[-1].bias.fill_(20)
            result = model(rgb, thermal, initial=o, association=a)
        ratio = gaussian_depth(result) / gaussian_depth(o)
        self.assertLessEqual(float(ratio.max()), 1.020202)
        self.assertGreater(float(ratio.max()), 1.01)

    def test_source_residual_rejects_occlusion_and_invalid_observations(self):
        o = scene()
        rgb = torch.full((1, 2, 3, 28, 28), 0.5)
        thermal = torch.full((1, 2, 1, 28, 28), 0.5)
        rendered = {
            "rgb": rgb - 0.1,
            "thermal": thermal - 0.1,
            "alpha": torch.ones_like(thermal),
            "depth": o.depth.clone(),
        }
        with patch(
            "depth_anything_3.rgbt.surface_reliability.render_rgbt",
            return_value=rendered,
        ):
            visible = source_color_residual(o, rgb, thermal)
            self.assertGreater(float(visible.max()), 0.09)
            self.assertLessEqual(float(visible.max()), 0.100001)
            masked = source_color_residual(o, rgb, thermal, torch.zeros_like(thermal))
            self.assertEqual(float(masked.abs().max()), 0)
            rendered["depth"] = o.depth * 0.5
            occluded = source_color_residual(o, rgb, thermal)
            self.assertEqual(float(occluded.abs().max()), 0)

    def test_feedback_derivative_matches_finite_difference(self):
        o = scene()
        rgb = torch.full((1, 2, 3, 28, 28), 0.5)
        thermal = torch.full((1, 2, 1, 28, 28), 0.5)

        def value(x):
            output = replace(o, thermal_colors=x.expand_as(o.thermal_colors))
            predicted = {
                "rgb": rgb,
                "thermal": x.expand_as(thermal),
                "alpha": torch.ones_like(thermal),
                "depth": o.depth,
            }
            with patch(
                "depth_anything_3.rgbt.surface_reliability.render_rgbt",
                return_value=predicted,
            ):
                return source_color_residual(output, rgb, thermal)[..., 3].mean()

        x = torch.tensor(0.4, requires_grad=True)
        derivative = torch.autograd.grad(value(x), x)[0]
        finite = (value(torch.tensor(0.4001)) - value(torch.tensor(0.3999))) / 0.0002
        torch.testing.assert_close(derivative, finite, atol=0.002, rtol=0.002)
        self.assertGreater(float(derivative.abs()), 0.1)

    def test_thermal_consensus_requires_independent_visible_support(self):
        o = scene()
        rgb = torch.full((1, 2, 3, 28, 28), 0.5)
        thermal = torch.full((1, 2, 1, 28, 28), 0.5)
        depth = o.depth.clone()
        depth[:, 1] = 1
        predicted = {
            "rgb": rgb - 0.1,
            "thermal": thermal - 0.1,
            "alpha": torch.ones_like(thermal),
            "depth": depth,
        }
        with patch(
            "depth_anything_3.rgbt.surface_reliability.render_rgbt",
            return_value=predicted,
        ):
            single = source_color_residual(o, rgb, thermal)
            robust = source_color_residual(o, rgb, thermal, thermal_consensus=True)
        self.assertGreater(float(single[..., 3].max()), 0.01)
        self.assertEqual(float(robust[..., 3].abs().max()), 0)
        torch.testing.assert_close(single[..., :3], robust[..., :3])

    def test_thermal_residual_filter_removes_high_frequency_without_rgb_change(self):
        o = scene()
        rgb = torch.full((1, 2, 3, 28, 28), 0.5)
        thermal = torch.full((1, 2, 1, 28, 28), 0.5)
        y, x = torch.meshgrid(torch.arange(28), torch.arange(28), indexing="ij")
        noise = ((x + y) % 2).float() * 0.2 - 0.1
        predicted = {
            "rgb": rgb - 0.05,
            "thermal": thermal + noise,
            "alpha": torch.ones_like(thermal),
            "depth": o.depth,
        }
        with patch(
            "depth_anything_3.rgbt.surface_reliability.render_rgbt",
            return_value=predicted,
        ):
            raw = source_color_residual(o, rgb, thermal)
            filtered = source_color_residual(o, rgb, thermal, thermal_blur=7)
        self.assertLess(
            float(filtered[..., 3].square().mean()),
            float(raw[..., 3].square().mean()) * 0.1,
        )
        torch.testing.assert_close(raw[..., :3], filtered[..., :3])


if __name__ == "__main__":
    unittest.main()
