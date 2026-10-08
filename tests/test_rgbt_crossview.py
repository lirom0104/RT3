"""V4 source exclusion, geometry learning, calibration and occlusion invariants."""

import sys
import unittest
from dataclasses import fields, replace
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import torch
from test_rgbt_refinement import scene

from depth_anything_3.rgbt.calibration import calibrate_output, triangulate_sources
from depth_anything_3.rgbt.legacy.crossview import CrossViewRefiner
from depth_anything_3.rgbt.attention import WindowNeighborhood
from depth_anything_3.rgbt.renderer import leave_one_out
from depth_anything_3.rgbt.geometry import gaussian_depth, pixel_grid
from depth_anything_3.rgbt.input_calibration import encode_radiometric, register_thermal
from depth_anything_3.rgbt.model import render_rgbt


class CrossViewTests(unittest.TestCase):
    def test_registration_supports_different_sensor_grids_and_validity(self):
        thermal = np.arange(12, dtype=np.float32).reshape(3, 4)
        mx = np.array([[0, 1, 2], [1, 2, 5]], np.float32)
        my = np.array([[0, 0, 0], [2, 2, 2]], np.float32)
        warped, valid = register_thermal(
            thermal, (2, 3), {"map_x": mx[None], "map_y": my[None]}, 0
        )
        np.testing.assert_allclose(warped[:1], [[0, 1, 2]])
        np.testing.assert_allclose(warped[1, :2], [9, 10])
        self.assertEqual(valid[1, 2], 0)

    def test_radiometric_encoding_is_shared_across_views(self):
        specification = {
            "raw_scale": 0.01,
            "raw_offset": -273.15,
            "temperature_min": 20,
            "temperature_max": 40,
            "unit": "C",
        }
        raw = np.array([[30000, 31000]], np.uint16)
        values = encode_radiometric(raw, specification)
        np.testing.assert_allclose(values, [[0.3425, 0.8425]], atol=2e-6)
        np.testing.assert_allclose(
            encode_radiometric(raw[:, :1], specification), values[:, :1]
        )
        with self.assertRaises(ValueError):
            encode_radiometric(np.zeros((2, 2, 3), np.uint8), specification)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA rasterization")
    def test_fused_renderer_matches_values_and_gradients(self):
        o = scene()
        o = replace(
            o,
            **{
                f.name: getattr(o, f.name).cuda()
                for f in fields(o)
                if isinstance(getattr(o, f.name), torch.Tensor)
            },
            gaussians=replace(
                o.gaussians,
                **{
                    f.name: getattr(o.gaussians, f.name).cuda()
                    for f in fields(o.gaussians)
                    if isinstance(getattr(o.gaussians, f.name), torch.Tensor)
                },
            ),
        )
        with torch.no_grad():
            o.gaussians.harmonics.normal_(0, 0.05)
        parameters = []
        for name in ["means", "scales", "rotations", "harmonics", "opacities"]:
            value = getattr(o.gaussians, name).detach().requires_grad_(True)
            o.gaussians = replace(o.gaussians, **{name: value})
            parameters.append(value)
        before = render_rgbt(o, o.extrinsics, o.intrinsics, 28, 28)
        after = render_rgbt(
            replace(o, fused_rendering=True), o.extrinsics, o.intrinsics, 28, 28
        )
        for key in before:
            torch.testing.assert_close(before[key], after[key], atol=2e-5, rtol=2e-5)
        g1 = torch.autograd.grad(
            sum(v.mean() for v in before.values()), parameters, retain_graph=True
        )
        g2 = torch.autograd.grad(sum(v.mean() for v in after.values()), parameters)
        for a, b in zip(g1, g2):
            torch.testing.assert_close(a, b, atol=2e-5, rtol=2e-3)

    def build(self):
        o = scene()
        model = CrossViewRefiner(channels=16, blocks=1, feedback=False)
        return model, o

    def test_geometry_receives_gradient_and_stays_bounded(self):
        model, o = self.build()
        rgb = torch.rand(1, 2, 3, 28, 28)
        thermal = torch.rand(1, 2, 1, 28, 28)
        out = model(rgb, thermal, initial=o)
        torch.testing.assert_close(
            out.gaussians.means, o.gaussians.means, atol=1e-6, rtol=1e-5
        )
        gaussian_depth(out).mean().backward()
        self.assertGreater(float(model.head.proposal[-1].weight.grad[2].abs().sum()), 0)
        with torch.no_grad():
            model.head.proposal[-1].bias[2] = 10
        out = model(rgb, thermal, initial=o)
        ratio = gaussian_depth(out) / gaussian_depth(o)
        self.assertGreater(float(ratio.min()), 1.01)
        self.assertLess(float(ratio.max()), 1.101)
        self.assertFalse(model.source_guard)

    def test_feedback_excludes_own_view(self):
        o = scene()
        h, w = o.depth.shape[-2:]

        def render(output, E, K, h, w):
            value = output.thermal_colors.mean()
            return {"thermal": value.expand(1, 1, 1, h, w)}

        altered = replace(o, thermal_colors=o.thermal_colors.clone())
        altered.thermal_colors[:, : h * w] = 0.9
        with patch("depth_anything_3.rgbt.renderer.render_rgbt", side_effect=render):
            before = leave_one_out(o, o.extrinsics, o.intrinsics, h, w)["thermal"]
            after = leave_one_out(altered, o.extrinsics, o.intrinsics, h, w)["thermal"]
        torch.testing.assert_close(before[:, 0], after[:, 0])
        self.assertGreater(float((before[:, 1] - after[:, 1]).abs().mean()), 0.3)

    def test_unsupported_single_view_has_no_feedback(self):
        o = scene()
        o = replace(o, depth=o.depth[:, :1])
        feedback = leave_one_out(o, o.extrinsics[:, :1], o.intrinsics[:, :1], 28, 28)
        self.assertEqual(float(feedback["alpha"].sum()), 0)

    def test_calibration_recovers_scale_and_rejects_sparse_support(self):
        o = scene()
        xy = pixel_grid(28, 28, "cpu")[3:25:3, 3:25:3].reshape(-1, 2).numpy()
        observations = np.concatenate([xy, np.full((len(xy), 1), 2.4, np.float32)], 1)
        with patch(
            "depth_anything_3.rgbt.calibration.triangulate_sources",
            return_value=[observations, observations],
        ):
            fixed, info = calibrate_output(o, torch.rand(1, 2, 3, 28, 28))
        torch.testing.assert_close(
            gaussian_depth(fixed), torch.full_like(o.depth, 2.4), atol=1e-5, rtol=1e-5
        )
        self.assertAlmostEqual(info["ratios"][0][0], 1.2, places=5)
        with patch(
            "depth_anything_3.rgbt.calibration.triangulate_sources",
            return_value=[observations[:5], observations[:5]],
        ):
            fixed, _ = calibrate_output(o, torch.rand(1, 2, 3, 28, 28))
        torch.testing.assert_close(
            fixed.gaussians.means, o.gaussians.means, atol=1e-6, rtol=1e-5
        )

    def test_triangulation_uses_supplied_cameras(self):
        o = scene()
        xy = pixel_grid(28, 28, "cpu")[4:24:3, 4:24:3].reshape(-1, 2).numpy()
        other = xy.copy()
        other[:, 0] -= 35 * 0.1 / 2.5
        desc = np.eye(len(xy), 128, dtype=np.float32)
        with patch(
            "depth_anything_3.rgbt.calibration.image_features",
            side_effect=[(xy, desc), (other, desc)],
        ):
            obs = triangulate_sources(
                torch.rand(2, 3, 28, 28), o.extrinsics[0], o.intrinsics[0]
            )
        self.assertGreater(len(obs[0]), 20)
        np.testing.assert_allclose(obs[0][:, 2], 2.5, atol=1e-4)

    def test_window_attention_masks_occluded_surfaces(self):
        torch.manual_seed(7)
        module = WindowNeighborhood(16)
        x = torch.randn(1, 2, 16, 4, 4)
        E = torch.eye(4)[None, None].repeat(1, 2, 1, 1)
        K = torch.tensor([[4.0, 0.0, 2.0], [0.0, 4.0, 2.0], [0.0, 0.0, 1.0]])[
            None, None
        ].repeat(1, 2, 1, 1)
        xy = pixel_grid(4, 4, "cpu")[None, None].expand(1, 2, -1, -1, -1)
        rays = torch.einsum(
            "bvij,bvhwj->bvhwi",
            K.inverse(),
            torch.cat([xy, torch.ones_like(xy[..., :1])], -1),
        )
        pts = rays * torch.tensor([2.0, 10.0])[None, :, None, None, None]
        y = x.clone()
        y[:, 1] = torch.randn_like(y[:, 1]) * 4
        a, b = module(x, pts, E, K, (4, 4)), module(y, pts, E, K, (4, 4))
        torch.testing.assert_close(a[:, 0], b[:, 0])
        a, b = module(x, rays * 2, E, K, (4, 4)), module(y, rays * 2, E, K, (4, 4))
        self.assertGreater(float((a[:, 0] - b[:, 0]).abs().max()), 0.001)


if __name__ == "__main__":
    unittest.main()
