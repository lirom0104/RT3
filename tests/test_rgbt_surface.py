"""Shared surfaces must remove source layering without altering source rays."""

import sys
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import torch
from test_rgbt_refinement import scene

from depth_anything_3.rgbt.legacy.surface import SurfaceRefiner
from depth_anything_3.rgbt.surface_consensus import (
    anchor_consensus,
    apply_consensus,
    build_consensus,
    plane_covariance,
)
from depth_anything_3.utils.geometry import affine_inverse, quat_to_mat


class SurfaceTests(unittest.TestCase):
    def setup_scene(self):
        o = scene()
        p = o.gaussians.means.reshape(1, 2, 28, 28, 3).clone()
        c = affine_inverse(o.extrinsics)[..., :3, 3][:, :, None, None]
        p[:, 1] = c[:, 1] + (p[:, 1] - c[:, 1]) * 1.05
        o = replace(o, gaussians=replace(o.gaussians, means=p.reshape(1, -1, 3)))
        ids = torch.zeros(1, 2, 28, 28, dtype=torch.long)
        ids[:, :, :3] = -1
        assoc = {
            "ids": ids,
            "normals": torch.tensor([[0.0, 0.0, -1.0]]),
            "offsets": torch.tensor([2.0]),
            "groups": [{"radius": 2.0, "views": [0, 1]}],
        }
        return o, assoc

    def test_projection_preserves_rays_and_unassigned_geometry(self):
        o, a = self.setup_scene()
        n = apply_consensus(o, a)
        mask = a["ids"].flatten() >= 0
        torch.testing.assert_close(
            n.gaussians.means[0, mask, 2],
            torch.full_like(n.gaussians.means[0, mask, 2], 2.0),
        )
        torch.testing.assert_close(
            n.gaussians.means[0, ~mask], o.gaussians.means[0, ~mask]
        )
        c = affine_inverse(o.extrinsics)[..., :3, 3][:, :, None, None]
        rays = [x.gaussians.means.reshape(1, 2, 28, 28, 3) - c for x in [o, n]]
        torch.testing.assert_close(
            rays[0] / rays[0][..., 2:], rays[1] / rays[1][..., 2:]
        )

    def test_group_parameters_receive_geometry_gradient(self):
        o, a = self.setup_scene()
        a["offsets"] = a["offsets"].requires_grad_(True)
        n = apply_consensus(o, a)
        n.gaussians.means[..., 2].mean().backward()
        self.assertTrue(torch.isfinite(a["offsets"].grad).all())
        self.assertGreater(a["offsets"].grad.abs().sum(), 0.1)

    def test_unsupported_plane_does_not_move_centers(self):
        o, a = self.setup_scene()
        a["offsets"] = torch.tensor([5.0])
        n = apply_consensus(o, a)
        torch.testing.assert_close(n.gaussians.means, o.gaussians.means)
        torch.testing.assert_close(n.gaussians.scales, o.gaussians.scales)

    def test_point_count_and_low_opacity_layers_are_retained_in_metrics(self):
        from depth_anything_3.rgbt.surface_consensus import surface_metrics

        o, a = self.setup_scene()
        o = replace(
            o,
            gaussians=replace(
                o.gaussians, opacities=torch.zeros_like(o.gaussians.opacities)
            ),
        )
        before = surface_metrics(o, a)
        after = apply_consensus(o, a)
        measured = surface_metrics(after, a)
        self.assertEqual(before["assigned_gaussians"], measured["assigned_gaussians"])
        self.assertEqual(before["total_gaussians"], measured["total_gaussians"])
        self.assertGreater(before["thickness"], 0.01)
        self.assertLess(measured["thickness"], 1e-6)

    def test_surface_normal_matches_thin_covariance_axis(self):
        o, a = self.setup_scene()
        n = apply_consensus(o, a, flatten=1.0)
        mask = a["ids"].flatten() >= 0
        rot = quat_to_mat(n.gaussians.rotations[0, mask][:, [1, 2, 3, 0]])
        self.assertLess(float(n.gaussians.scales[0, mask, 2].max()), 0.003)
        torch.testing.assert_close(rot[:, :, 2], a["normals"].expand(len(rot), -1))

    def test_learned_head_keeps_common_plane(self):
        o, a = self.setup_scene()
        model = SurfaceRefiner(channels=16, plane_cov=True, geometry_guard=False)
        rgb = torch.rand(1, 2, 3, 28, 28)
        thermal = torch.rand(1, 2, 1, 28, 28)
        with torch.no_grad():
            model.plane_head[-1].bias.copy_(torch.tensor([0.1, -0.1, 0.0, 0.2]))
        n = model(rgb, thermal, initial=o, association=a)
        plane = model.last_association
        mask = a["ids"].flatten() >= 0
        error = n.gaussians.means[0, mask] @ plane["normals"][0] + plane["offsets"][0]
        self.assertLess(float(error.abs().max()), 1e-5)
        n.gaussians.means[..., 2].mean().backward()
        self.assertTrue(torch.isfinite(model.plane_head[-1].bias.grad).all())
        self.assertGreater(float(model.plane_head[-1].bias.grad.abs().sum()), 1e-5)

    def test_source_triangulation_recovers_plane_position(self):
        o, a = self.setup_scene()
        y, x = np.meshgrid(np.arange(5, 24, 2), np.arange(2, 26, 2), indexing="ij")
        observations = np.stack(
            [x.flatten() + 0.5, y.flatten() + 0.5, np.full(x.size, 1.9)], -1
        ).astype(np.float32)
        with patch(
            "depth_anything_3.rgbt.calibration.triangulate_sources",
            return_value=[observations, observations],
        ):
            anchored = anchor_consensus(o, torch.zeros(1, 2, 3, 28, 28), a)
        self.assertTrue(anchored["anchor_audit"][0]["accepted"])
        self.assertAlmostEqual(float(anchored["offsets"][0]), 1.9, places=5)

    def test_two_distinct_surfaces_are_not_merged(self):
        o = scene()
        h = w = 112
        v = 2
        y, x = torch.meshgrid(
            torch.arange(h) + 0.5, torch.arange(w) + 0.5, indexing="ij"
        )
        K = (
            torch.tensor([[100.0, 0, 56], [0, 100.0, 56], [0, 0, 1]])[None, None]
            .expand(1, v, 3, 3)
            .clone()
        )
        uv = torch.stack([x, y, torch.ones_like(x)], -1)
        rays = torch.einsum("bvij,hwj->bvhwi", K.inverse(), uv)
        depth = torch.where(x < w // 2, 2.0, 2.3)[None, None].expand(1, v, h, w).clone()
        center = affine_inverse(o.extrinsics)[..., :3, 3][:, :, None, None]
        pts = center + rays * depth[..., None]
        o = replace(
            o,
            depth=depth,
            intrinsics=K,
            gaussians=replace(o.gaussians, means=pts.reshape(1, -1, 3)),
        )
        a = build_consensus(o, torch.zeros(1, v, 3, h, w))
        self.assertGreaterEqual(len(a["groups"]), 2)
        for g in range(len(a["groups"])):
            selected = pts.reshape(-1, 3)[a["ids"].flatten() == g, 2]
            self.assertLess(float(selected.max() - selected.min()), 0.01)

    def test_thin_covariance_preserves_source_image_footprint(self):
        o, a = self.setup_scene()
        o = apply_consensus(o, a)
        thin = plane_covariance(o, a, thickness=0.001)
        mask = a["ids"].flatten() >= 0
        c = (
            affine_inverse(o.extrinsics)[..., :3, 3][:, :, None, None]
            .expand(-1, -1, 28, 28, -1)
            .reshape(-1, 3)
        )
        cam = o.gaussians.means[0] - c
        x, y, z = cam.unbind(-1)
        jac = torch.stack(
            [
                1 / z,
                torch.zeros_like(z),
                -x / z.square(),
                torch.zeros_like(z),
                1 / z,
                -y / z.square(),
            ],
            -1,
        ).reshape(-1, 2, 3)
        covs = []
        for out in [o, thin]:
            rot = quat_to_mat(out.gaussians.rotations[0][:, [1, 2, 3, 0]])
            cov = (
                rot
                @ torch.diag_embed(out.gaussians.scales[0].square())
                @ rot.transpose(-1, -2)
            )
            covs.append(jac @ cov @ jac.transpose(-1, -2))
        torch.testing.assert_close(covs[0][mask], covs[1][mask], rtol=1e-4, atol=1e-8)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA renderer")
    def test_scalar_thermal_sh_matches_both_renderers(self):
        from dataclasses import fields

        from depth_anything_3.rgbt.model import render_rgbt

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
                },
            ),
        )
        sh = torch.zeros(1, o.gaussians.means.shape[1], 9, device="cuda")
        sh[..., 0] = (0.4 - 0.5) / 0.28209479177387814
        sh[..., 2] = 0.03
        o = replace(o, thermal_harmonics=sh)
        a = render_rgbt(o, o.extrinsics, o.intrinsics, 28, 28)
        b = render_rgbt(
            replace(o, fused_rendering=True), o.extrinsics, o.intrinsics, 28, 28
        )
        torch.testing.assert_close(a["thermal"], b["thermal"], rtol=2e-5, atol=2e-5)


if __name__ == "__main__":
    unittest.main()
