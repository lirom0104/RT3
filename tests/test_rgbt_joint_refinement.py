"""Shared geometry must preserve results, gradients and cache invalidation."""

import sys
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import torch
import torch.nn.functional as F
from test_rgbt_refinement import scene

from depth_anything_3.rgbt.attention import WindowNeighborhood
from depth_anything_3.rgbt.source_geometry import SourceGeometry
from depth_anything_3.rgbt.legacy.refinement import NeighborhoodRefiner
from depth_anything_3.rgbt.attention import ProjectedNeighborhood


class JointRefinementTests(unittest.TestCase):
    def test_source_observations_reused_only_while_inputs_are_unchanged(self):
        output = scene()
        rgb = torch.rand(1, 2, 3, 28, 28)
        context = SourceGeometry(rgb)
        with patch(
            "depth_anything_3.rgbt.source_geometry.triangulate_sources",
            side_effect=lambda *args: object(),
        ) as triangulate:
            first = context.observations(output)
            # Gaussian/depth updates do not change source triangulation.
            self.assertIs(
                first, context.observations(replace(output, depth=output.depth * 2))
            )
            self.assertEqual(triangulate.call_count, 1)
            for tensor in (output.extrinsics, output.intrinsics, rgb):
                tensor.add_(0.01)
                self.assertIsNot(first, context.observations(output))
                first = context.observations(output)
            self.assertEqual(triangulate.call_count, 4)
            replaced = replace(output, extrinsics=output.extrinsics.clone())
            self.assertIsNot(first, context.observations(replaced))
            self.assertEqual(triangulate.call_count, 5)
            SourceGeometry(rgb).observations(output)
            self.assertEqual(triangulate.call_count, 6)

    def test_inference_tensors_and_inplace_cameras_are_supported(self):
        with torch.inference_mode():
            output = scene()
            context = SourceGeometry(torch.rand(1, 2, 3, 28, 28))
            with patch(
                "depth_anything_3.rgbt.source_geometry.triangulate_sources",
                side_effect=lambda *args: object(),
            ) as triangulate:
                first = context.observations(output)
                self.assertIs(first, context.observations(output))
                output.extrinsics[:, 1, 0, 3].add_(0.1)
                self.assertIsNot(first, context.observations(output))
                self.assertEqual(triangulate.call_count, 2)

    def test_shared_attention_preserves_outputs_and_feature_gradients(self):
        torch.manual_seed(38)
        output = scene()
        points = output.gaussians.means.reshape(1, 2, 28, 28, 3)
        points = (
            F.interpolate(
                points.permute(0, 1, 4, 2, 3).flatten(0, 1), (7, 7), mode="area"
            )
            .reshape(1, 2, 3, 7, 7)
            .permute(0, 1, 3, 4, 2)
        )
        for cls, method in (
            (ProjectedNeighborhood, "project_pair"),
            (WindowNeighborhood, "window_pair"),
        ):
            with self.subTest(attention=cls.__name__):
                blocks = torch.nn.ModuleList([cls(16), cls(16)])
                feature = torch.randn(1, 2, 16, 7, 7, requires_grad=True)
                results = []
                for shared in (False, True):
                    blocks.zero_grad(set_to_none=True)
                    feature.grad = None
                    correspondence = {} if shared else None
                    with patch.object(
                        cls, method, wraps=getattr(cls, method)
                    ) as compute:
                        value = feature
                        for block in blocks:
                            value = block(
                                value,
                                points,
                                output.extrinsics,
                                output.intrinsics,
                                (28, 28),
                                correspondence=correspondence,
                            )
                        value.square().mean().backward()
                        self.assertEqual(compute.call_count, 2 if shared else 4)
                    results.append(
                        (
                            value.detach().clone(),
                            feature.grad.clone(),
                            [p.grad.clone() for p in blocks.parameters()],
                        )
                    )
                for left, right in zip(results[0][:2], results[1][:2]):
                    torch.testing.assert_close(left, right, rtol=0, atol=0)
                for left, right in zip(results[0][2], results[1][2]):
                    torch.testing.assert_close(left, right, rtol=0, atol=0)

    def test_correspondence_is_rebuilt_after_each_gaussian_update(self):
        torch.manual_seed(39)
        model = NeighborhoodRefiner(channels=16, blocks=2, iterations=2)
        with torch.no_grad():
            model.head[-1].weight.normal_(0, 0.01)
            model.head[-1].bias.normal_(0, 0.01)
        output = scene()
        rgb, thermal = torch.rand(1, 2, 3, 28, 28), torch.rand(1, 2, 1, 28, 28)
        with torch.no_grad():
            expected = model(rgb, thermal, initial=output)
            with patch.object(
                ProjectedNeighborhood,
                "project_pair",
                wraps=ProjectedNeighborhood.project_pair,
            ) as compute:
                actual = model(rgb, thermal, initial=output, share_projection=True)
                self.assertEqual(compute.call_count, 4)
        self.assertGreater(
            float((actual.gaussians.means - output.gaussians.means).abs().max()), 1e-5
        )
        for name in ("means", "scales", "rotations", "harmonics", "opacities"):
            torch.testing.assert_close(
                getattr(expected.gaussians, name),
                getattr(actual.gaussians, name),
                rtol=0,
                atol=0,
            )
        torch.testing.assert_close(
            expected.thermal_colors, actual.thermal_colors, rtol=0, atol=0
        )


if __name__ == "__main__":
    unittest.main()
