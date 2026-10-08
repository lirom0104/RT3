"""The shared student must use cached source data equivalently and actually learn."""

import sys
import unittest
from dataclasses import fields, is_dataclass, replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import torch
from test_rgbt_refinement import scene

from depth_anything_3.rgbt.learned_joint import LearnedJointRefiner
from depth_anything_3.rgbt.renderer import render_rgbt


def cuda(value):
    if isinstance(value, torch.Tensor):
        return value.cuda()
    if is_dataclass(value):
        return replace(
            value, **{f.name: cuda(getattr(value, f.name)) for f in fields(value)}
        )
    return value


@unittest.skipUnless(torch.cuda.is_available(), "CUDA Gaussian rendering")
class LearnedJointTests(unittest.TestCase):
    def test_ordered_updates_reuse_weights_and_preserve_depth_before_appearance_trust(
        self,
    ):
        from unittest.mock import patch

        from depth_anything_3.rgbt.local_trust import spatial_trust

        model = (
            LearnedJointRefiner(
                channels=16,
                ordered_updates=True,
                phase_modulation=True,
                shared_residual=True,
            )
            .cuda()
            .eval()
        )
        initial = cuda(scene())
        rgb = torch.rand(1, 2, 3, 28, 28, device="cuda")
        thermal = torch.rand(1, 2, 1, 28, 28, device="cuda")
        before = {name: id(parameter) for name, parameter in model.named_parameters()}
        with torch.no_grad():
            model.core.head[-1].bias[2] = 0.5
            p = model.prepare(rgb, thermal, initial)
            with (
                patch.object(model.core, "forward", wraps=model.core.forward) as core,
                patch(
                    "depth_anything_3.rgbt.ordered_joint.spatial_trust",
                    wraps=spatial_trust,
                ) as trust,
            ):
                direct = model(rgb, thermal, initial=initial)
                self.assertEqual(core.call_count, 3)
                self.assertEqual(trust.call_count, 2)
                reference, proposal = trust.call_args.args[:2]
                torch.testing.assert_close(
                    reference.gaussians.means,
                    proposal.gaussians.means,
                    rtol=0,
                    atol=0,
                )
            cached = model(rgb, thermal, initial=initial, prepared=p)
        torch.testing.assert_close(
            direct.gaussians.means, cached.gaussians.means, rtol=0, atol=0
        )
        self.assertEqual(
            before,
            {name: id(parameter) for name, parameter in model.named_parameters()},
        )
        self.assertEqual(list(dict(model.named_children())), ["core"])
        self.assertIsNone(model.core._phase_index)
        restored = LearnedJointRefiner(**model.config).cuda()
        restored.load_state_dict(model.state_dict(), strict=True)

    def test_fused_update_preserves_thin_surface_at_zero_initialization(self):
        model = (
            LearnedJointRefiner(
                channels=16, trust=False, color_feedback=False, flexible_geometry=True
            )
            .cuda()
            .eval()
        )
        initial = cuda(scene())
        scales = initial.gaussians.scales.clone()
        scales[..., 0] = 0.0001
        initial = replace(initial, gaussians=replace(initial.gaussians, scales=scales))
        rgb = torch.rand(1, 2, 3, 28, 28, device="cuda")
        thermal = torch.rand(1, 2, 1, 28, 28, device="cuda")
        feedback = {"rgb": rgb, "thermal": thermal, "alpha": torch.ones_like(thermal)}
        prepared = {"base": initial, "association": {}, "feedback": feedback}
        with torch.no_grad():
            result = model(rgb, thermal, initial=initial, prepared=prepared)
        torch.testing.assert_close(result.gaussians.scales, scales, rtol=0, atol=0)

    def test_repeated_shared_update_refreshes_feedback_after_geometry_changes(self):
        from unittest.mock import patch

        model = (
            LearnedJointRefiner(
                channels=16,
                iterations=2,
                trust=False,
                color_feedback=False,
                global_context=True,
            )
            .cuda()
            .eval()
        )
        initial = cuda(scene())
        rgb = torch.rand(1, 2, 3, 28, 28, device="cuda")
        thermal = torch.rand(1, 2, 1, 28, 28, device="cuda")
        feedback = {"rgb": rgb, "thermal": thermal, "alpha": torch.ones_like(thermal)}
        prepared = {"base": initial, "association": {}, "feedback": feedback}
        with (
            torch.no_grad(),
            patch(
                "depth_anything_3.rgbt.learned_joint.render_rgbt", wraps=render_rgbt
            ) as render,
        ):
            model.core.head[-1].bias.fill_(0.02)
            result = model(rgb, thermal, initial=initial, prepared=prepared)
            self.assertEqual(render.call_count, 1)
        self.assertTrue(torch.isfinite(result.gaussians.means).all())
        self.assertIsNone(model.core._prepared_feedback)
        self.assertIsNone(model.core._feedback_gaussians)

    def test_source_preparation_and_cached_forward_are_equivalent(self):
        torch.manual_seed(99)
        model = LearnedJointRefiner(channels=16).cuda().eval()
        output = cuda(scene())
        rgb, thermal = (
            torch.rand(1, 2, 3, 28, 28, device="cuda"),
            torch.rand(1, 2, 1, 28, 28, device="cuda"),
        )
        valid = torch.ones_like(thermal)
        with torch.no_grad():
            model.core.head[-1].weight.normal_(0, 0.002)
            p = model.prepare(rgb, thermal, output, valid, True)
            cached = model(
                rgb,
                thermal,
                output.extrinsics,
                output.intrinsics,
                initial=output,
                source_valid=valid,
                prepared=p,
            )
            direct = model(
                rgb,
                thermal,
                output.extrinsics,
                output.intrinsics,
                initial=output,
                source_valid=valid,
            )
        for name in ["means", "scales", "rotations", "harmonics", "opacities"]:
            torch.testing.assert_close(
                getattr(cached.gaussians, name),
                getattr(direct.gaussians, name),
                rtol=0,
                atol=0,
            )
        torch.testing.assert_close(
            cached.thermal_colors, direct.thermal_colors, rtol=0, atol=0
        )
        self.assertIsNone(model.core._prepared_feedback)
        self.assertEqual(list(dict(model.named_children())), ["core"])
        self.assertFalse(any("teacher" in name for name in model.state_dict()))

    def test_target_render_trains_shared_encoder_and_both_attention_types(self):
        torch.manual_seed(100)
        model = LearnedJointRefiner(channels=16, trust=False).cuda().train()
        initial = cuda(scene())
        rgb, thermal = (
            torch.rand(1, 2, 3, 28, 28, device="cuda"),
            torch.rand(1, 2, 1, 28, 28, device="cuda"),
        )
        with torch.no_grad():
            model.core.head[-1].weight.normal_(0, 0.002)
        output = model(
            rgb, thermal, initial.extrinsics, initial.intrinsics, initial=initial
        )
        rendered = render_rgbt(
            output, initial.extrinsics[:, :1], initial.intrinsics[:, :1], 28, 28
        )
        loss = rendered["rgb"].square().mean() + rendered["thermal"].square().mean()
        loss.backward()
        for module in [model.core.neck, *model.core.blocks, model.core.head]:
            gradients = [p.grad for p in module.parameters() if p.grad is not None]
            self.assertTrue(gradients)
            self.assertTrue(all(torch.isfinite(g).all() for g in gradients))
            self.assertGreater(sum(float(g.abs().sum()) for g in gradients), 0)
        restored = LearnedJointRefiner(**model.config).cuda()
        restored.load_state_dict(model.state_dict(), strict=True)


if __name__ == "__main__":
    unittest.main()
