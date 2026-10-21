import json
import tempfile
import unittest
from contextlib import nullcontext
from unittest.mock import patch
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from depth_anything_3.thermal.detail import (
    rgb_boundary_preservation, rgb_teacher_inputs, scatter_rgb_teacher,
    relax_pseudo_edges, local_rgb_depth_preservation,
)
from depth_anything_3.thermal.detail_data import DetailSEARDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from train_thermal_detail import teacher_prediction


class DetailLossTests(unittest.TestCase):
    def batch(self, target, hard=False):
        return {"thermal_mask": torch.tensor([[False, True]]),
                "depth": target.clone(), "valid": torch.ones_like(target, dtype=torch.bool),
                "colmap_mask": torch.full((1, 2), hard, dtype=torch.bool)}

    def target(self):
        target = torch.ones(1, 2, 20, 24)
        target[..., 12:] = 4
        return target

    def test_positive_scene_scale_does_not_change_boundary_loss(self):
        teacher = self.target()
        batch = self.batch(teacher)
        pred = (teacher * 7).requires_grad_()
        loss, metrics = rgb_boundary_preservation({"depth": pred}, teacher, torch.ones_like(teacher), batch)
        self.assertLess(float(loss), 1e-7)
        self.assertGreater(float(metrics["teacher_boundary_fraction"]), 0)

    def test_soft_boundary_penalized_only_on_rgb_with_finite_gradients(self):
        teacher = self.target()
        pred = torch.linspace(1, 4, 24).reshape(1, 1, 1, 24).expand_as(teacher).clone().requires_grad_()
        loss, _ = rgb_boundary_preservation({"depth": pred}, teacher, torch.ones_like(teacher), self.batch(teacher))
        self.assertGreater(float(loss), .01)
        loss.backward()
        self.assertTrue(torch.isfinite(pred.grad).all())
        self.assertGreater(float(pred.grad[:, 0].abs().sum()), 0)
        self.assertEqual(float(pred.grad[:, 1].abs().sum()), 0)

    def test_teacher_boundary_conflicting_with_colmap_is_rejected(self):
        teacher = self.target()
        pred = torch.ones_like(teacher).requires_grad_()
        loss, metrics = rgb_boundary_preservation({"depth": pred}, teacher, torch.ones_like(teacher),
                                                  self.batch(torch.ones_like(teacher), hard=True))
        self.assertEqual(float(loss), 0)
        self.assertEqual(float(metrics["teacher_boundary_fraction"]), 0)

    def test_invalid_teacher_pixels_do_not_create_nan_or_false_edges(self):
        teacher = torch.full_like(self.target(), float("nan"))
        pred = torch.ones_like(teacher).requires_grad_()
        loss, _ = rgb_boundary_preservation({"depth": pred}, teacher, torch.ones_like(teacher), self.batch(pred))
        loss.backward()
        self.assertEqual(float(loss), 0)
        self.assertTrue(torch.isfinite(pred.grad).all())

    def test_gather_and_scatter_preserve_rgb_indices(self):
        batch = {"images": torch.arange(4).reshape(1, 4, 1, 1, 1).expand(1, 4, 3, 2, 2),
                 "thermal_mask": torch.tensor([[False, True, False, True]]),
                 "depth": torch.ones(1, 4, 2, 2)}
        indices, images = rgb_teacher_inputs(batch)
        self.assertEqual(indices.tolist(), [0, 2])
        self.assertEqual(images[0, :, 0, 0, 0].tolist(), [0, 2])
        out = {"depth": torch.tensor([2., 3.]).reshape(1, 2, 1, 1).expand(1, 2, 2, 2),
               "depth_conf": torch.full((1, 2, 2, 2), 7.)}
        depth, conf = scatter_rgb_teacher(batch, indices, out)
        self.assertEqual(depth[0, :, 0, 0].tolist(), [2, 1, 3, 1])
        self.assertEqual(conf[0, :, 0, 0].tolist(), [7, 0, 7, 0])

    def test_teacher_context_preserves_same_rgb_output_mapping(self):
        class Teacher(torch.nn.Module):
            def forward(self, images, **kwargs):
                self.seen_views = images.shape[1]
                d = images[:, :, 0].float() + 1
                return {"depth": d, "depth_conf": torch.ones_like(d)}
        model = Teacher().train()
        batch = {"images": torch.arange(4).reshape(1, 4, 1, 1, 1).expand(1, 4, 3, 2, 2),
                 "thermal_mask": torch.tensor([[False, True, False, True]]),
                 "depth": torch.ones(1, 4, 2, 2)}
        with patch("torch.autocast", return_value=nullcontext()):
            rgb_depth, _ = teacher_prediction(model, batch, "rgb")
            self.assertEqual(model.seen_views, 2)
            mixed_depth, mixed_conf = teacher_prediction(model, batch, "mixed")
            self.assertEqual(model.seen_views, 4)
        self.assertTrue(torch.equal(rgb_depth, mixed_depth))
        self.assertEqual(mixed_conf[0, :, 0, 0].tolist(), [1, 0, 1, 0])
        self.assertTrue(model.training)

    def test_relaxation_keeps_colmap_and_thermal_geometry(self):
        target = torch.ones(1, 3, 20, 24)
        batch = {"depth": target, "valid": torch.ones_like(target, dtype=torch.bool),
                 "thermal_mask": torch.tensor([[False, False, True]]),
                 "colmap_mask": torch.tensor([[False, True, False]])}
        modified, relaxed = relax_pseudo_edges(batch, torch.ones_like(target, dtype=torch.bool))
        self.assertFalse(modified["valid"][:, 0].any())
        self.assertTrue(modified["valid"][:, 1:].all())
        self.assertTrue(batch["valid"].all())
        self.assertTrue(relaxed[:, 0].all())
        self.assertFalse(relaxed[:, 1:].any())

    def test_relaxation_retains_scale_support_and_local_loss_scale_invariant(self):
        teacher = self.target()
        batch = self.batch(teacher)
        batch["thermal_mask"][:] = False
        focus = torch.ones_like(teacher, dtype=torch.bool)
        modified, relaxed = relax_pseudo_edges(batch, focus)
        self.assertTrue(modified["valid"].all())
        self.assertFalse(relaxed.any())
        pred = (teacher * 7).requires_grad_()
        loss = local_rgb_depth_preservation({"depth": pred}, teacher, focus, batch)
        self.assertLess(float(loss.detach()), 1e-7)


class DetailSamplingTests(unittest.TestCase):
    def test_deterministic_unique_views_reference_and_eval_unweighted(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            Image.fromarray(np.full((28, 42, 3), 128, dtype=np.uint8)).save(folder / "image.png")
            np.save(folder / "depth.npy", np.ones((28, 42), dtype=np.float32))
            frame = dict(file_path="image.png", depth_file_path="depth.npy", w=42, h=28,
                         fl_x=35., fl_y=35., cx=20., cy=13., transform_matrix=np.eye(4).tolist(),
                         depth_source="COLMAP geometric MVS")
            frames = [{"rgb": frame, "thermal": frame} for _ in range(24)]
            (folder / "transforms.json").write_text(json.dumps({"frames": frames}))
            scenes = [{"scene": "TrainScene/VGGT", "transforms": str(folder / "transforms.json")}]
            scores = {"TrainScene/VGGT": [0]*12 + [1]*12}
            train = DetailSEARDataset(scenes, 8, 28, 42, True, 0, scores=scores)
            a, b = train[0], train[0]
            self.assertTrue(torch.equal(a["frame_indices"], b["frame_indices"]))
            self.assertEqual(len(set(a["frame_indices"].tolist())), 8)
            if (~a["thermal_mask"]).any():
                self.assertFalse(bool(a["thermal_mask"][0]))
            self.assertTrue(torch.allclose(a["extrinsics"][0], torch.eye(4)))
            self.assertTrue(a["colmap_mask"].all())
            val = DetailSEARDataset(scenes, 8, 28, 42, False, 0, scores=scores)
            mixed = val[1]
            self.assertEqual(mixed["thermal_mask"].tolist(), [False, True]*4)
            self.assertEqual(mixed["frame_indices"].tolist(), np.linspace(0, 23, 8, dtype=int).tolist())


if __name__ == "__main__":
    unittest.main()
