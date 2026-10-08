"""Geometry/fusion invariants independent of the 5 GB foundation checkpoint."""

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import torch
from PIL import Image

from depth_anything_3.model.gs_adapter import GaussianAdapter
from depth_anything_3.rgbt.data import SceneDataset, grayscale
from depth_anything_3.rgbt.geometry import (
    depth_to_range,
    gaussian_depth,
    multiview_depth_loss,
    pixel_grid,
)
from depth_anything_3.rgbt.model import (
    RGBTOutput,
    render_rgbt,
)
from depth_anything_3.rgbt.reconstruction import DeformableThermalAdapter
from depth_anything_3.specs import Gaussians


class RGBTTests(unittest.TestCase):
    def test_camera_z_anchor_and_final_gaussian_supervision(self):
        # Off-center intrinsics and a translated source: final centers must land
        # on the z=2 plane and project back to their source pixel coordinates.
        h, w = 12, 16
        E = torch.eye(4)[None, None].repeat(1, 2, 1, 1)
        angle = torch.tensor(0.2)
        E[0, 1, :2, :2] = torch.stack(
            [
                torch.stack([angle.cos(), -angle.sin()]),
                torch.stack([angle.sin(), angle.cos()]),
            ]
        )
        E[0, 1, 0, 3] = -0.2
        K = torch.tensor([[20.0, 0.0, 5.0], [0.0, 21.0, 7.0], [0.0, 0.0, 1.0]])[
            None, None
        ].repeat(1, 2, 1, 1)
        depth = torch.full((1, 2, h, w), 2.0)
        xy = pixel_grid(h, w, "cpu")[None, None].expand(1, 2, -1, -1, -1)
        raw = torch.zeros(1, 2, h, w, 37)
        raw[..., 8] = 1
        adapter = GaussianAdapter(sh_degree=2, pred_offset_depth=True)
        gs = adapter(
            E,
            K,
            depth_to_range(depth, K, xy),
            torch.ones_like(depth) * 0.5,
            raw,
            (h, w),
        )
        gs.means.requires_grad_()
        out = RGBTOutput(gs, torch.ones(1, 2 * h * w, 1), depth, E, K, None, None)
        torch.testing.assert_close(gaussian_depth(out), depth, rtol=1e-5, atol=1e-5)
        points = gs.means.reshape(1, 2, h, w, 3)
        cam = (
            torch.einsum("bvij,bvhwj->bvhwi", E[..., :3, :3], points)
            + E[..., :3, 3][:, :, None, None]
        )
        uvz = torch.einsum("bvij,bvhwj->bvhwi", K, cam)
        torch.testing.assert_close(
            uvz[..., :2] / uvz[..., 2:], xy, rtol=1e-5, atol=1e-5
        )
        valid = torch.ones(1, 2, 1, h, w)
        self.assertLess(
            float(multiview_depth_loss(out, valid, stride=2).detach()), 1e-5
        )
        with torch.no_grad():
            gs.means[:, : h * w, 2] += 0.2
        loss = multiview_depth_loss(out, valid, stride=2)
        self.assertGreater(float(loss.detach()), 0.01)
        loss.backward()
        self.assertGreater(float(gs.means.grad.abs().sum()), 0)

    def test_diverse_views_exclude_target_and_expand_coverage(self):
        root = Path(__file__).resolve().parents[1] / "outputs/rgbt/data_336"
        if not (root / "manifest.json").exists():
            self.skipTest("Run preparation first")
        ds = SceneDataset(root, "val", views=6, sampling="wide")
        a, b = ds[0], ds[0]
        self.assertEqual(a["context_ids"], b["context_ids"])
        self.assertEqual(len(set(a["context_ids"])), 6)
        self.assertNotIn(a["id"], a["context_ids"])
        scene = ds.scenes[0]
        heldout = {scene["frames"][i]["id"] for i in scene["target_pool"]}
        self.assertFalse(heldout & set(a["context_ids"]))
        near = SceneDataset(root, "val", views=3)[0]
        self.assertTrue(set(near["context_ids"]).isdisjoint(a["context_ids"]))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA renderer test")
    def test_shared_geometry_renderer_and_offcenter_intrinsics(self):
        device = "cuda:0"
        means = torch.tensor([[[0.0, 0.0, 2.0]]], device=device, requires_grad=True)
        sh = torch.zeros(1, 1, 3, 9, device=device)
        sh[..., 0] = (0.25 - 0.5) / 0.28209479177387814
        gs = Gaussians(
            means=means,
            scales=torch.full((1, 1, 3), 0.04, device=device),
            rotations=torch.tensor([[[1.0, 0.0, 0.0, 0.0]]], device=device),
            harmonics=sh,
            opacities=torch.full((1, 1), 0.9, device=device),
        )
        E = torch.eye(4, device=device)[None, None]
        K = torch.tensor(
            [[[[40.0, 0.0, 9.5], [0.0, 40.0, 13.5], [0.0, 0.0, 1.0]]]], device=device
        )
        thermal = torch.full((1, 1, 1), 0.25, device=device, requires_grad=True)
        output = RGBTOutput(
            gs, thermal, torch.ones(1, 1, 28, 28, device=device), E, K, None, None
        )
        p = render_rgbt(output, E, K, 28, 28)
        torch.testing.assert_close(
            p["rgb"][:, :, :1], p["thermal"], atol=1e-6, rtol=1e-5
        )
        peak = int(p["thermal"].flatten().argmax())
        self.assertLessEqual(abs(peak % 28 - 9), 1)
        self.assertLessEqual(abs(peak // 28 - 13), 1)
        p["thermal"].sum().backward()
        self.assertTrue(torch.isfinite(means.grad).all())
        self.assertGreater(float(thermal.grad.abs().sum()), 0)

    def test_zero_gate_preserves_rgb_and_gate_can_learn(self):
        torch.manual_seed(4)
        module = DeformableThermalAdapter(
            32, channels=16, hidden=16, heads=4, points=2, levels=2
        )
        tokens = torch.randn(2, 3, 13, 32, requires_grad=True)
        maps = [torch.randn(6, 16, 8, 6), torch.randn(6, 16, 4, 3)]
        out = module(tokens, maps, (4, 3))
        torch.testing.assert_close(out, tokens, rtol=0, atol=0)
        out.square().sum().backward()
        self.assertGreater(abs(float(module.gamma.grad)), 0)
        with torch.no_grad():
            module.gamma.fill_(0.1)
        a = module(tokens, maps, (4, 3))
        b = module(tokens, [m + 2 for m in maps], (4, 3))
        self.assertGreater(float((a[:, :, 1:] - b[:, :, 1:]).abs().max().detach()), 0)
        torch.testing.assert_close(a[:, :, :1], tokens[:, :, :1], rtol=0, atol=0)

    def test_gray_has_fixed_scale(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "test.png"
            a = np.array(
                [[[255, 0, 0], [0, 255, 0], [0, 0, 255], [128, 128, 128]]], np.uint8
            )
            Image.fromarray(a).save(p)
            np.testing.assert_allclose(
                grayscale(p), [[0.299, 0.587, 0.114, 128 / 255]], atol=1e-6
            )
            Image.fromarray(np.array([[32768, 65535]], np.uint16)).save(p)
            np.testing.assert_allclose(grayscale(p), [[32768 / 65535, 1]], atol=1e-6)

    def test_prepared_scene_and_view_separation(self):
        root = Path(__file__).resolve().parents[1] / "outputs/rgbt/data_336"
        if not (root / "manifest.json").exists():
            self.skipTest("Run data preparation first")
        sets = {
            split: SceneDataset(root, split, views=3)
            for split in ("train", "val", "test")
        }
        for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
            self.assertFalse(
                {s["id"] for s in sets[a].scenes} & {s["id"] for s in sets[b].scenes}
            )
            self.assertFalse(
                {s["sequence"] for s in sets[a].mm5}
                & {s["sequence"] for s in sets[b].mm5}
            )
        for ds in sets.values():
            for i in range(min(3, len(ds))):
                x = ds[i]
                self.assertNotIn(x["id"], x["context_ids"])
                torch.testing.assert_close(
                    x["E"][0], torch.eye(4), atol=1e-5, rtol=1e-5
                )
                self.assertTrue(torch.isfinite(x["rgb"]).all())
                self.assertEqual(x["rgb"].shape[0], 4)


if __name__ == "__main__":
    unittest.main()
