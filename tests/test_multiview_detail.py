import numpy as np
from pathlib import Path
import sys
import unittest
from depth_anything_3.thermal.multiview_detail import transfer_boundaries, reconcile_depths

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from refine_multiview_detail import export_masks


class MultiviewDetailTests(unittest.TestCase):
    def test_sharp_boundary_uses_base_scale_and_preserves_thermal(self):
        teacher = np.ones((2, 32, 48), dtype=np.float32)
        teacher[:, :, 24:] = 2
        base = teacher * 3
        base[0, :, 20:28] = np.linspace(3, 6, 8)
        result = transfer_boundaries(base, teacher, [False, True], radius=4, iterations=1000)
        assert np.array_equal(result[1], base[1])
        assert np.allclose(result[0, :, :12], 3, atol=.01)
        assert np.allclose(result[0, :, 36:], 6, atol=.01)
        assert np.median(result[0, :, 24] / result[0, :, 23]) > 1.9


    def test_translated_plane_and_out_of_view_support(self):
        depth = np.full((3, 20, 30), 2., dtype=np.float32)
        e = np.tile(np.eye(4, dtype=np.float32)[:3], (3, 1, 1))
        e[1, 0, 3] = -.2
        e[2, 0, 3] = .2
        k = np.tile(np.array([[20., 0, 15], [0, 20., 10], [0, 0, 1]], dtype=np.float32), (3, 1, 1))
        result, support = reconcile_depths(depth, e, k, [False, False, True], [2, 2, 0])
        assert np.allclose(result, depth, atol=1e-6)
        assert np.all(support[0, :, 4:-4] == 1)
        assert np.all(support[0, :, :2] == 0)
        assert np.all(support[2, :, 4:-4] == 2)


    def test_whole_thin_rod_keeps_base_geometry_and_multiview_support(self):
        # The rod fits wholly inside the weak boundary band. Different teacher
        # biases used to move it to two depths, then the exporter removed it.
        base = np.full((3, 48, 64), 6., dtype=np.float32)
        base[:, 12:18, 12:52] = 3.
        teacher = np.full_like(base, 2.)
        teacher[0, 12:18, 12:52] = .7
        teacher[1, 12:18, 12:52] = .9
        teacher[2, 12:18, 12:52] = 1.
        edge = transfer_boundaries(base, teacher, [False, False, True])
        assert np.array_equal(edge[2], base[2])
        assert np.allclose(edge[:2, 14:16, 15:49], 3., rtol=.015)
        e = np.tile(np.eye(4, dtype=np.float32)[:3], (3, 1, 1))
        k = np.tile(np.eye(3, dtype=np.float32), (3, 1, 1))
        result, support = reconcile_depths(edge, e, k, [False, False, True], [2, 2, 0])
        assert np.allclose(result[:2, 14:16, 15:49], 3., rtol=.015)
        assert np.all(support[:2, 14:16, 15:49] == 1)


    def test_narrow_background_hole_is_not_anchored_to_filled_base(self):
        teacher = np.full((2, 64, 64), 2., dtype=np.float32)
        teacher[:, 8:56, 8:56] = 1.
        teacher[:, 28:36, 28:36] = 2.
        base = teacher * 3
        base[:, 28:36, 28:36] = 3.  # The base missed this opening.
        result = transfer_boundaries(base, teacher, [False, True])
        assert np.array_equal(result[1], base[1])
        assert np.median(result[0, 29:35, 29:35]) / np.median(result[0, 20:24, 28:36]) > 1.7


    def test_low_confidence_thin_export_requires_two_rgb_views_and_thermal_match(self):
        depth = np.full((4, 48, 64), 6., dtype=np.float32)
        depth[:, 12:18, 12:52] = 3.
        teacher = depth / 3
        confidence = np.full_like(depth, 10.)
        confidence[:, 12:18, 12:52] = 7.  # Below the scene median, still useful.
        e = np.tile(np.eye(4, dtype=np.float32)[:3], (4, 1, 1))
        k = np.tile(np.eye(3, dtype=np.float32), (4, 1, 1))
        base = dict(depth=depth, depth_conf=confidence, extrinsics=e, intrinsics=k)
        support = np.full(depth.shape, 2, dtype=np.uint8)
        support[0, 14, 24] = 1
        confidence[0, 14, 25] = 4.  # Also reject extremely weak confidence.
        changed = depth.copy()
        changed[3, 14, 26] = 5.  # Thermal disagreement must not be filled.
        valid, _, recovered = export_masks(base, dict(depth=teacher), changed, support,
                                           np.array([False, False, False, True]), [3, 3, 3, 0])
        assert not valid[0, 14, 24]
        assert not valid[0, 14, 25]
        assert valid[0, 14, 26] and recovered[0, 14, 26]
        assert valid[3, 14, 27] and recovered[3, 14, 27]
        assert not valid[3, 14, 26]


    def test_inconsistent_view_does_not_average_into_plane(self):
        depth = np.full((3, 16, 24), 2., dtype=np.float32)
        depth[1] = 4.
        e = np.tile(np.eye(4, dtype=np.float32)[:3], (3, 1, 1))
        k = np.tile(np.eye(3, dtype=np.float32), (3, 1, 1))
        result, support = reconcile_depths(depth, e, k, [False, False, True], [2, 2, 0])
        assert np.allclose(result, depth, atol=1e-6)
        assert not support[:2].any()


    def test_rejects_invalid_depth_and_camera_shapes(self):
        bad = np.ones((2, 8, 8), dtype=np.float32)
        bad[0, 1, 1] = 0
        with self.assertRaises(ValueError):
            transfer_boundaries(bad, np.ones_like(bad), [False, True])
        good = np.ones_like(bad)
        with self.assertRaises(ValueError):
            reconcile_depths(good, np.eye(4), np.eye(3), [False, True], [1, 0])


if __name__ == "__main__":
    unittest.main()
