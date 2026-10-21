import unittest
import numpy as np

from depth_anything_3.thermal.detail_refinement import refine_rgb_depth


class DetailRefinementTests(unittest.TestCase):
    def test_preserves_thermal_and_does_not_mutate_inputs(self):
        teacher = np.ones((4, 20, 24), dtype=np.float32)
        teacher[..., 12:] = 3
        student = np.full_like(teacher, 2.)
        original = student.copy()
        mask = np.array([False, True, False, True])
        out, info = refine_rgb_depth(teacher, student, mask, 3.)
        self.assertTrue(np.array_equal(out[mask], student[mask]))
        self.assertTrue(np.array_equal(student, original))
        self.assertTrue(info["thermal_depths_exactly_unchanged"])
        self.assertTrue(np.isfinite(out).all())
        self.assertTrue((out > 0).all())

    def test_teacher_units_cancel_and_student_units_are_preserved(self):
        teacher = np.linspace(1, 4, 480).reshape(1, 20, 24).astype(np.float32)
        student = np.flip(teacher, -1).copy() * 2
        mask = np.array([False])
        expected, _ = refine_rgb_depth(teacher, student, mask, 3.)
        actual, _ = refine_rgb_depth(teacher * 7, student * 5, mask, 3.)
        np.testing.assert_allclose(actual, expected * 5, rtol=2e-6)

    def test_recovers_fine_depth_structure_lost_in_flat_student(self):
        teacher = np.ones((2, 32, 64), dtype=np.float32)
        teacher[..., 1::2] = 2
        student = np.full_like(teacher, 3.)
        out, _ = refine_rgb_depth(teacher, student, np.array([False, True]), 3.)
        ratio = out[0, :, 25::2] / out[0, :, 24:-1:2]
        self.assertGreater(float(ratio.min()), 1.9)
        self.assertTrue(np.array_equal(out[1], student[1]))

    def test_no_rgb_passthrough_and_invalid_input_rejected(self):
        a = np.ones((2, 20, 24), dtype=np.float32)
        out, info = refine_rgb_depth(a, a * 3, np.array([True, True]), 3.)
        self.assertTrue(np.array_equal(out, a * 3))
        self.assertEqual(info["rgb_views"], 0)
        with self.assertRaises(ValueError):
            refine_rgb_depth(a * 0, a, np.array([False, True]), 3.)


if __name__ == "__main__":
    unittest.main()
