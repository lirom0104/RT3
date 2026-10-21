import unittest

from depth_anything_3.thermal.resume import map_resume_batch


class ResumeTests(unittest.TestCase):
    def test_two_gpus_to_one_preserves_prefix_and_steps(self):
        old = {"world_size": 2, "accumulation": 1}
        new = {"world_size": 1, "accumulation": 2}
        self.assertEqual(map_resume_batch(old, new, 259, 598, True), 518)
        self.assertEqual(map_resume_batch(old, new, 299, 598, True), 598)
        self.assertEqual(map_resume_batch(old, new, 0, 598, True), 0)
        # An odd dataset has one padded DDP sample at the end of the epoch.
        self.assertEqual(map_resume_batch(old, new, 300, 599, True), 599)

    def test_requires_explicit_change_and_same_effective_batch(self):
        old = {"world_size": 2, "accumulation": 1}
        with self.assertRaises(ValueError):
            map_resume_batch(old, {"world_size": 1, "accumulation": 2}, 259, 598)
        with self.assertRaises(ValueError):
            map_resume_batch(old, {"world_size": 1, "accumulation": 1}, 259, 598, True)

    def test_unchanged_layout(self):
        settings = {"world_size": 1, "accumulation": 2}
        self.assertEqual(map_resume_batch(settings, settings, 518, 598), 518)


if __name__ == "__main__":
    unittest.main()
