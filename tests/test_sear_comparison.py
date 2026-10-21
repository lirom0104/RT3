import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import open3d as o3d
from PIL import Image


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/infer_sear_comparison.py"
SPEC = importlib.util.spec_from_file_location("infer_sear_comparison", SCRIPT)
comparison = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(comparison)


class ComparisonTests(unittest.TestCase):
    def test_discovery_prefers_processed_images_and_needs_no_calibration(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            scene = root / "dataset/scene"
            processed = scene / "VGGT"
            processed.mkdir(parents=True)
            Image.new("RGB", (28, 28)).save(processed / "rgb.png")
            Image.new("RGB", (28, 28)).save(processed / "thermal.png")
            frame = {"rgb": {"file_path": "rgb.png"}, "thermal": {"file_path": "thermal.png"}}
            (processed / "transforms.json").write_text(json.dumps({"frames": [frame] * 25}))
            (scene / "transforms.json").write_text(json.dumps({"frames": [frame]}))
            scenes = comparison.discover_scenes(root, 12)
            self.assertEqual(len(scenes), 1)
            result = scenes[0]
            self.assertEqual(result["total_pairs"], 25)
            self.assertEqual(len(result["selected_pair_indices"]), 12)
            self.assertEqual(result["selected_pair_indices"][0], 0)
            self.assertEqual(result["selected_pair_indices"][-1], 24)
            self.assertEqual([v["modality"] for v in result["views"]], ["rgb"] * 12 + ["thermal"] * 12)
            self.assertEqual(Path(result["transforms"]), processed / "transforms.json")

    def test_point_projection_and_ply_can_be_read_by_open3d(self):
        depth = np.full((2, 2), 2.0, dtype=np.float32)
        intrinsic = np.eye(3, dtype=np.float32)
        extrinsic = np.column_stack((np.eye(3), [1, 2, 3]))
        world = comparison.depth_to_world(depth, extrinsic, intrinsic)
        np.testing.assert_allclose(world[0, 0], [-1, -2, -1])
        np.testing.assert_allclose(world[1, 1], [1, 0, -1])
        colors = np.tile([1.0, 0.0, 0.5], (4, 1))
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "cloud.ply"
            comparison.write_ply(path, world.reshape(-1, 3), colors)
            loaded = o3d.io.read_point_cloud(str(path))
            np.testing.assert_allclose(np.asarray(loaded.points), world.reshape(-1, 3))
            np.testing.assert_allclose(np.asarray(loaded.colors), np.tile([1, 0, 128 / 255], (4, 1)))

    def test_resume_rejects_changed_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            image = folder / "depth.png"
            Image.new("RGB", (4, 4)).save(image)
            record = {"files": comparison.fingerprints(folder)}
            comparison.verify_completed(folder, record)
            image.write_bytes(b"changed")
            with self.assertRaises(ValueError):
                comparison.verify_completed(folder, record)


if __name__ == "__main__":
    unittest.main()
