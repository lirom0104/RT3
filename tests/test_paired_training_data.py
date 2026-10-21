import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch

from depth_anything_3.thermal.data import SEARDataset


class PairedTrainingDataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        folder = Path(self.temp.name)
        frames = []
        for index in range(16):
            pair = {}
            for thermal, modality in enumerate(("rgb", "thermal")):
                image = np.zeros((28, 42, 3), dtype=np.uint8)
                image[..., thermal] = 20 + index + 40 * thermal
                image_path = f"{modality}_{index}.png"
                depth_path = f"{modality}_{index}.npy"
                Image.fromarray(image).save(folder / image_path)
                np.save(folder / depth_path, np.full((28, 42), 2 + index + 2 * thermal, np.float32))
                extrinsic = np.eye(4)
                extrinsic[0, 3] = index * .1 + thermal * .03
                pair[modality] = {
                    "file_path": image_path, "depth_file_path": depth_path,
                    "transform_matrix": extrinsic.tolist(), "w": 42, "h": 28,
                    "fl_x": 30 + thermal * 5, "fl_y": 31 + thermal * 3,
                    "cx": 21, "cy": 14,
                }
            frames.append(pair)
        transforms = folder / "transforms.json"
        transforms.write_text(json.dumps({"frames": frames}))
        self.scenes = [{"scene": "synthetic", "transforms": str(transforms)}]

    def dataset(self, training, paired_inputs=True):
        return SEARDataset(self.scenes, views=8, height=28, width=42, training=training,
                           seed=0, paired_inputs=paired_inputs)

    def test_training_keeps_batch_count_and_eight_matching_pairs(self):
        original, paired = self.dataset(True, False), self.dataset(True)
        self.assertEqual(len(paired), len(original))
        batch = paired[0]
        self.assertEqual(tuple(batch["images"].shape), (16, 3, 28, 42))
        self.assertEqual(batch["thermal_mask"].tolist(), [False] * 8 + [True] * 8)
        torch.testing.assert_close(batch["frame_indices"][:8], batch["frame_indices"][8:])
        self.assertEqual(len(set(batch["frame_indices"][:8].tolist())), 8)
        paired.epoch = 1
        self.assertFalse(torch.equal(batch["frame_indices"], paired[0]["frame_indices"]))

    def test_paired_validation_preserves_each_modality_geometry_and_images(self):
        data = self.dataset(False)
        self.assertEqual([data[i]["mode"] for i in range(len(data))],
                         ["rgb", "mixed", "thermal", "paired"])
        batch = data[3]
        indices = batch["frame_indices"]
        mask = batch["thermal_mask"]
        raw_depth = 2 + indices.float() + 2 * mask.float()
        scale = raw_depth.median()
        torch.testing.assert_close(batch["depth"][:, 0, 0] * scale, raw_depth)
        expected_translation = (indices.float() - indices[0]) * .1 + mask.float() * .03
        torch.testing.assert_close(batch["extrinsics"][:, 0, 3] * scale, expected_translation)
        torch.testing.assert_close(batch["intrinsics"][:, 0, 0], 30 + mask.float() * 5)
        pixels = batch["images"] * torch.tensor([.229, .224, .225])[None, :, None, None]
        pixels += torch.tensor([.485, .456, .406])[None, :, None, None]
        for i in range(16):
            channel = int(mask[i])
            self.assertAlmostEqual(float(pixels[i, channel, 0, 0] * 255),
                                   20 + int(indices[i]) + 40 * channel, places=4)

    def test_existing_sampling_and_validation_remain_available(self):
        train = self.dataset(True, False)[0]
        self.assertEqual(train["images"].shape[0], 8)
        self.assertEqual(len(set(train["frame_indices"].tolist())), 8)
        validation = self.dataset(False, False)
        self.assertEqual([validation[i]["mode"] for i in range(len(validation))],
                         ["rgb", "mixed", "thermal"])
        self.assertTrue(all(validation[i]["images"].shape[0] == 8 for i in range(3)))


if __name__ == "__main__":
    unittest.main()
