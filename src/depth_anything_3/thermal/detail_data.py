"""Native-resolution sequences with train-only structural frame sampling."""
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageEnhance
import torch

from depth_anything_3.thermal.data import SEARDataset, read_frame


def structure_scores(scenes, max_frames=48):
    """Score labels' thin foreground dips supported by image contrast.

    Scores only affect sampling, never become depth targets. Missing structures
    in source labels may be missed; teacher boundaries are also weighted online.
    Uniform exploration remains active so these imperfect proxies cannot exclude
    other frames. Call with the training manifest, never the validation manifest.
    """
    result = {}
    kernel = np.ones((7, 7), np.uint8)
    for scene in scenes:
        folder = Path(scene["transforms"]).parent
        frames = json.loads(Path(scene["transforms"]).read_text())["frames"]
        indices = np.unique(np.linspace(0, len(frames)-1, min(max_frames, len(frames)), dtype=int))
        values = []
        for i in indices:
            frame = frames[int(i)]["rgb"]
            depth = np.load(folder / frame["depth_file_path"]).squeeze().astype(np.float32)
            valid = np.isfinite(depth) & (depth > 1e-6)
            fill = float(np.median(depth[valid])) if valid.any() else 1.0
            log_depth = cv2.resize(np.log(np.where(valid, depth, fill)),
                                   (259, 196), interpolation=cv2.INTER_NEAREST)
            mask = cv2.resize(valid.astype(np.uint8), (259, 196), interpolation=cv2.INTER_NEAREST)
            interior = cv2.erode(mask, kernel).astype(bool)
            thin = cv2.morphologyEx(log_depth, cv2.MORPH_BLACKHAT, kernel) > .04
            with Image.open(folder / frame["file_path"]) as image:
                gray = np.array(image.convert("L").resize((259, 196)))
            contrast = cv2.dilate(cv2.Canny(gray, 40, 100), np.ones((3, 3), np.uint8)) > 0
            values.append(float((thin & interior & contrast).mean()))
        result[scene["scene"]] = np.interp(np.arange(len(frames)), indices, values).tolist()
    return result


class DetailSEARDataset(SEARDataset):
    def __init__(self, *args, scores=None, sampling_boost=3.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.scores, self.sampling_boost = scores or {}, sampling_boost

    def __getitem__(self, index):
        scene_idx, sample_idx = self.items[index]
        scene, frames = self.scenes[scene_idx], self.frames[scene_idx]
        self.height, self.width = frames[0]["rgb"]["h"], frames[0]["rgb"]["w"]
        if self.height % 14 or self.width % 14:
            raise ValueError("Native dimensions must be divisible by 14")
        rng = np.random.default_rng(self.seed + index * 1009 + self.epoch * 1000003)
        if self.training:
            raw = np.asarray(self.scores.get(scene["scene"], np.zeros(len(frames))))
            if raw.shape != (len(frames),) or not np.isfinite(raw).all() or (raw < 0).any():
                raise ValueError("Invalid structural sampling scores")
            weights = 1 + self.sampling_boost * np.clip(raw / max(float(np.percentile(raw, 90)), 1e-6), 0, 1)
            span = min(len(frames), self.views * 4)
            start_weights = np.convolve(weights, np.ones(span) / span, mode="valid")
            probabilities = .5 / len(start_weights) + .5 * start_weights / start_weights.sum()
            start = int(rng.choice(len(start_weights), p=probabilities))
            local = weights[start:start+span]
            probabilities = .5 / span + .5 * local / local.sum()
            indices = rng.choice(np.arange(start, start+span), self.views, replace=False, p=probabilities)
            ratio = float(rng.uniform())
            mask = rng.random(self.views) < ratio
            mode = "train"
        else:
            indices = np.linspace(0, len(frames)-1, self.views, dtype=int)
            mode = ("rgb", "mixed", "thermal")[sample_idx]
            mask = np.zeros(self.views, dtype=bool)
            if mode == "thermal":
                mask[:] = True
            elif mode == "mixed":
                mask[1::2] = True
        # If there is RGB, reference the same first RGB camera in both networks.
        if not mask.all() and mask[0]:
            first_rgb = int(np.flatnonzero(~mask)[0])
            indices[[0, first_rgb]] = indices[[first_rgb, 0]]
            mask[[0, first_rgb]] = mask[[first_rgb, 0]]
        images, depths, extrinsics, intrinsics, hard = [], [], [], [], []
        gain, gamma = rng.uniform(.85, 1.15), rng.uniform(.85, 1.15)
        folder = Path(scene["transforms"]).parent
        for v, frame_idx in enumerate(indices):
            frame = frames[int(frame_idx)]["thermal" if mask[v] else "rgb"]
            image, depth, ext, intr = read_frame(folder, frame, self.height, self.width)
            if self.training and not mask[v]:
                image = ImageEnhance.Brightness(image).enhance(gain)
            a = np.asarray(image).astype(np.float32) / 255
            if self.training and mask[v]:
                a = np.clip(gain * np.power(a, gamma), 0, 1)
            images.append(torch.from_numpy(a.copy()).permute(2, 0, 1))
            depths.append(torch.from_numpy(depth.copy()))
            extrinsics.append(torch.from_numpy(ext))
            intrinsics.append(torch.from_numpy(intr))
            hard.append(frame.get("depth_source") == "COLMAP geometric MVS")
        images = torch.stack(images)
        images = (images - images.new_tensor([.485, .456, .406])[None, :, None, None]) / images.new_tensor([.229, .224, .225])[None, :, None, None]
        depths = torch.stack(depths)
        valid = torch.isfinite(depths) & (depths > 1e-6)
        if valid.sum() < 100:
            raise ValueError("Insufficient valid geometry")
        depths = torch.where(valid, depths, 0)
        extrinsics = torch.stack(extrinsics)
        extrinsics = extrinsics @ torch.linalg.inv(extrinsics[0])
        scale = depths[valid].median().clamp_min(1e-6)
        depths /= scale
        extrinsics[:, :3, 3] /= scale
        return {"images": images, "depth": depths, "valid": valid,
                "extrinsics": extrinsics, "intrinsics": torch.stack(intrinsics),
                "thermal_mask": torch.from_numpy(mask), "colmap_mask": torch.tensor(hard),
                "frame_indices": torch.from_numpy(indices.copy()),
                "scene": scene["scene"], "mode": mode}
