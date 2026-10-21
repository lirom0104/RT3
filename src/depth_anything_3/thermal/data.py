"""Read SEAR's processed paired transforms, preserving each modality's calibration."""

import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance
import torch
from torch.utils.data import Dataset


def canonical_name(name):
    return "".join(name.split()).lower()


def build_manifest(root, official_split, difficult_scenes):
    root = Path(root).resolve()
    split = json.loads(Path(official_split).read_text())
    names = {k: {canonical_name(s) for s in v} for k, v in split.items()}
    difficult = {canonical_name(s) for s in difficult_scenes}
    result = {"root": str(root), "train": [], "eval": [], "held_out": [], "excluded": []}
    required = ("file_path", "depth_file_path", "transform_matrix", "w", "h",
                "fl_x", "fl_y", "cx", "cy")
    for path in sorted(root.rglob("transforms.json")):
        content = json.loads(path.read_text())
        frames = content.get("frames", [])
        if not frames or not all("rgb" in f and "thermal" in f for f in frames):
            continue
        scene_name = path.parent.parent.name if path.parent.name == "VGGT" else path.parent.name
        name = canonical_name(scene_name)
        record = {"scene": str(path.parent.relative_to(root)), "pairs": len(frames),
                  "transforms": str(path), "name": scene_name}
        problems = []
        for i, frame in enumerate(frames):
            for modality in ("rgb", "thermal"):
                f = frame[modality]
                missing = [k for k in required if k not in f]
                if missing:
                    problems.append(f"frame {i} {modality}: missing {missing}")
                    continue
                for k in ("file_path", "depth_file_path"):
                    if not (path.parent / f[k]).is_file():
                        problems.append(f"frame {i} {modality}: missing file {f[k]}")
        if problems:
            record.update(reason=problems[:8], problem_count=len(problems))
            result["excluded"].append(record)
        elif name in names["train"]:
            result["train"].append(record)
        elif name in names["eval"]:
            result["eval"].append(record)
        else:
            record["reason"] = "SEAR difficult scene" if name in difficult else "Not in official split"
            result["held_out"].append(record)
    if not result["train"] or not result["eval"]:
        raise ValueError("No usable train/eval scenes found")
    return result


def read_frame(folder, frame, height, width):
    with Image.open(folder / frame["file_path"]) as im:
        image = im.convert("RGB")
    depth = np.load(folder / frame["depth_file_path"]).squeeze().astype(np.float32)
    if image.size != (frame["w"], frame["h"]) or depth.shape != (frame["h"], frame["w"]):
        raise ValueError(f"Image/depth/calibration dimensions disagree: {folder / frame['file_path']}")
    # Aspect-preserving resize followed by center crop; apply identical geometry
    # to image, depth and intrinsic parameters. Nearest keeps invalid depth holes.
    scale = max(width / image.width, height / image.height)
    rw, rh = max(width, round(image.width * scale)), max(height, round(image.height * scale))
    left, top = (rw - width) // 2, (rh - height) // 2
    box = (left, top, left + width, top + height)
    image = image.resize((rw, rh), Image.Resampling.BILINEAR).crop(box)
    depth = np.array(Image.fromarray(depth).resize((rw, rh), Image.Resampling.NEAREST).crop(box))
    sx, sy = rw / frame["w"], rh / frame["h"]
    intrinsic = np.array([[frame["fl_x"] * sx, 0, frame["cx"] * sx - left],
                          [0, frame["fl_y"] * sy, frame["cy"] * sy - top],
                          [0, 0, 1]], dtype=np.float32)
    extrinsic = np.eye(4, dtype=np.float32)
    extrinsic[:3] = np.asarray(frame["transform_matrix"], dtype=np.float32)[:3]
    return image, depth, extrinsic, intrinsic


class SEARDataset(Dataset):
    def __init__(self, scenes, views=4, height=252, width=336, training=True, seed=0,
                 paired_inputs=False):
        if views < 2 or height % 14 or width % 14:
            raise ValueError("Use at least 2 views and dimensions divisible by 14")
        self.scenes = scenes
        self.views, self.height, self.width = views, height, width
        self.training, self.seed, self.epoch = training, seed, 0
        # views counts sampled frame indices. Paired training loads both modalities
        # at each index, preserving the original number of batches per epoch.
        self.paired_inputs = paired_inputs
        self.eval_modes = ("rgb", "mixed", "thermal", "paired") if paired_inputs else ("rgb", "mixed", "thermal")
        self.frames = [json.loads(Path(s["transforms"]).read_text())["frames"] for s in scenes]
        self.items = []
        for i, frames in enumerate(self.frames):
            if len(frames) < views:
                raise ValueError(f"Too few paired frames: {scenes[i]['scene']}")
            count = math.ceil(len(frames) / views) if training else len(self.eval_modes)
            self.items.extend((i, j) for j in range(count))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        scene_idx, sample_idx = self.items[index]
        rng = np.random.default_rng(self.seed + index * 1009 + self.epoch * 1000003)
        scene, frames = self.scenes[scene_idx], self.frames[scene_idx]
        if self.training:
            # Nearby views retain overlap, with varying baselines and reference frames.
            span = min(len(frames), self.views * 4)
            start = int(rng.integers(len(frames) - span + 1))
            indices = rng.choice(np.arange(start, start + span), self.views, replace=False)
            if self.paired_inputs:
                mask = np.zeros(self.views, dtype=bool)
            else:
                ratio = float(rng.uniform())
                mask = rng.random(self.views) < ratio
            mode = "train"
        else:
            indices = np.linspace(0, len(frames) - 1, self.views, dtype=int)
            mode = self.eval_modes[sample_idx]
            mask = np.zeros(self.views, dtype=bool)
            if mode == "thermal":
                mask[:] = True
            elif mode == "mixed":
                mask[::2] = True
        if self.paired_inputs and (self.training or mode == "paired"):
            # Match inference: all RGB views, then thermal views in the same order.
            indices = np.tile(indices, 2)
            mask = np.repeat(np.array([False, True]), self.views)
        images, depths, extrinsics, intrinsics = [], [], [], []
        folder = Path(scene["transforms"]).parent
        gain, gamma = rng.uniform(.85, 1.15), rng.uniform(.85, 1.15)
        for i, frame_idx in enumerate(indices):
            # Some source files have incorrect 'type' fields; the paired dictionary
            # key, never that field, determines modality.
            f = frames[int(frame_idx)]["thermal" if mask[i] else "rgb"]
            image, depth, ext, intr = read_frame(folder, f, self.height, self.width)
            if self.training and not mask[i]:
                image = ImageEnhance.Brightness(image).enhance(gain)
            a = np.asarray(image).astype(np.float32) / 255
            if self.training and mask[i]:
                a = np.clip(gain * np.power(a, gamma), 0, 1)
            images.append(torch.from_numpy(a.copy()).permute(2, 0, 1))
            depths.append(torch.from_numpy(depth.copy()))
            extrinsics.append(torch.from_numpy(ext))
            intrinsics.append(torch.from_numpy(intr))
        images = torch.stack(images)
        images = (images - images.new_tensor([.485, .456, .406])[None, :, None, None]) / images.new_tensor([.229, .224, .225])[None, :, None, None]
        depths = torch.stack(depths)
        valid = torch.isfinite(depths) & (depths > 1e-6)
        if valid.sum() < 100:
            raise ValueError(f"Insufficient valid depth: {scene['scene']}")
        depths = torch.where(valid, depths, 0)
        extrinsics = torch.stack(extrinsics)
        # OpenCV world-to-camera, referenced to the actual first sampled camera.
        extrinsics = extrinsics @ torch.linalg.inv(extrinsics[0])
        scale = depths[valid].median().clamp_min(1e-6)
        depths = depths / scale
        extrinsics[:, :3, 3] /= scale
        return {"images": images, "depth": depths, "valid": valid,
                "extrinsics": extrinsics, "intrinsics": torch.stack(intrinsics),
                "frame_indices": torch.from_numpy(indices.copy()).long(),
                "thermal_mask": torch.from_numpy(mask), "scene": scene["scene"], "mode": mode}
