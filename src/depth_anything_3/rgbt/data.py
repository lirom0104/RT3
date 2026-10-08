"""Paired data preparation with explicit scene splits and camera conventions."""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from depth_anything_3.utils.read_write_model import read_model


def camera_parameters(camera):
    p = camera.params
    if camera.model in ("PINHOLE", "OPENCV", "FULL_OPENCV"):
        fx, fy, cx, cy = p[:4]
        distortion = np.array(p[4:], np.float64) if len(p) > 4 else np.zeros(4)
    elif camera.model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"):
        fx, cx, cy = p[:3]
        fy = fx
        distortion = np.zeros(4)
        distortion[: len(p) - 3] = p[3:]
    else:
        raise ValueError(f"Unsupported COLMAP model: {camera.model}")
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], np.float32), distortion


def grayscale(path):
    """Fixed display luminance for pseudocolor, preserving uint16 precision otherwise.

    Pseudocolor luminance is not inverse-palette temperature recovery. No per-frame
    min/max stretch is applied to 8-bit multi-view data.
    """
    with Image.open(path) as im:
        a = np.asarray(im)
    if a.ndim == 3:
        return (
            a[..., :3].astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32)
        ) / 255
    if a.dtype == np.uint8:
        return a.astype(np.float32) / 255
    return a.astype(np.float32) / 65535


def prepare_dataset(root, output, size=(168, 224)):
    root, output = Path(root), Path(output)
    (output / "frames").mkdir(parents=True, exist_ok=True)
    scenes = []
    audit = []
    val_names = {"RGBT-Scenes/DailyStuff", "ThermoScenes/heater_water_kettle"}
    test_names = {
        "RGBT-Scenes/Building",
        "RGBT-Scenes/Parterre",
        "ThermoScenes/raspberrypi",
        "ThermoScenes/trees",
    }
    for domain in ["RGBT-Scenes", "ThermoScenes"]:
        for scene in sorted((root / domain).iterdir()):
            if not scene.is_dir():
                continue
            sparse = scene / "sparse/0"
            if not (sparse / "images.bin").is_file():
                continue
            cams, images, points = read_model(str(sparse), ext=".bin")
            rgb_paths = {
                p.stem.lower(): p
                for p in (scene / "rgb").rglob("*")
                if p.suffix.lower() in (".png", ".jpg", ".jpeg")
            }
            thermal_paths = {
                p.stem.lower(): p
                for p in (scene / "thermal").rglob("*")
                if p.suffix.lower() in (".png", ".jpg", ".jpeg")
            }
            key = f"{domain}/{scene.name}"
            split = (
                "val" if key in val_names else "test" if key in test_names else "train"
            )
            records = []
            h, w = size if domain == "RGBT-Scenes" else size[::-1]
            for im in sorted(images.values(), key=lambda im: im.name):
                stem = Path(im.name).stem.lower()
                if stem not in rgb_paths or stem not in thermal_paths:
                    continue
                camera = cams[im.camera_id]
                K, dist = camera_parameters(camera)
                with Image.open(rgb_paths[stem]) as pil:
                    rgb = np.asarray(pil.convert("RGB"), np.float32) / 255
                t = grayscale(thermal_paths[stem])
                if rgb.shape[:2] != t.shape:
                    raise ValueError(f"Unregistered image sizes: {key}/{stem}")
                if (rgb.shape[1], rgb.shape[0]) != (camera.width, camera.height):
                    raise ValueError("Camera/image resolution mismatch")
                new_K = K.copy()
                new_K[0] *= w / camera.width
                new_K[1] *= h / camera.height
                mx, my = cv2.initUndistortRectifyMap(
                    K, dist, None, new_K, (w, h), cv2.CV_32FC1
                )
                r = cv2.remap(rgb, mx, my, cv2.INTER_LINEAR)
                t = cv2.remap(t, mx, my, cv2.INTER_LINEAR)
                valid = (
                    (mx >= 0)
                    & (mx < camera.width - 1)
                    & (my >= 0)
                    & (my < camera.height - 1)
                ).astype(np.float32)
                E = np.eye(4, dtype=np.float32)
                E[:3, :3] = im.qvec2rotmat()
                E[:3, 3] = im.tvec
                dep = np.zeros((h, w), np.float32)
                good = [
                    p
                    for p in im.point3D_ids
                    if p in points
                    and points[p].error < 2
                    and len(points[p].image_ids) >= 3
                ]
                if good:
                    xyz = np.array([points[p].xyz for p in good], np.float32)
                    xyz = xyz @ E[:3, :3].T + E[:3, 3]
                    uv = xyz @ new_K.T
                    uv = uv[:, :2] / np.maximum(uv[:, 2:], 1e-6)
                    uv = np.rint(uv).astype(int)
                    inside = (
                        (xyz[:, 2] > 0)
                        & (uv[:, 0] >= 0)
                        & (uv[:, 0] < w)
                        & (uv[:, 1] >= 0)
                        & (uv[:, 1] < h)
                    )
                    dep[uv[inside, 1], uv[inside, 0]] = xyz[inside, 2]
                frame_id = f"{domain}_{scene.name}_{stem}"
                dest = output / "frames" / f"{frame_id}.npz"
                np.savez_compressed(
                    dest,
                    rgb=r.transpose(2, 0, 1),
                    thermal=t[None],
                    valid=valid[None],
                    depth=dep,
                    K=new_K,
                    E=E,
                )
                records.append(
                    {
                        "id": frame_id,
                        "cache": str(dest.resolve()),
                        "name": im.name,
                        "center": np.linalg.inv(E)[:3, 3].tolist(),
                        "forward": E[2, :3].tolist(),
                        "native_eval": (
                            "eval" in stem or "test" in str(rgb_paths[stem].parent.name)
                        ),
                    }
                )
            if len(records) < 4:
                raise ValueError(f"Too few paired registered frames: {key}")
            # Reserve target images within each scene; held-out scenes were never optimized.
            target_idx = [i for i, r in enumerate(records) if r["native_eval"]]
            if not target_idx:
                target_idx = list(range(0, len(records), 8))
            contexts = [i for i in range(len(records)) if i not in target_idx]
            scenes.append(
                {
                    "id": key,
                    "split": split,
                    "domain": domain,
                    "frames": records,
                    "context_pool": contexts,
                    "target_pool": target_idx,
                    "size": [h, w],
                }
            )
            audit.append(
                {
                    "scene": key,
                    "split": split,
                    "paired_registered": len(records),
                    "render_targets": len(target_idx),
                    "camera_models": sorted({c.model for c in cams.values()}),
                    "registration": "Paired dataset grid assumed; both images undistorted with the RGB camera. No separate thermal extrinsics supplied.",
                }
            )
            print("PREPARED", key, split, len(records), flush=True)
    # MM5 entries are independent captures. Never interpret numeric IDs as camera views.
    mm5 = []
    mmroot = root / "MM5_ALIGNED"
    for rp in sorted((mmroot / "RGB3").glob("*.png"), key=lambda p: int(p.stem)):
        stem = rp.stem
        meta = json.loads((mmroot / "META" / f"{stem}.json").read_text())
        group = str(meta["sequence"])
        bucket = int(hashlib.sha256(group.encode()).hexdigest()[:8], 16) % 10
        split = "test" if bucket == 0 else "val" if bucket == 1 else "train"
        with Image.open(rp) as im:
            r = np.asarray(im.convert("RGB"), np.float32) / 255
        with Image.open(mmroot / "T16" / rp.name) as im:
            t = np.asarray(im, np.float32)
        lo, hi = np.percentile(t, [1, 99])
        t = np.clip((t - lo) / max(hi - lo, 1), 0, 1)
        with Image.open(mmroot / "D16" / rp.name) as im:
            d = np.asarray(im, np.float32)
        h, w = size
        r = cv2.resize(r, (w, h), interpolation=cv2.INTER_AREA)
        t = cv2.resize(t, (w, h), interpolation=cv2.INTER_AREA)
        d = cv2.resize(d, (w, h), interpolation=cv2.INTER_NEAREST)
        dest = output / "frames" / f"MM5_{stem}.npz"
        np.savez_compressed(
            dest,
            rgb=r.transpose(2, 0, 1),
            thermal=t[None],
            depth=d,
            valid=np.ones((1, h, w), np.float32),
        )
        mm5.append(
            {
                "id": f"MM5_{stem}",
                "cache": str(dest.resolve()),
                "split": split,
                "sequence": group,
            }
        )
    manifest = {"scenes": scenes, "mm5": mm5}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (output / "protocol.json").write_text(
        json.dumps(
            {
                "image_size": list(size),
                "audit": audit,
                "mm5": {
                    "count": len(mm5),
                    "usage": "Single-view auxiliary relative depth supervision; no NVS claims",
                    "split": "Sequence-grouped SHA256 split",
                    "thermal": "T16, 1st/99th percentile normalized per independent capture",
                },
                "rgbt_thermal": "Fixed 0.299 R + 0.587 G + 0.114 B luminance / 255, not temperature",
                "source_images_modified": False,
                "validation": "Scene-disjoint; validation selects the checkpoint",
                "test": "Scene-disjoint; evaluated only after checkpoint selection",
                "poses": "Existing COLMAP camera calibration; images used by original COLMAP may include held-out targets",
                "assumptions": "Static scene, stable thermal appearance, paired RGB/T grid correspondence",
            },
            indent=2,
        )
    )
    return manifest


class SceneDataset(Dataset):
    def __init__(
        self,
        root,
        split,
        views=3,
        targets_per_scene=6,
        training=False,
        sampling="nearest",
    ):
        self.root = Path(root)
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        self.scenes = [s for s in self.manifest["scenes"] if s["split"] == split]
        self.views = views
        self.training = training
        if sampling not in ("nearest", "diverse", "wide"):
            raise ValueError(sampling)
        self.sampling = sampling
        self.items = []
        for si, s in enumerate(self.scenes):
            targets = s["context_pool"] if training else s["target_pool"]
            if not training:
                targets = [
                    targets[j]
                    for j in np.linspace(
                        0, len(targets) - 1, min(targets_per_scene, len(targets))
                    )
                    .round()
                    .astype(int)
                ]
            self.items.extend((si, t) for t in targets)
        self.mm5 = [r for r in self.manifest["mm5"] if r["split"] == split]

    def __len__(self):
        return len(self.items)

    @staticmethod
    @lru_cache(maxsize=96)
    def frame(path):
        with np.load(path) as z:
            return {k: torch.from_numpy(z[k].copy()).float() for k in z.files}

    def get_mm5(self, index):
        r = self.mm5[index % len(self.mm5)]
        out = self.frame(r["cache"]).copy()
        out["id"] = r["id"]
        return out

    def __getitem__(self, index):
        si, ti = self.items[index]
        s = self.scenes[si]
        target = s["frames"][ti]
        positions = np.array([f["center"] for f in s["frames"]])
        direction = np.array([f["forward"] for f in s["frames"]])
        dist = np.linalg.norm(positions - positions[ti], axis=1)
        penalty = np.maximum(0, 1 - direction @ direction[ti])
        score = dist + penalty * max(np.median(dist), 1e-3) * 2
        candidates = [i for i in s["context_pool"] if i != ti]
        candidates = sorted(candidates, key=lambda i: score[i])
        if self.sampling != "nearest":
            # Keep visual overlap with the target, but spread source camera centers
            # and directions. Only poses are used; target pixels/depth are excluded.
            gap = self.views if self.sampling == "wide" else 0
            candidates = candidates[gap:]
            limit = max(self.views * 6, 24)
            if self.training:
                limit = int(np.random.randint(self.views * 3, self.views * 6 + 1))
            pool = candidates[:limit]
            context = [pool[0]]
            radius = max(float(np.median(dist[pool])), 1e-6)
            while len(context) < self.views:
                remaining = [j for j in pool if j not in context]
                if not remaining:
                    raise ValueError(f"Too few context frames in {s['id']}")

                def coverage(j):
                    spatial = (
                        np.linalg.norm(positions[context] - positions[j], axis=-1)
                        / radius
                    )
                    angular = np.maximum(0, 1 - direction[context] @ direction[j])
                    return float(np.min(spatial + angular)) / (1 + score[j] / radius)

                context.append(max(remaining, key=coverage))
        elif self.training:
            pool = candidates[: max(self.views + 2, 8)]
            context = [candidates[0]] + list(
                np.random.choice(pool[1:], self.views - 1, replace=False)
            )
        else:
            context = candidates[: self.views]
        chosen = [s["frames"][i] for i in context] + [target]
        fs = [self.frame(f["cache"]) for f in chosen]
        tensors = {k: torch.stack([f[k] for f in fs]) for k in fs[0]}
        # Normalize pose frame and baseline using only the source cameras.
        E = tensors["E"]
        anchor = torch.linalg.inv(E[0])
        E = E @ anchor
        centers = torch.linalg.inv(E)[: self.views, :3, 3]
        scale = torch.pdist(centers).median().clamp_min(1e-4)
        E[:, :3, 3] /= scale
        tensors["E"] = E
        tensors["depth"] = tensors["depth"] / scale
        tensors.update(
            id=target["id"],
            scene=s["id"],
            context_ids=[s["frames"][i]["id"] for i in context],
        )
        assert target["id"] not in tensors["context_ids"]
        return tensors
