#!/usr/bin/env python3
"""Freeze additional development and untouched test scenes before evaluation."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from depth_anything_3.rgbt.data import prepare_dataset
from depth_anything_3.utils.read_write_model import read_model, write_model

# Declared before any quantitative evaluation. Neither split is training data.
SCENES = {
    "ThermalMix_lion": ("/home/lf/data/ThermalMix/lion", "colmap", "val"),
    "ThermalMix_panel": ("/home/lf/data/ThermalMix/panel", "colmap", "val"),
    "ThermalMix_laptop": ("/home/lf/data/ThermalMix/laptop", "colmap", "test"),
    "ThermalMix_pan": ("/home/lf/data/ThermalMix/pan", "colmap", "test"),
    "Extended_DarkScenes": (
        "/home/lf/data/thermal3dgs/RGBT-Scenes-extend/DarkScenes",
        "sparse/0",
        "test",
    ),
    "Extended_GlassCup": (
        "/home/lf/data/thermal3dgs/RGBT-Scenes-extend/Glass Cup",
        "sparse/0",
        "test",
    ),
    "Capture_chazuo": ("/home/lf/data/chazuo", "sparse/0", "test"),
}

p = argparse.ArgumentParser()
p.add_argument("--output", default="outputs/rgbt/datasets/external_336")
a = p.parse_args()
out = Path(a.output).resolve()
if out.exists():
    raise ValueError("External split must be created in a fresh directory")
out.mkdir(parents=True)
staging = out / "staging"
(staging / "ThermoScenes").mkdir(parents=True)
locked = {
    name: {"root": root, "sparse": sparse, "split": split}
    for name, (root, sparse, split) in SCENES.items()
}
(out / "split_lock.json").write_text(json.dumps(locked, indent=2))
for name, (root, sparse, split) in SCENES.items():
    root = Path(root)
    stage = staging / "RGBT-Scenes" / name
    (stage / "sparse/0").mkdir(parents=True)
    for folder in ["rgb", "thermal"]:
        if name == "Capture_chazuo" and folder == "thermal":
            for image in (root / folder).rglob("*.png"):
                rel = image.relative_to(root / folder)
                dest = stage / folder / rel.parent / rel.name.replace("_IR", "_RGB")
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.symlink_to(image)
        else:
            (stage / folder).symlink_to(root / folder, target_is_directory=True)
    folder = root / sparse
    ext = ".bin" if (folder / "images.bin").exists() else ".txt"
    cameras, images, points = read_model(str(folder), ext=ext)
    write_model(cameras, images, points, str(stage / "sparse/0"), ext=".bin")
manifest = prepare_dataset(staging, out, (252, 336))
for scene in manifest["scenes"]:
    name = scene["id"].split("/")[-1]
    scene["id"] = "External/" + name
    scene["domain"] = "External"
    scene["split"] = SCENES[name][2]
(out / "manifest.json").write_text(json.dumps(manifest, indent=2))
protocol = json.loads((out / "protocol.json").read_text())
protocol.update(
    external_split=locked,
    training_usage="None: original 14 scenes remain the only training scenes",
    test_policy="Test scenes locked before model development. Do not inspect test metrics or use them for selection.",
    thermal_limit="Display intensity only; provided paired registration assumed. Dynamic/radiometric validity not certified.",
    sha256_manifest=hashlib.sha256((out / "manifest.json").read_bytes()).hexdigest(),
)
for audit in protocol["audit"]:
    name = audit["scene"].split("/")[-1]
    audit["scene"] = "External/" + name
    audit["split"] = SCENES[name][2]
(out / "protocol.json").write_text(json.dumps(protocol, indent=2))
print(
    "LOCKED",
    [(s["id"], s["split"], len(s["frames"])) for s in manifest["scenes"]],
    flush=True,
)
