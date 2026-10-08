#!/usr/bin/env python3
"""Official DA3 source-only Gaussian render at Building target img_025."""

import json
import sys
from dataclasses import replace
from pathlib import Path

import matplotlib
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


PROJECT = Path(__file__).resolve().parents[1]
OFFICIAL = Path("/home/lf/feedforward/DA3base")
sys.path.insert(0, str(OFFICIAL / "src"))

from depth_anything_3.api import DepthAnything3  # noqa: E402
from depth_anything_3.model.utils.gs_renderer import render_3dgs  # noqa: E402
from depth_anything_3.utils.pose_align import align_poses_umeyama  # noqa: E402


WEIGHTS = OFFICIAL / "checkpoints/DA3-GIANT-1.1"
DATA = PROJECT / "outputs/rgbt/data_336/frames"
DEPTH_TEST = PROJECT / "outputs/rgbt/depth_test_20260922/RGBT-Scenes/Building/nearest3/raw"
OUTPUT = PROJECT / "outputs/rgbt/official_da3_building_nearest3_20260923/target_025"
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
SOURCE_IDS = (26, 24, 27)
TARGET_ID = 25


def camera_in_gs_frame(reference_extrinsic, rotation, translation, scale):
    camera_reference = np.linalg.inv(reference_extrinsic)
    camera_gs = np.eye(4, dtype=np.float32)
    camera_gs[:3, :3] = rotation.T @ camera_reference[:3, :3]
    camera_gs[:3, 3] = rotation.T @ (camera_reference[:3, 3] - translation) / scale
    return np.linalg.inv(camera_gs).astype(np.float32)


def homogeneous(extrinsics):
    if extrinsics.shape[-2:] == (4, 4):
        return extrinsics
    result = np.broadcast_to(np.eye(4, dtype=extrinsics.dtype), (*extrinsics.shape[:-2], 4, 4)).copy()
    result[..., :3, :4] = extrinsics
    return result


def visualize(depth, valid, limits):
    lo, hi = limits
    values = np.log(np.maximum(depth, lo) / lo) / np.log(hi / lo)
    colors = (matplotlib.colormaps["magma"](values.clip(0, 1))[..., :3] * 255 + 0.5).astype(np.uint8)
    yy, xx = np.indices(depth.shape)
    checker = np.where((xx // 8 + yy // 8) % 2, 105, 135).astype(np.uint8)
    bad = ~valid | ~np.isfinite(depth) | (depth <= 0)
    colors[bad] = checker[bad, None]
    return Image.fromarray(colors)


def make_panel(path, arrays, masks, limits, title, note):
    h, w = arrays[0].shape
    gap, label, header, footer = 8, 30, 65, 80
    canvas = Image.new("RGB", (3 * w + 4 * gap, header + 2 * (h + label + gap) + footer), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = ImageFont.truetype(FONT, 17)
    label_font = ImageFont.truetype(FONT, 12)
    small_font = ImageFont.truetype(FONT, 10)
    draw.text((gap, 8), title, font=title_font, fill="#171717")
    draw.text((gap, 35), note, font=small_font, fill="#555555")
    names = ("COLMAP MVS reference", "Official DA3: all 3 sources", "Only source 026", "Only source 024", "Only source 027")
    for i, (name, depth, valid) in enumerate(zip(names, arrays, masks)):
        x = gap + (i % 3) * (w + gap)
        y = header + (i // 3) * (h + label + gap)
        draw.text((x, y), name, font=label_font, fill="#171717")
        canvas.paste(visualize(depth, valid, limits), (x, y + label))
    bar_y = canvas.height - footer + 6
    ramp = np.geomspace(*limits, canvas.width - 2 * gap).astype(np.float32)[None].repeat(12, axis=0)
    canvas.paste(visualize(ramp, np.ones_like(ramp, dtype=bool), limits), (gap, bar_y))
    draw.text((gap, bar_y + 18), f"Near {limits[0]:.3g}", font=label_font, fill="#444444")
    draw.text((canvas.width - 75, bar_y + 18), f"Far {limits[1]:.3g}", font=label_font, fill="#444444")
    draw.text((gap, bar_y + 42), "Official RGB-only model and renderer; target RGB was not input. Gray checkerboard = invalid.", font=small_font, fill="#555555")
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


@torch.no_grad()
def main():
    api_path = Path(sys.modules["depth_anything_3.api"].__file__).resolve()
    renderer_path = Path(sys.modules["depth_anything_3.model.utils.gs_renderer"].__file__).resolve()
    assert api_path == (OFFICIAL / "src/depth_anything_3/api.py").resolve()
    assert renderer_path == (OFFICIAL / "src/depth_anything_3/model/utils/gs_renderer.py").resolve()
    model = DepthAnything3.from_pretrained(str(WEIGHTS)).to("cuda:0").eval()
    source_images, source_extrinsics, source_intrinsics = [], [], []
    for frame_id in SOURCE_IDS:
        with np.load(DATA / f"RGBT-Scenes_Building_img_{frame_id:03d}.npz") as data:
            rgb = (np.moveaxis(data["rgb"], 0, -1).clip(0, 1) * 255 + 0.5).astype(np.uint8)
            source_images.append(rgb)
            source_extrinsics.append(data["E"].copy())
            source_intrinsics.append(data["K"].copy())
    with np.load(DATA / f"RGBT-Scenes_Building_img_{TARGET_ID:03d}.npz") as data:
        target_rgb = (np.moveaxis(data["rgb"], 0, -1).clip(0, 1) * 255 + 0.5).astype(np.uint8)
        target_extrinsic = data["E"].copy()
        target_intrinsic = data["K"].copy()
        target_valid = data["valid"][0] > 0.5
    source_extrinsics = np.stack(source_extrinsics)
    source_intrinsics = np.stack(source_intrinsics)
    captured = {}

    def capture_pose(_module, _inputs, output):
        captured["predicted_extrinsics"] = output["extrinsics"][0].detach().float().cpu().numpy()

    hook = model.model.register_forward_hook(capture_pose)
    try:
        prediction = model.inference(
            source_images,
            extrinsics=source_extrinsics,
            intrinsics=source_intrinsics,
            process_res=336,
            process_res_method="upper_bound_resize",
            infer_gs=True,
        )
    finally:
        hook.remove()
    if prediction.gaussians is None:
        raise RuntimeError("Official DA3 did not return its Gaussian branch")
    h, w = prediction.depth.shape[-2:]
    if (h, w) != target_valid.shape:
        raise RuntimeError(f"Unexpected resolution: {(h, w)}")

    # The official API normalizes source poses relative to source 1 and divides
    # translations by a clamped median baseline before model forward.
    source_relative = source_extrinsics @ np.linalg.inv(source_extrinsics[0])
    centers = np.linalg.inv(source_relative)[:, :3, 3]
    normalization_scale = max(float(np.median(np.linalg.norm(centers, axis=-1))), 0.1)
    source_normalized = source_relative.copy()
    source_normalized[:, :3, 3] /= normalization_scale
    target_normalized = target_extrinsic @ np.linalg.inv(source_extrinsics[0])
    target_normalized[:3, 3] /= normalization_scale

    predicted_extrinsics = captured["predicted_extrinsics"]
    # The untouched official Gaussian adapter scales predicted camera centers
    # and Gaussian depths by this clamped alignment factor.
    _, _, adapter_scale = align_poses_umeyama(source_normalized, predicted_extrinsics)
    adapter_scale = float(np.clip(adapter_scale, 1 / 3, 3))
    predicted_cameras_scaled = np.linalg.inv(homogeneous(predicted_extrinsics))
    predicted_cameras_scaled[:, :3, 3] *= adapter_scale
    predicted_extrinsics_scaled = np.linalg.inv(predicted_cameras_scaled)
    rotation, translation, alignment_scale = align_poses_umeyama(
        source_normalized, predicted_extrinsics_scaled
    )
    if not np.isfinite(alignment_scale) or alignment_scale <= 0:
        raise RuntimeError(f"Invalid pose alignment scale: {alignment_scale}")
    source_in_gs = np.stack(
        [camera_in_gs_frame(e, rotation, translation, alignment_scale) for e in source_normalized]
    )
    target_in_gs = camera_in_gs_frame(target_normalized, rotation, translation, alignment_scale)
    source_reprojection_error = float(
        np.max(np.abs(source_in_gs[:, :3, 3] - predicted_extrinsics_scaled[:, :3, 3]))
    )
    gs = prediction.gaussians
    views = [target_in_gs, source_in_gs[0]]
    rendered = []
    for view_index, extrinsic in enumerate(views):
        K = target_intrinsic if view_index == 0 else source_intrinsics[0]
        K_normalized = K.copy()
        K_normalized[0] /= w
        K_normalized[1] /= h
        result = []
        for gaussian_group in [gs] + [
            replace(
                gs,
                **{
                    name: getattr(gs, name)[:, i * h * w : (i + 1) * h * w]
                    for name in ("means", "scales", "rotations", "harmonics", "opacities")
                },
            )
            for i in range(3)
        ]:
            color, depth = render_3dgs(
                extrinsics=torch.from_numpy(extrinsic)[None].to(gs.means),
                intrinsics=torch.from_numpy(K_normalized)[None].to(gs.means),
                image_shape=(h, w),
                gaussian=gaussian_group,
                num_view=1,
                color_mode="RGB+ED",
            )
            # Convert Gaussian-frame depth to the original COLMAP scene units.
            multiplier = float(alignment_scale * normalization_scale)
            result.append((color[0].detach().float().cpu().numpy(), depth[0].detach().float().cpu().numpy() * multiplier))
        rendered.append(result)
    target_depths = np.stack([item[1] for item in rendered[0]])
    source_rerender = rendered[1][0][0]
    source_rgb = np.moveaxis(source_rerender, 0, -1)
    source_rgb = (source_rgb.clip(0, 1) * 255 + 0.5).astype(np.uint8)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    Image.fromarray(source_rgb).save(OUTPUT / "source_026_rerender_check.png")

    with np.load(DEPTH_TEST / "RGBT-Scenes_Building_img_025" / "target_depth.npz") as old:
        reference = old["colmap_reference"].astype(np.float32)
        limits = tuple(float(x) for x in old["color_limits"])
    arrays = [reference, *target_depths]
    masks = [reference > 0] + [target_valid & np.isfinite(z) & (z > 0) for z in target_depths]
    make_panel(
        OUTPUT / "official_da3_source_only_target_depth_shared_scale.png",
        arrays,
        masks,
        limits,
        "Official DA3-GIANT-1.1 | target img_025 from three source RGB views",
        "Official Gaussian branch; source 026/024/027; target RGB excluded; Building scene scale",
    )
    relative_limits = tuple(float(x) for x in np.percentile(target_depths[0][masks[1]], [2, 98]))
    visualize(target_depths[0], masks[1], limits).save(
        OUTPUT / "official_da3_img_025_depth_scene_scale.png"
    )
    visualize(target_depths[0], masks[1], relative_limits).save(
        OUTPUT / "official_da3_img_025_depth_visible.png"
    )
    # The native DA3 depth head can also predict img_025 directly when its RGB
    # image is supplied as a fourth input. Keep this separate from the held-out
    # source-only Gaussian render above.
    direct = model.inference(
        [*source_images, target_rgb],
        extrinsics=np.concatenate([source_extrinsics, target_extrinsic[None]]),
        intrinsics=np.concatenate([source_intrinsics, target_intrinsic[None]]),
        process_res=336,
        process_res_method="upper_bound_resize",
        infer_gs=False,
    )
    direct_target_depth = np.asarray(direct.depth[3], dtype=np.float32)
    direct_mask = target_valid & np.isfinite(direct_target_depth) & (direct_target_depth > 0)
    direct_limits = tuple(float(x) for x in np.percentile(direct_target_depth[direct_mask], [2, 98]))
    visualize(direct_target_depth, direct_mask, direct_limits).save(
        OUTPUT / "official_da3_img_025_direct_depth_target_rgb_input.png"
    )
    visualize(direct_target_depth, direct_mask, limits).save(
        OUTPUT / "official_da3_img_025_direct_depth_scene_scale.png"
    )
    make_panel(
        OUTPUT / "official_da3_source_only_target_depth_relative.png",
        arrays,
        masks,
        relative_limits,
        "Official DA3-GIANT-1.1 | target img_025 from three source RGB views",
        "Official Gaussian branch; source 026/024/027; target RGB excluded; range from official render",
    )
    np.savez_compressed(
        OUTPUT / "official_da3_source_only_target_depth.npz",
        colmap_reference=reference,
        official_target_depth=target_depths[0],
        official_source_contributions=target_depths[1:],
        official_direct_target_depth=direct_target_depth,
        valid=target_valid,
        source_ids=np.array(SOURCE_IDS),
        source_extrinsics=source_extrinsics,
        target_extrinsic=target_extrinsic,
        target_intrinsic=target_intrinsic,
        normalization_scale=np.float32(normalization_scale),
        adapter_scale=np.float32(adapter_scale),
        alignment_scale=np.float32(alignment_scale),
        color_limits=np.array(limits),
    )
    metadata = {
        "official_api": str(api_path),
        "official_renderer": str(renderer_path),
        "weights": str(WEIGHTS / "model.safetensors"),
        "source_ids": list(SOURCE_IDS),
        "target_id": TARGET_ID,
        "source_only_target_rgb_was_input": False,
        "infer_gs": True,
        "pose_alignment": "Official Gaussian frame aligned to the normalized known source cameras by a similarity transform; target known camera mapped into Gaussian frame",
        "normalization_scale": normalization_scale,
        "adapter_scale": adapter_scale,
        "alignment_scale": float(alignment_scale),
        "source_reprojection_translation_error": source_reprojection_error,
        "relative_color_limits": relative_limits,
        "direct_target_rgb_was_input": True,
        "direct_depth_source": "Official DA3 native depth head on four RGB inputs: source 026/024/027 plus target 025",
        "direct_color_limits": direct_limits,
        "scene_color_limits": limits,
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
