"""Gaussian PLY and paired RGB/thermal camera-path previews."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from depth_anything_3.rgbt.geometry import gaussian_depth
from depth_anything_3.rgbt.renderer import render_rgbt
from depth_anything_3.utils.camera_trj_helpers import interpolate_extrinsics
from depth_anything_3.utils.geometry import affine_inverse
from depth_anything_3.utils.gsply_helpers import save_gaussian_ply


@torch.no_grad()
def export_scene(output, directory, image_size, frames=24):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    gs = output.gaussians
    save_gaussian_ply(
        gs,
        str(directory / "rgb.ply"),
        output.depth[0, ..., None],
        save_sh_dc_only=False,
        prune_border_gs=False,
    )
    # Export the scalar thermal signal as three identical SH DC channels.
    sh = torch.zeros_like(gs.harmonics)
    sh[..., 0] = (output.thermal_colors.expand(-1, -1, 3) - 0.5) / 0.28209479177387814
    if output.thermal_harmonics is not None:
        sh = output.thermal_harmonics[:, :, None, :].expand(-1, -1, 3, -1).contiguous()
    save_gaussian_ply(
        replace(gs, harmonics=sh),
        str(directory / "thermal.ply"),
        output.depth[0, ..., None],
        save_sh_dc_only=False,
        prune_border_gs=False,
    )
    np.savez_compressed(
        directory / "cameras.npz",
        extrinsics=output.extrinsics.cpu().numpy(),
        intrinsics=output.intrinsics.cpu().numpy(),
    )
    final_z = gaussian_depth(output)
    ratio = final_z / output.depth.clamp_min(1e-6)
    poses = affine_inverse(output.extrinsics)[0]
    direction = poses[:, :3, 2]
    angles = torch.rad2deg(torch.acos((direction @ direction.T).clamp(-1, 1)))
    (directory / "geometry_audit.json").write_text(
        json.dumps(
            {
                "input_views": int(output.depth.shape[1]),
                "gaussians_before_border_trim": int(gs.means.shape[1]),
                "gaussians_exported": int(gs.means.shape[1]),
                "prune_border_gs": False,
                "final_z_to_base_depth_min_median_max": [
                    float(ratio.min()),
                    float(ratio.median()),
                    float(ratio.max()),
                ],
                "max_input_optical_axis_angle_degrees": float(angles.max()),
                "note": "Depth ratio is a construction invariant, not proof of absolute geometric accuracy. Optical-axis angle is not full scene coverage.",
            },
            indent=2,
        )
    )
    export_point_preview(output, directory / "geometry.html")
    h, w = image_size
    if len(poses) < 2:
        return
    times = torch.linspace(0, 1, frames, device=poses.device)
    c2w = interpolate_extrinsics(poses[0], poses[-1], times)
    E = affine_inverse(c2w)[None]
    K = (
        (1 - times[:, None, None]) * output.intrinsics[0, 0]
        + times[:, None, None] * output.intrinsics[0, -1]
    )[None]
    images = []
    for i in range(frames):
        p = render_rgbt(output, E[:, i : i + 1], K[:, i : i + 1], h, w)
        rgb = p["rgb"][0, 0].permute(1, 2, 0).clamp(0, 1).cpu().numpy()
        t = (
            p["thermal"][0, 0]
            .permute(1, 2, 0)
            .expand(-1, -1, 3)
            .clamp(0, 1)
            .cpu()
            .numpy()
        )
        im = Image.fromarray((np.concatenate([rgb, t], axis=1) * 255).astype(np.uint8))
        canvas = Image.new("RGB", (w * 2, h + 24), (18, 21, 30))
        canvas.paste(im, (0, 24))
        draw = ImageDraw.Draw(canvas)
        draw.text((5, 5), "RGB / Thermal - interpolated camera path", fill="white")
        images.append(canvas)
    images[0].save(
        directory / "rgbt_trajectory.gif",
        save_all=True,
        append_images=images[1:] + images[::-1],
        duration=90,
        loop=0,
    )


def export_point_preview(output, path):
    """Offline interactive preview of Gaussian centers (not Gaussian rasterization)."""
    import plotly.graph_objects as go

    gs = output.gaussians
    valid = torch.nonzero(gs.opacities[0].reshape(-1) > 0.1).flatten()
    if len(valid) > 16000:
        valid = valid[
            torch.linspace(0, len(valid) - 1, 16000, device=valid.device).long()
        ]
    xyz = gs.means[0, valid].detach().cpu().numpy()
    rgb = (gs.harmonics[0, valid, :, 0] * 0.28209479177387814 + 0.5).clamp(0, 1)
    t = output.thermal_colors[0, valid].expand(-1, 3).clamp(0, 1)
    fig = go.Figure()
    for name, color in [("RGB", rgb), ("Thermal", t)]:
        color = (color.detach().cpu().numpy() * 255).astype(np.uint8)
        colors = [f"rgb({r},{g},{b})" for r, g, b in color]
        fig.add_trace(
            go.Scatter3d(
                x=xyz[:, 0],
                y=xyz[:, 1],
                z=xyz[:, 2],
                mode="markers",
                marker={"size": 1.5, "color": colors},
                name=name,
                visible=name == "RGB",
                hoverinfo="skip",
            )
        )
    poses = affine_inverse(output.extrinsics)[0].detach().cpu().numpy()
    centers = poses[:, :3, 3]
    fig.add_trace(
        go.Scatter3d(
            x=centers[:, 0],
            y=centers[:, 1],
            z=centers[:, 2],
            mode="markers+text",
            text=[f"Camera {i + 1}" for i in range(len(centers))],
            marker={"size": 5, "color": "#ffd166"},
            name="Source cameras",
        )
    )
    arrow_length = float(output.depth[0].median().detach().cpu()) * 0.08
    ends = centers + poses[:, :3, 2] * arrow_length
    lines = [[], [], []]
    for c, end in zip(centers, ends):
        for axis in range(3):
            lines[axis].extend([float(c[axis]), float(end[axis]), None])
    fig.add_trace(
        go.Scatter3d(
            x=lines[0],
            y=lines[1],
            z=lines[2],
            mode="lines",
            line={"color": "#ffd166", "width": 4},
            name="Viewing directions",
        )
    )
    fig.update_layout(
        template="plotly_dark",
        title="Shared Gaussian centers — point preview (drag to rotate)",
        scene={"aspectmode": "data"},
        margin={"l": 0, "r": 0, "b": 0, "t": 65},
        updatemenus=[
            {
                "type": "buttons",
                "direction": "right",
                "x": 0,
                "y": 1.08,
                "buttons": [
                    {
                        "label": "RGB",
                        "method": "update",
                        "args": [{"visible": [True, False, True, True]}],
                    },
                    {
                        "label": "Thermal",
                        "method": "update",
                        "args": [{"visible": [False, True, True, True]}],
                    },
                ],
            }
        ],
    )
    fig.write_html(str(path), include_plotlyjs=True, full_html=True)
