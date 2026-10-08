"""Historical independent cross-view refinement network."""

from dataclasses import replace

import torch
import torch.nn.functional as F
from torch import nn

from ..attention import WindowNeighborhood
from ..calibration import calibrate_output
from ..renderer import leave_one_out, render_rgbt
from .refinement import NeighborhoodRefiner


class SpatialHead(nn.Module):
    def __init__(self, head, channels):
        super().__init__()
        self.proposal = head
        self.gate = nn.Conv2d(channels + 4, 1, 1)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, -0.5)
        self.last_gate = None

    def forward(self, x):
        gate = self.gate(x).float().sigmoid()
        self.last_gate = gate
        # Parent applies tanh; atanh preserves the explicitly bounded local gate.
        return torch.atanh(
            (self.proposal(x).float().tanh() * gate).clamp(-0.999, 0.999)
        )


class CrossViewRefiner(NeighborhoodRefiner):
    def __init__(
        self,
        channels=96,
        blocks=2,
        depth_limit=0.10,
        position_strength=0.2,
        rgb_limit=0.02,
        scale_limit=0.8,
        opacity_limit=2.0,
        color_limit=0.15,
        calibration_strength=0.0,
        window=True,
        feedback=True,
        trust_mode="none",
        fused_rendering=False,
        update_strength=1.0,
        thermal_strength=1.0,
        differentiate_trust=False,
        refiner_thermal=True,
        compact=False,
    ):
        super().__init__(
            channels=channels,
            blocks=blocks,
            iterations=1,
            render_feedback=feedback,
            depth_limit=depth_limit,
            position_strength=position_strength,
            rgb_limit=rgb_limit,
            scale_limit=scale_limit,
            opacity_limit=opacity_limit,
            color_limit=color_limit,
            support_context=True,
            source_guard=False,
            compact=compact,
        )
        self.aligned_points = True
        self.fused_rendering = fused_rendering
        self.update_strength = update_strength
        self.thermal_strength = thermal_strength
        self.differentiate_trust = differentiate_trust
        self.refiner_thermal = refiner_thermal
        if window:
            self.blocks = nn.ModuleList(
                [WindowNeighborhood(channels) for _ in range(blocks)]
            )
        self.head = SpatialHead(self.head, channels)
        self.calibration_strength = calibration_strength
        self.calibration_info = None
        self.trust_mode = trust_mode
        self.config = {
            "channels": channels,
            "blocks": blocks,
            "depth_limit": depth_limit,
            "position_strength": position_strength,
            "rgb_limit": rgb_limit,
            "scale_limit": scale_limit,
            "opacity_limit": opacity_limit,
            "color_limit": color_limit,
            "calibration_strength": calibration_strength,
            "window": window,
            "feedback": feedback,
            "trust_mode": trust_mode,
            "fused_rendering": fused_rendering,
            "update_strength": update_strength,
            "thermal_strength": thermal_strength,
            "differentiate_trust": differentiate_trust,
            "refiner_thermal": refiner_thermal,
            "compact": compact,
        }

    def calibrate_initial(self, initial, rgb, known_cameras=True, source_geometry=None):
        if self.calibration_strength and known_cameras:
            calibrated, self.calibration_info = calibrate_output(
                initial,
                rgb,
                observations=None
                if source_geometry is None
                else source_geometry.observations(initial),
            )
            strength = self.calibration_strength
            initial = replace(
                initial,
                gaussians=replace(
                    initial.gaussians,
                    means=initial.gaussians.means.lerp(
                        calibrated.gaussians.means, strength
                    ),
                    scales=initial.gaussians.scales.lerp(
                        calibrated.gaussians.scales, strength
                    ),
                ),
                depth=initial.depth.lerp(calibrated.depth, strength),
            )
        return initial

    def source_feedback(self, o, E, K, h, w):
        return leave_one_out(
            replace(o, fused_rendering=self.fused_rendering), E, K, h, w
        )

    def forward(
        self,
        rgb,
        thermal,
        extrinsics=None,
        intrinsics=None,
        initial=None,
        return_initial=False,
        source_valid=None,
        reference_initial=None,
        observed_thermal=None,
        source_geometry=None,
        share_projection=False,
    ):
        if initial is None:
            raise ValueError(
                "Cross-view refinement requires a neighborhood reconstruction"
            )
        reference = initial if reference_initial is None else reference_initial
        initial = self.calibrate_initial(
            initial, rgb, extrinsics is not None, source_geometry
        )
        observed_thermal = thermal if observed_thermal is None else observed_thermal
        conditioning_thermal = (
            thermal if self.refiner_thermal else torch.full_like(thermal, 0.5)
        )
        output = super().forward(
            rgb,
            conditioning_thermal,
            extrinsics,
            intrinsics,
            initial=initial,
            source_valid=source_valid,
            share_projection=share_projection,
        )
        output = replace(output, fused_rendering=self.fused_rendering)
        if self.update_strength != 1 or self.thermal_strength != 1:
            strength = self.update_strength
            gs = replace(
                reference.gaussians,
                **{
                    name: getattr(reference.gaussians, name).lerp(
                        getattr(output.gaussians, name), strength
                    )
                    for name in [
                        "means",
                        "scales",
                        "rotations",
                        "harmonics",
                        "opacities",
                    ]
                },
            )
            gs = replace(gs, rotations=F.normalize(gs.rotations, dim=-1))
            output = replace(
                output,
                gaussians=gs,
                thermal_colors=reference.thermal_colors.lerp(
                    output.thermal_colors, strength * self.thermal_strength
                ),
            )
        if self.trust_mode != "none":
            from ..local_trust import spatial_trust

            output = spatial_trust(
                reference,
                output,
                rgb,
                thermal if observed_thermal is None else observed_thermal,
                source_valid,
                mode=self.trust_mode,
                detach=not (self.training and self.differentiate_trust),
            )
        return (output, initial) if return_initial else output
