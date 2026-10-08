"""Cache source triangulation while RGB and camera inputs remain unchanged."""

import torch

from .calibration import triangulate_sources


class SourceGeometry:
    """Lazy source observations, owned by one forward and never shared across calls."""

    def __init__(self, rgb):
        self.rgb = rgb
        self._inputs = None
        self._observations = None

    @torch.no_grad()
    def observations(self, output):
        inputs = (self.rgb, output.extrinsics, output.intrinsics)
        # Keep the actual tensors alive, and notice in-place camera changes.
        versions = tuple(
            None if value.is_inference() else value._version for value in inputs
        )
        if (
            self._inputs is None
            or any(value is not old for value, old in zip(inputs, self._inputs[0]))
            or versions != self._inputs[1]
            or any(
                snapshot is not None and not torch.equal(value, snapshot)
                for value, snapshot in zip(inputs, self._inputs[2])
            )
        ):
            self._observations = [
                triangulate_sources(
                    self.rgb[b], output.extrinsics[b], output.intrinsics[b]
                )
                for b in range(len(self.rgb))
            ]
            # Inference tensors do not expose version counters. Small camera
            # snapshots (and RGB only when itself an inference tensor) preserve
            # invalidation semantics for callers using torch.inference_mode().
            snapshots = tuple(
                value.clone() if version is None else None
                for value, version in zip(inputs, versions)
            )
            self._inputs = (inputs, versions, snapshots)
        return self._observations
