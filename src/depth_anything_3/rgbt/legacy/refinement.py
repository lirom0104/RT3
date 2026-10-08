"""Original neighborhood-stage interface for historical checkpoints."""

from ..gaussian_update import GaussianUpdate


class NeighborhoodRefiner(GaussianUpdate):
    """Retain the original stage name while sharing the Gaussian update implementation."""
