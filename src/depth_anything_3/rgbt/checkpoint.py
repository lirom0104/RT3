"""Strict loading and atomic saving of complete, portable RGB-T checkpoints."""

from pathlib import Path

import torch

from .model import RGBTGaussianNetwork

FORMAT = "da3-rgbt-unified"


def load_rgbt_model(path, device="cuda:0"):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    if checkpoint.get("format") != FORMAT:
        raise ValueError(
            "Expected a unified RGB-T checkpoint containing the full model"
        )
    model_config = checkpoint["model_config"]
    if model_config.get("learned_joint") is None:
        from .legacy.model import LegacyRGBTGaussianNetwork

        model = LegacyRGBTGaussianNetwork(**model_config)
    else:
        model = RGBTGaussianNetwork(**model_config)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model.to(device).eval(), checkpoint


def save_checkpoint(path, model, optimizer=None, **metadata):
    """Save all stages and their config, without external weight dependencies."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    state = {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}
    try:
        torch.save(
            dict(
                **metadata,
                format=FORMAT,
                model_config=model.config,
                state_dict=state,
                optimizer=optimizer.state_dict() if optimizer else None,
            ),
            tmp,
        )
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)
