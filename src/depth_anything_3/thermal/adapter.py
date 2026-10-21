"""Adapters use ordinary Linear calls so Giant's SwiGLU cannot bypass LoRA."""

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path

import torch
from torch import nn
from safetensors.torch import load_file

from depth_anything_3.cfg import create_object
from depth_anything_3.model.dinov2.layers.swiglu_ffn import SwiGLUFFN


@dataclass(frozen=True)
class AdapterConfig:
    rank: int = 64
    alpha: float = 128.0
    dropout: float = 0.1
    start_block: int = 13
    gradient_checkpointing: bool = True
    lora_scope: str = "shared"

    def __post_init__(self):
        if self.lora_scope not in ("shared", "thermal_only"):
            raise ValueError("lora_scope must be shared or thermal_only")


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, config: AdapterConfig):
        super().__init__()
        if config.rank <= 0 or not 0 <= config.dropout < 1:
            raise ValueError("LoRA needs positive rank and dropout in [0, 1)")
        self.base = base.requires_grad_(False)
        self.lora_scope = config.lora_scope
        self.scale = config.alpha / config.rank
        self.dropout = nn.Dropout(config.dropout)
        self.lora_A = nn.Parameter(torch.empty(
            config.rank, base.in_features, device=base.weight.device, dtype=base.weight.dtype
        ))
        self.lora_B = nn.Parameter(torch.zeros(
            base.out_features, config.rank, device=base.weight.device, dtype=base.weight.dtype
        ))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x, lora_mask=None):
        if self.lora_scope == "thermal_only":
            if lora_mask is None:
                raise ValueError("Thermal-only LoRA requires an explicit per-token mask")
            if lora_mask.dtype != torch.bool or lora_mask.shape != x.shape[:-1]:
                raise ValueError("lora_mask must be boolean and match the input token dimensions")
        update = (self.dropout(x) @ self.lora_A.T) @ self.lora_B.T
        if self.lora_scope == "thermal_only":
            # Keep the zero-gradient graph for RGB-only batches (including DDP).
            update = torch.where(lora_mask[..., None], update, 0)
        return self.base(x) + update * self.scale


def install_adapter(model, config=AdapterConfig()):
    """Call AFTER loading pretrained weights; freezes all except adapter parameters."""
    backbone = model.backbone.pretrained
    if hasattr(backbone, "thermal_camera_token"):
        raise ValueError("A thermal adapter is already installed")
    if not hasattr(backbone, "camera_token") or not 0 <= config.start_block < len(backbone.blocks):
        raise ValueError("Adapter needs a camera-token backbone and a valid start block")
    model.requires_grad_(False)
    backbone.thermal_camera_token = nn.Parameter(backbone.camera_token.detach().clone())
    backbone.gradient_checkpointing = config.gradient_checkpointing
    for block in backbone.blocks[config.start_block:]:
        block.thermal_only_lora = config.lora_scope == "thermal_only"
        old = block.mlp
        if not hasattr(old, "w12") or old.w12 is None:
            raise ValueError("This adapter targets Giant's packed SwiGLU FFN")
        # Keep the pretrained Linear objects and their exact weights, but use an
        # explicit forward instead of xFormers' direct .weight access.
        explicit = SwiGLUFFN.__new__(SwiGLUFFN)
        nn.Module.__init__(explicit)
        explicit.w12, explicit.w3 = old.w12, old.w3
        block.mlp = explicit
        for parent, name in ((block.attn, "qkv"), (block.attn, "proj"),
                             (block.mlp, "w12"), (block.mlp, "w3")):
            setattr(parent, name, LoRALinear(getattr(parent, name), config))
    model.thermal_adapter_config = asdict(config)
    return model


def adapter_state_dict(model):
    return {n: p.detach().cpu().clone() for n, p in model.named_parameters()
            if n.endswith(("lora_A", "lora_B", "thermal_camera_token"))}


def load_adapter(model, checkpoint):
    """Load an adapter export or training checkpoint onto a fresh pretrained model."""
    if isinstance(checkpoint, (str, Path)):
        checkpoint = torch.load(checkpoint, map_location="cpu", weights_only=True)
    config = AdapterConfig(**checkpoint["adapter_config"])
    if not hasattr(model, "thermal_adapter_config"):
        install_adapter(model, config)
    elif AdapterConfig(**model.thermal_adapter_config) != config:
        raise ValueError("Installed adapter configuration does not match checkpoint")
    expected = adapter_state_dict(model)
    state = checkpoint["adapter"]
    if expected.keys() != state.keys():
        raise ValueError("Adapter checkpoint has missing or unexpected parameters")
    params = dict(model.named_parameters())
    with torch.no_grad():
        for name, value in state.items():
            if value.shape != params[name].shape:
                raise ValueError(f"Adapter shape mismatch: {name}")
            params[name].copy_(value)
    return model


def load_giant(checkpoint_dir, *, include_gaussians=False):
    """Load local Giant weights, optionally retaining its pretrained GSDPT branch.

    Depth training omits the Gaussian branch by default to save memory. Novel
    view synthesis must load it here and also pass ``infer_gs=True`` at forward.
    """
    checkpoint_dir = Path(checkpoint_dir)
    metadata = json.loads((checkpoint_dir / "config.json").read_text())
    if metadata["model_name"] != "da3-giant":
        raise ValueError("Expected standalone da3-giant pretrained weights")
    config = metadata["config"]
    if include_gaussians:
        if not config.get("gs_head") or not config.get("gs_adapter"):
            raise ValueError("This checkpoint has no configured Gaussian branch")
    else:
        config.pop("gs_head", None)
        config.pop("gs_adapter", None)
    model = create_object(config)
    state = load_file(str(checkpoint_dir / "model.safetensors"), device="cpu")
    state = {k.removeprefix("model."): v for k, v in state.items()
             if include_gaussians or not k.startswith(("model.gs_head.", "model.gs_adapter."))}
    # HF safetensors omits aliases of shared LayerNorm parameters in DualDPT.
    # Fill only proven aliases, then strictly check every remaining weight.
    aliases = {}
    for name, parameter in model.named_parameters(remove_duplicate=False):
        aliases.setdefault(id(parameter), []).append(name)
    for names in aliases.values():
        source = next((n for n in names if n in state), None)
        if source is not None:
            for name in names:
                state.setdefault(name, state[source])
    model.load_state_dict(state, strict=True)
    model.base_checkpoint = str(checkpoint_dir.resolve())
    return model
