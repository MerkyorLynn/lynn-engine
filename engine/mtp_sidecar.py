"""Runtime helpers for Lynn-shaped Qwen3-Next MTP sidecars.

The A100 training scripts use the same tensor contract as the resident runner:
`mtp.fc -> one full-attention MTP layer -> mtp.norm -> shared lm_head`.  Keeping
that contract in `engine` lets serving and benchmarks shadow-check MTP draft
acceptance without importing research scripts from production code.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from safetensors import safe_open

from engine.full_forward import _layer_forward, _rms_norm


def load_mtp_sidecar(
    path: str | Path,
    *,
    device: str,
    dtype: torch.dtype,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Load an MTP sidecar and return tensors plus a compact inventory."""
    tensors: dict[str, torch.Tensor] = {}
    inventory: dict[str, Any] = {}
    sidecar_path = Path(path)
    with safe_open(sidecar_path, framework="pt", device="cpu") as f:
        metadata = f.metadata() or {}
        for key in f.keys():
            tensor = f.get_tensor(key)
            inventory[key] = {
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype).replace("torch.", ""),
                "numel": int(tensor.numel()),
            }
            tensors[key] = tensor.to(device=device, dtype=dtype)
    return tensors, {"path": str(sidecar_path), "metadata": metadata, "tensors": inventory}


def mtp_layer_weights(sidecar: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Strip the `mtp.layers.0.` prefix into a regular layer-weight mapping."""
    prefix = "mtp.layers.0."
    out: dict[str, torch.Tensor] = {}
    for key, tensor in sidecar.items():
        if key.startswith(prefix):
            out[key.removeprefix(prefix)] = tensor

    required = {
        "input_layernorm.weight",
        "post_attention_layernorm.weight",
        "self_attn.q_norm.weight",
        "self_attn.k_norm.weight",
        "self_attn.q_proj.weight",
        "self_attn.k_proj.weight",
        "self_attn.v_proj.weight",
        "self_attn.o_proj.weight",
        "mlp.gate.weight",
        "mlp.experts.gate_up_proj",
        "mlp.experts.down_proj",
        "mlp.shared_expert.gate_proj.weight",
        "mlp.shared_expert.up_proj.weight",
        "mlp.shared_expert.down_proj.weight",
        "mlp.shared_expert_gate.weight",
    }
    missing = sorted(required - set(out))
    if missing:
        raise KeyError(f"MTP sidecar is missing layer tensors after prefix strip: {missing}")
    return out


def mtp_layer_config(base_cfg: dict[str, Any], mtp_w: dict[str, torch.Tensor]) -> dict[str, Any]:
    """Build the one-layer MTP config from the resident model config."""
    text_cfg = base_cfg.get("text_config") or base_cfg
    cfg = dict(text_cfg)
    cfg["num_experts"] = int(mtp_w["mlp.experts.gate_up_proj"].shape[0])
    cfg["num_experts_per_tok"] = int(text_cfg.get("num_experts_per_tok", 8))
    cfg["expert_intermediate"] = int(mtp_w["mlp.experts.down_proj"].shape[-1])
    cfg["layer_idx"] = 0
    return cfg


def mtp_logits(
    *,
    sidecar: dict[str, torch.Tensor],
    mtp_w: dict[str, torch.Tensor],
    mtp_cfg: dict[str, Any],
    embed_weight: torch.Tensor,
    lm_head_fn: Any,
    base_hidden: torch.Tensor,
    current_token_id: int,
    current_pos: int,
    device: str,
) -> torch.Tensor:
    """Return MTP draft logits for the token after `current_token_id`."""
    input_embed = embed_weight[int(current_token_id)].view(1, 1, -1)
    hidden_part = _rms_norm(base_hidden, sidecar["mtp.pre_fc_norm_hidden.weight"])
    embed_part = _rms_norm(input_embed, sidecar["mtp.pre_fc_norm_embedding.weight"])
    mtp_hidden = F.linear(torch.cat([hidden_part, embed_part], dim=-1), sidecar["mtp.fc.weight"])
    pos = torch.tensor([[int(current_pos)]], device=device, dtype=torch.long)
    mtp_out = _layer_forward(mtp_hidden, pos, "full_attention", mtp_w, mtp_cfg)
    mtp_normed = _rms_norm(mtp_out, sidecar["mtp.norm.weight"])
    return lm_head_fn(mtp_normed)
