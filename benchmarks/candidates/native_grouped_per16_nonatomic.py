#!/usr/bin/env python3
"""P134 candidate backend: packed-NVFP4 grouped-per16 non-atomic MoE path.

This adapts the existing native CUDA `active_moe_grouped_per16_nonatomic_reference`
ABI to the p134 fixture contract. It is intentionally a fixture-level admission
backend: candidates must pass here before they deserve full P37/P25 service time.
"""
from __future__ import annotations

import os
from pathlib import Path
import sys
from typing import Any

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

_PACKED_CACHE: dict[tuple[str, int, str], dict[str, torch.Tensor]] = {}


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    value = int(raw)
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def _load_native_extension():
    from engine.native_cuda import load_lynn_native_extension

    ext = load_lynn_native_extension(verbose=os.environ.get("LYNN_NATIVE_CUDA_VERBOSE", "0") == "1")
    if not hasattr(ext, "active_moe_grouped_per16_nonatomic_reference"):
        raise RuntimeError("native extension missing active_moe_grouped_per16_nonatomic_reference")
    return ext


def _packed_aliases(
    layer_weights: dict[str, Any],
    cfg: dict[str, Any],
    *,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Return packed NVFP4 active-expert aliases for this fixture layer."""
    required = (
        "mlp.experts._gate_up_packed",
        "mlp.experts._gate_up_scale",
        "mlp.experts._gate_up_global_scale",
        "mlp.experts._down_packed",
        "mlp.experts._down_scale",
        "mlp.experts._down_global_scale",
    )
    if all(key in layer_weights for key in required):
        return {key: layer_weights[key] for key in required}

    if "model_dir" not in cfg or "layer_id" not in cfg:
        raise KeyError(
            "packed aliases are absent from layer_weights and cfg lacks model_dir/layer_id"
        )
    model_dir = str(cfg["model_dir"])
    layer_id = int(cfg["layer_id"])
    cache_key = (model_dir, layer_id, str(device))
    cached = _PACKED_CACHE.get(cache_key)
    if cached is not None:
        return cached

    from engine.nvfp4_runtime import load_grouped_nvfp4_weight

    prefix = f"model.language_model.layers.{layer_id}.mlp.experts"
    gate_packed, gate_scale, gate_global = load_grouped_nvfp4_weight(
        model_dir,
        f"{prefix}.gate_up_proj",
        device=str(device),
    )
    down_packed, down_scale, down_global = load_grouped_nvfp4_weight(
        model_dir,
        f"{prefix}.down_proj",
        device=str(device),
    )
    aliases = {
        "mlp.experts._gate_up_packed": gate_packed.contiguous(),
        "mlp.experts._gate_up_scale": gate_scale.contiguous(),
        "mlp.experts._gate_up_global_scale": gate_global.contiguous(),
        "mlp.experts._down_packed": down_packed.contiguous(),
        "mlp.experts._down_scale": down_scale.contiguous(),
        "mlp.experts._down_global_scale": down_global.contiguous(),
    }
    _PACKED_CACHE[cache_key] = aliases
    return aliases


def moe_forward_routed_only(
    hidden_in: torch.Tensor,
    expert_ids: torch.Tensor,
    routing_weights: torch.Tensor,
    layer_weights: dict[str, Any],
    cfg: dict[str, Any],
) -> torch.Tensor:
    """Run the existing packed-NVFP4 native-owned scratch path."""
    ext = _load_native_extension()
    x = hidden_in.view(-1).contiguous().to(torch.bfloat16)
    aliases = _packed_aliases(layer_weights, cfg, device=x.device)
    return ext.active_moe_grouped_per16_nonatomic_reference(
        x,
        expert_ids.to(torch.int32).contiguous(),
        routing_weights.to(torch.float32).contiguous(),
        aliases["mlp.experts._gate_up_packed"],
        aliases["mlp.experts._gate_up_scale"],
        aliases["mlp.experts._gate_up_global_scale"],
        aliases["mlp.experts._down_packed"],
        aliases["mlp.experts._down_scale"],
        aliases["mlp.experts._down_global_scale"],
        _env_int("LYNN_NATIVE_GATEUP_TILE_INTER", 2),
        _env_int("LYNN_NATIVE_DOWN_TILE_HIDDEN", 2),
    ).view(1, -1)


def moe_forward_fixture(
    hidden_in: torch.Tensor,
    expert_ids: torch.Tensor,
    routing_weights: torch.Tensor,
    layer_weights: dict[str, Any],
    cfg: dict[str, Any],
) -> torch.Tensor:
    """Full MoE output for p134 full-output probes."""
    moe_out = moe_forward_routed_only(hidden_in, expert_ids, routing_weights, layer_weights, cfg)
    h_flat = hidden_in.to(torch.bfloat16)
    if "mlp.shared_expert.gate_proj.weight" in layer_weights:
        gate_s = F.linear(h_flat, layer_weights["mlp.shared_expert.gate_proj.weight"])
        up_s = F.linear(h_flat, layer_weights["mlp.shared_expert.up_proj.weight"])
        shared_ffn = F.linear(F.silu(gate_s) * up_s, layer_weights["mlp.shared_expert.down_proj.weight"])
        if "mlp.shared_expert_gate.weight" in layer_weights:
            shared_ffn = shared_ffn * torch.sigmoid(
                F.linear(h_flat, layer_weights["mlp.shared_expert_gate.weight"])
            )
        moe_out = moe_out + shared_ffn
    return moe_out
