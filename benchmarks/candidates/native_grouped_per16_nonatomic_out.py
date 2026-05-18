#!/usr/bin/env python3
"""P136 candidate backend: graph-safe caller-owned packed-NVFP4 active MoE path."""
from __future__ import annotations

import os
from pathlib import Path
import sys
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from benchmarks.candidates.native_grouped_per16_nonatomic import _env_int, _packed_aliases


def _load_native_extension():
    from engine.native_cuda import load_lynn_native_extension

    ext = load_lynn_native_extension(verbose=os.environ.get("LYNN_NATIVE_CUDA_VERBOSE", "0") == "1")
    if not hasattr(ext, "active_moe_grouped_per16_nonatomic_out_reference"):
        raise RuntimeError("native extension missing active_moe_grouped_per16_nonatomic_out_reference")
    return ext


def moe_forward_routed_only(
    hidden_in: torch.Tensor,
    expert_ids: torch.Tensor,
    routing_weights: torch.Tensor,
    layer_weights: dict[str, Any],
    cfg: dict[str, Any],
) -> torch.Tensor:
    """Run packed-NVFP4 active MoE using caller-owned inter/out scratch tensors."""
    ext = _load_native_extension()
    x = hidden_in.view(-1).contiguous().to(torch.bfloat16)
    expert_ids_i32 = expert_ids.to(torch.int32).contiguous()
    routing_weights_f32 = routing_weights.to(torch.float32).contiguous()
    aliases = _packed_aliases(layer_weights, cfg, device=x.device)
    inter = torch.empty((expert_ids_i32.numel(), 512), device=x.device, dtype=torch.bfloat16)
    out = torch.empty((2048,), device=x.device, dtype=torch.bfloat16)
    return ext.active_moe_grouped_per16_nonatomic_out_reference(
        x,
        expert_ids_i32,
        routing_weights_f32,
        aliases["mlp.experts._gate_up_packed"],
        aliases["mlp.experts._gate_up_scale"],
        aliases["mlp.experts._gate_up_global_scale"],
        aliases["mlp.experts._down_packed"],
        aliases["mlp.experts._down_scale"],
        aliases["mlp.experts._down_global_scale"],
        inter,
        out,
        _env_int("LYNN_NATIVE_GATEUP_TILE_INTER", 2),
        _env_int("LYNN_NATIVE_DOWN_TILE_HIDDEN", 2),
    ).view(1, -1)
