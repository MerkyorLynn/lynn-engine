"""Packed NVFP4 MoE decode path."""
from __future__ import annotations

import os

import torch
import torch.nn.functional as F

from engine.nvfp4_runtime import dual_scalar_bridge
from triton_kernels.nvfp4_moe import (
    nvfp4_grouped_down_weighted_sum,
    nvfp4_grouped_gate_up_silu,
)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    value = int(raw)
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return raw.lower() not in {"0", "false", "no", "off"}


def moe_forward_decode_packed_nvfp4(h: torch.Tensor, w: dict, cfg: dict) -> torch.Tensor:
    """Decode-only MoE using packed NVFP4 expert weights.

    Router and shared expert stay on the existing BF16 resident path for now;
    active routed experts consume packed NVFP4 weights directly. This is the
    first production-shaped bridge from P10-H into the resident runner.
    """
    if h.shape[1] != 1:
        raise NotImplementedError("packed NVFP4 MoE path is decode-only")
    required = (
        "mlp.experts._gate_up_packed",
        "mlp.experts._gate_up_scale",
        "mlp.experts._gate_up_global_scale",
        "mlp.experts._down_packed",
        "mlp.experts._down_scale",
        "mlp.experts._down_global_scale",
    )
    missing = [key for key in required if key not in w]
    if missing:
        raise KeyError(f"packed NVFP4 MoE aliases missing: {missing}")

    h_flat = h.reshape(-1, h.shape[-1])
    if h_flat.shape[0] != 1:
        raise NotImplementedError("packed NVFP4 MoE path currently supports batch=1")

    router_logits = F.linear(h_flat, w["mlp.gate.weight"])
    routing_weights, expert_indices = torch.topk(
        router_logits,
        int(cfg["num_experts_per_tok"]),
        dim=-1,
        sorted=_env_bool("LYNN_ROUTER_TOPK_SORTED", False),
    )
    routing_weights = F.softmax(routing_weights, dim=-1, dtype=torch.float32)[0]
    expert_ids = expert_indices[0].to(torch.long)
    topk_limit = os.environ.get("LYNN_MOE_PROFILE_TOPK_LIMIT")
    if topk_limit:
        limit = int(topk_limit)
        if not (1 <= limit <= expert_ids.numel()):
            raise ValueError(f"LYNN_MOE_PROFILE_TOPK_LIMIT must be in [1, {expert_ids.numel()}], got {limit}")
        expert_ids = expert_ids[:limit].contiguous()
        routing_weights = routing_weights[:limit].contiguous()
        routing_weights = routing_weights / routing_weights.sum().clamp_min(1e-20)
    hidden = h_flat[0]

    if os.environ.get("LYNN_MOE_PROFILE_SKIP_ACTIVE", "0") == "1":
        moe_out = torch.zeros_like(h_flat)
    else:
        inter = nvfp4_grouped_gate_up_silu(
            hidden,
            expert_ids,
            w["mlp.experts._gate_up_packed"],
            w["mlp.experts._gate_up_scale"],
            w["mlp.experts._gate_up_global_scale"],
            block_inter=_env_int("LYNN_MOE_GATE_BLOCK_INTER", 8),
            block_hidden=_env_int("LYNN_MOE_GATE_BLOCK_HIDDEN", 256),
            num_warps=_env_int("LYNN_MOE_GATE_NUM_WARPS", 4),
        )
        moe_out = nvfp4_grouped_down_weighted_sum(
            inter,
            expert_ids,
            routing_weights,
            w["mlp.experts._down_packed"],
            w["mlp.experts._down_scale"],
            w["mlp.experts._down_global_scale"],
            block_hidden=_env_int("LYNN_MOE_DOWN_BLOCK_HIDDEN", 8),
            block_inter=_env_int("LYNN_MOE_DOWN_BLOCK_INTER", 512),
            num_warps=_env_int("LYNN_MOE_DOWN_NUM_WARPS", 8),
        ).reshape_as(h_flat)

    if os.environ.get("LYNN_MOE_PROFILE_SKIP_SHARED", "0") == "1":
        return moe_out.to(h.dtype).reshape_as(h)

    if "mlp.shared_expert.gate_proj.weight" in w:
        if (
            "mlp.shared_expert.gate_proj.weight.packed" in w
            and "mlp.shared_expert.up_proj.weight.packed" in w
            and "mlp.shared_expert.down_proj.weight.packed" in w
        ):
            gate_s, up_s = dual_scalar_bridge(
                h_flat[0],
                w["mlp.shared_expert.gate_proj.weight.packed"],
                w["mlp.shared_expert.up_proj.weight.packed"],
            )
            shared = w["mlp.shared_expert.down_proj.weight.packed"](
                (F.silu(gate_s) * up_s).to(h.dtype)
            ).reshape_as(h_flat)
        elif (
            _env_bool("LYNN_SHARED_EXPERT_GATE_UP_FUSED", True)
            and "mlp.shared_expert._gate_up_proj.weight" in w
        ):
            gate_up_s = F.linear(h_flat, w["mlp.shared_expert._gate_up_proj.weight"])
            gate_s, up_s = gate_up_s.chunk(2, dim=-1)
            shared = F.linear(F.silu(gate_s) * up_s, w["mlp.shared_expert.down_proj.weight"])
        else:
            gate_s = F.linear(h_flat, w["mlp.shared_expert.gate_proj.weight"])
            up_s = F.linear(h_flat, w["mlp.shared_expert.up_proj.weight"])
            shared = F.linear(F.silu(gate_s) * up_s, w["mlp.shared_expert.down_proj.weight"])
        if "mlp.shared_expert_gate.weight" in w:
            shared = shared * torch.sigmoid(F.linear(h_flat, w["mlp.shared_expert_gate.weight"]))
        moe_out = moe_out + shared

    return moe_out.to(h.dtype).reshape_as(h)
