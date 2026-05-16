"""Packed NVFP4 MoE decode path."""
from __future__ import annotations

import os

import torch
import torch.nn.functional as F

from engine.nvfp4_runtime import dual_scalar_bridge
from triton_kernels.nvfp4_moe import (
    nvfp4_grouped_down_weighted_sum,
    nvfp4_grouped_gate_up_silu,
    nvfp4_grouped_gate_up_silu_fast_decode,
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


def _env_first(names: tuple[str, ...]) -> str | None:
    for name in names:
        raw = os.environ.get(name)
        if raw is not None and raw != "":
            return raw
    return None


def _topk_limit_from_env(top_k: int) -> int:
    raw = _env_first(("LYNN_MOE_TOPK_LIMIT", "LYNN_MOE_PROFILE_TOPK_LIMIT"))
    if raw is None:
        return top_k
    limit = int(raw)
    if not (1 <= limit <= top_k):
        raise ValueError(f"MoE top-k limit must be in [1, {top_k}], got {limit}")
    return limit


def _skip_shared_from_env() -> bool:
    raw = _env_first(("LYNN_MOE_SKIP_SHARED", "LYNN_MOE_PROFILE_SKIP_SHARED"))
    if raw is None:
        return False
    return raw.lower() not in {"0", "false", "no", "off"}


def _layer_selected_for_native_cuda(cfg: dict) -> bool:
    spec = os.environ.get("LYNN_NATIVE_ACTIVE_MOE_LAYERS")
    if not spec:
        return True
    layer_idx = cfg.get("layer_idx")
    if layer_idx is None:
        return False
    layer_idx = int(layer_idx)
    from engine.inference_state import LAYER_TYPES

    selected: set[int] = set()
    for raw in spec.split(","):
        item = raw.strip().lower()
        if not item:
            continue
        if item in {"full", "full_attention"}:
            selected.update(i for i, t in enumerate(LAYER_TYPES) if t == "full_attention")
        elif item in {"linear", "linear_attention"}:
            selected.update(i for i, t in enumerate(LAYER_TYPES) if t == "linear_attention")
        else:
            selected.add(int(item))
    return layer_idx in selected


def _active_moe_native_cuda_scalar(
    hidden: torch.Tensor,
    expert_ids: torch.Tensor,
    routing_weights: torch.Tensor,
    w: dict,
) -> torch.Tensor:
    """Opt-in native CUDA scalar contract path.

    This is intentionally slower than the Triton default today. It exists so
    the real grouped native-FP4 kernel can replace the scalar inner loops behind
    the same runtime contract.
    """
    from engine.native_cuda import load_lynn_native_extension

    ext = load_lynn_native_extension(verbose=_env_bool("LYNN_NATIVE_CUDA_VERBOSE", False))
    inter = ext.gate_up_silu_scalar(
        hidden,
        expert_ids,
        w["mlp.experts._gate_up_packed"],
        w["mlp.experts._gate_up_scale"],
        w["mlp.experts._gate_up_global_scale"],
    )
    return ext.down_weighted_sum_scalar(
        inter,
        expert_ids,
        routing_weights,
        w["mlp.experts._down_packed"],
        w["mlp.experts._down_scale"],
        w["mlp.experts._down_global_scale"],
    )


def _gate_up_native_cuda_tile_inter(
    hidden: torch.Tensor,
    expert_ids: torch.Tensor,
    w: dict,
) -> torch.Tensor:
    """P55 opt-in tile-inter CUDA scalar gate/up projection."""
    from engine.native_cuda import load_lynn_native_extension

    ext = load_lynn_native_extension(verbose=_env_bool("LYNN_NATIVE_CUDA_VERBOSE", False))
    return ext.gate_up_silu_tile_inter_scalar(
        hidden,
        expert_ids,
        w["mlp.experts._gate_up_packed"],
        w["mlp.experts._gate_up_scale"],
        w["mlp.experts._gate_up_global_scale"],
        _env_int("LYNN_NATIVE_GATEUP_TILE_INTER", 2),
    )


def _down_weighted_sum_native_cuda_tile(
    inter: torch.Tensor,
    expert_ids: torch.Tensor,
    routing_weights: torch.Tensor,
    w: dict,
) -> torch.Tensor:
    """P48 opt-in tile-hidden non-atomic down projection."""
    from engine.native_cuda import load_lynn_native_extension

    ext = load_lynn_native_extension(verbose=_env_bool("LYNN_NATIVE_CUDA_VERBOSE", False))
    return ext.down_weighted_sum_tile_scalar(
        inter,
        expert_ids,
        routing_weights,
        w["mlp.experts._down_packed"],
        w["mlp.experts._down_scale"],
        w["mlp.experts._down_global_scale"],
        _env_int("LYNN_NATIVE_DOWN_TILE_HIDDEN", 2),
    )


def _active_moe_native_cuda_scalar_contract(
    hidden: torch.Tensor,
    expert_ids: torch.Tensor,
    routing_weights: torch.Tensor,
    w: dict,
) -> torch.Tensor:
    """One-call native active-MoE contract for the future grouped FP4 kernel.

    The P45 implementation still delegates to the scalar reference kernels
    inside the extension.  Its purpose is to freeze the Python/CUDA ABI that the
    true grouped/block-diagonal FP4 kernel will replace.
    """
    from engine.native_cuda import load_lynn_native_extension

    ext = load_lynn_native_extension(verbose=_env_bool("LYNN_NATIVE_CUDA_VERBOSE", False))
    return ext.active_moe_scalar_contract(
        hidden,
        expert_ids,
        routing_weights,
        w["mlp.experts._gate_up_packed"],
        w["mlp.experts._gate_up_scale"],
        w["mlp.experts._gate_up_global_scale"],
        w["mlp.experts._down_packed"],
        w["mlp.experts._down_scale"],
        w["mlp.experts._down_global_scale"],
    )


def _active_moe_native_grouped_per16(
    hidden: torch.Tensor,
    expert_ids: torch.Tensor,
    routing_weights: torch.Tensor,
    w: dict,
) -> torch.Tensor:
    """Reserved production ABI for the real grouped per-16 FP4 kernel.

    P56/P58 closed the tempting scalar tile-inter bridge: it has local speed
    signal, but full-generate greedy drift.  Keep a named backend for the real
    implementation so experiments fail loudly instead of accidentally falling
    back to a rejected scalar path.
    """
    _ = (hidden, expert_ids, routing_weights, w)
    raise NotImplementedError(
        "LYNN_NATIVE_ACTIVE_MOE_BACKEND=grouped_per16 is reserved for the true "
        "grouped per-16 native-FP4 active expert kernel. Current scalar/tile "
        "bridges are intentionally rejected by P56/P58; use 'triton' for "
        "production or add the real grouped_per16 CUDA/CUTLASS implementation "
        "behind this ABI."
    )


def _moe_forward_decode_packed_nvfp4_fixed_triton(h: torch.Tensor, w: dict, cfg: dict) -> torch.Tensor:
    """Fixed-config production fast path for the current R6000 best profile."""
    h_flat = h.reshape(-1, h.shape[-1])
    router_logits = F.linear(h_flat, w["mlp.gate.weight"])
    top_k = int(cfg["num_experts_per_tok"])
    routing_weights, expert_indices = torch.topk(
        router_logits,
        top_k,
        dim=-1,
        sorted=False,
    )
    routing_weights = F.softmax(routing_weights, dim=-1, dtype=torch.float32)[0].contiguous()
    expert_ids = expert_indices[0].to(torch.int32).contiguous()
    limit = _topk_limit_from_env(top_k)
    if limit != top_k:
        expert_ids = expert_ids[:limit].contiguous()
        routing_weights = routing_weights[:limit].contiguous()
        if _env_bool("LYNN_MOE_TOPK_RENORMALIZE", True):
            routing_weights = routing_weights / routing_weights.sum().clamp_min(1e-20)
    hidden = h_flat[0]
    gateup_backend = os.environ.get("LYNN_NATIVE_GATEUP_BACKEND", "triton")
    if gateup_backend == "cuda_tile_inter" and _layer_selected_for_native_cuda(cfg):
        inter = _gate_up_native_cuda_tile_inter(hidden, expert_ids, w)
    elif gateup_backend == "triton_fast_decode":
        inter = nvfp4_grouped_gate_up_silu_fast_decode(
            hidden,
            expert_ids,
            w["mlp.experts._gate_up_packed"],
            w["mlp.experts._gate_up_scale"],
            w["mlp.experts._gate_up_global_scale"],
            block_inter=8,
            block_hidden=256,
            num_warps=4,
        )
    elif gateup_backend == "triton":
        inter = nvfp4_grouped_gate_up_silu(
            hidden,
            expert_ids,
            w["mlp.experts._gate_up_packed"],
            w["mlp.experts._gate_up_scale"],
            w["mlp.experts._gate_up_global_scale"],
            block_inter=8,
            block_hidden=256,
            num_warps=4,
        )
    elif gateup_backend == "cuda_tile_inter":
        inter = nvfp4_grouped_gate_up_silu(
            hidden,
            expert_ids,
            w["mlp.experts._gate_up_packed"],
            w["mlp.experts._gate_up_scale"],
            w["mlp.experts._gate_up_global_scale"],
            block_inter=8,
            block_hidden=256,
            num_warps=4,
        )
    else:
        raise ValueError(
            "LYNN_NATIVE_GATEUP_BACKEND must be 'triton', 'triton_fast_decode', or 'cuda_tile_inter', got "
            f"{gateup_backend!r}"
        )
    down_backend = os.environ.get("LYNN_NATIVE_DOWN_BACKEND", "triton")
    if down_backend == "cuda_tile" and _layer_selected_for_native_cuda(cfg):
        moe_out = _down_weighted_sum_native_cuda_tile(inter, expert_ids, routing_weights, w).reshape_as(h_flat)
    elif down_backend == "triton":
        moe_out = nvfp4_grouped_down_weighted_sum(
            inter,
            expert_ids,
            routing_weights,
            w["mlp.experts._down_packed"],
            w["mlp.experts._down_scale"],
            w["mlp.experts._down_global_scale"],
            block_hidden=8,
            block_inter=512,
            num_warps=8,
        ).reshape_as(h_flat)
    elif down_backend == "cuda_tile":
        moe_out = nvfp4_grouped_down_weighted_sum(
            inter,
            expert_ids,
            routing_weights,
            w["mlp.experts._down_packed"],
            w["mlp.experts._down_scale"],
            w["mlp.experts._down_global_scale"],
            block_hidden=8,
            block_inter=512,
            num_warps=8,
        ).reshape_as(h_flat)
    else:
        raise ValueError("LYNN_NATIVE_DOWN_BACKEND must be 'triton' or 'cuda_tile', got " f"{down_backend!r}")

    if _skip_shared_from_env():
        return moe_out.to(h.dtype).reshape_as(h)

    if "mlp.shared_expert.gate_proj.weight" in w:
        if "mlp.shared_expert._gate_up_proj.weight" in w:
            gate_up_s = F.linear(h_flat, w["mlp.shared_expert._gate_up_proj.weight"])
            gate_s, up_s = gate_up_s.chunk(2, dim=-1)
        else:
            gate_s = F.linear(h_flat, w["mlp.shared_expert.gate_proj.weight"])
            up_s = F.linear(h_flat, w["mlp.shared_expert.up_proj.weight"])
        shared = F.linear(F.silu(gate_s) * up_s, w["mlp.shared_expert.down_proj.weight"])
        if "mlp.shared_expert_gate.weight" in w:
            shared = shared * torch.sigmoid(F.linear(h_flat, w["mlp.shared_expert_gate.weight"]))
        moe_out = moe_out + shared
    return moe_out.to(h.dtype).reshape_as(h)


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
    if _env_bool("LYNN_MOE_FAST_FIXED", True):
        if _env_bool("LYNN_ROUTER_TOPK_SORTED", False):
            raise RuntimeError("LYNN_MOE_FAST_FIXED requires LYNN_ROUTER_TOPK_SORTED=0")
        if os.environ.get("LYNN_MOE_PROFILE_SKIP_ACTIVE", "0") == "1":
            raise RuntimeError("LYNN_MOE_FAST_FIXED does not support LYNN_MOE_PROFILE_SKIP_ACTIVE")
        if os.environ.get("LYNN_NATIVE_ACTIVE_MOE_BACKEND", "triton") != "triton":
            raise RuntimeError("LYNN_MOE_FAST_FIXED requires LYNN_NATIVE_ACTIVE_MOE_BACKEND=triton")
        if os.environ.get("LYNN_NATIVE_GATEUP_BACKEND", "triton") not in {
            "triton",
            "triton_fast_decode",
            "cuda_tile_inter",
        }:
            raise RuntimeError(
                "LYNN_MOE_FAST_FIXED requires LYNN_NATIVE_GATEUP_BACKEND=triton, "
                "triton_fast_decode, or cuda_tile_inter"
            )
        if os.environ.get("LYNN_NATIVE_DOWN_BACKEND", "triton") not in {"triton", "cuda_tile"}:
            raise RuntimeError("LYNN_MOE_FAST_FIXED requires LYNN_NATIVE_DOWN_BACKEND=triton or cuda_tile")
        if (
            _env_int("LYNN_MOE_GATE_BLOCK_INTER", 8),
            _env_int("LYNN_MOE_GATE_BLOCK_HIDDEN", 256),
            _env_int("LYNN_MOE_DOWN_BLOCK_HIDDEN", 8),
            _env_int("LYNN_MOE_DOWN_BLOCK_INTER", 512),
            _env_int("LYNN_MOE_GATE_NUM_WARPS", 4),
            _env_int("LYNN_MOE_DOWN_NUM_WARPS", 8),
        ) != (8, 256, 8, 512, 4, 8):
            raise RuntimeError("LYNN_MOE_FAST_FIXED only supports the current R6000 best MoE kernel config")
        return _moe_forward_decode_packed_nvfp4_fixed_triton(h, w, cfg)

    router_logits = F.linear(h_flat, w["mlp.gate.weight"])
    routing_weights, expert_indices = torch.topk(
        router_logits,
        int(cfg["num_experts_per_tok"]),
        dim=-1,
        sorted=_env_bool("LYNN_ROUTER_TOPK_SORTED", False),
    )
    routing_weights = F.softmax(routing_weights, dim=-1, dtype=torch.float32)[0]
    # Triton kernels consume int32 expert ids. Keep this as int32 once here so
    # gate/up and down do not each pay a tiny per-layer dtype conversion.
    expert_ids = expert_indices[0].to(torch.int32).contiguous()
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
        backend = os.environ.get("LYNN_NATIVE_ACTIVE_MOE_BACKEND", "triton")
        down_backend = os.environ.get("LYNN_NATIVE_DOWN_BACKEND", "triton")
        if backend == "grouped_per16" and _layer_selected_for_native_cuda(cfg):
            moe_out = _active_moe_native_grouped_per16(hidden, expert_ids, routing_weights, w).reshape_as(h_flat)
        elif backend == "cuda_scalar_contract" and _layer_selected_for_native_cuda(cfg):
            moe_out = _active_moe_native_cuda_scalar_contract(hidden, expert_ids, routing_weights, w).reshape_as(h_flat)
        elif backend == "cuda_scalar" and _layer_selected_for_native_cuda(cfg):
            moe_out = _active_moe_native_cuda_scalar(hidden, expert_ids, routing_weights, w).reshape_as(h_flat)
        elif backend in {"triton", "cuda_scalar", "cuda_scalar_contract", "grouped_per16"}:
            gateup_backend = os.environ.get("LYNN_NATIVE_GATEUP_BACKEND", "triton")
            if gateup_backend == "cuda_tile_inter" and _layer_selected_for_native_cuda(cfg):
                inter = _gate_up_native_cuda_tile_inter(hidden, expert_ids, w)
            elif gateup_backend == "triton_fast_decode":
                inter = nvfp4_grouped_gate_up_silu_fast_decode(
                    hidden,
                    expert_ids,
                    w["mlp.experts._gate_up_packed"],
                    w["mlp.experts._gate_up_scale"],
                    w["mlp.experts._gate_up_global_scale"],
                    block_inter=_env_int("LYNN_MOE_GATE_BLOCK_INTER", 8),
                    block_hidden=_env_int("LYNN_MOE_GATE_BLOCK_HIDDEN", 256),
                    num_warps=_env_int("LYNN_MOE_GATE_NUM_WARPS", 4),
                )
            elif gateup_backend == "triton":
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
            elif gateup_backend == "cuda_tile_inter":
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
            else:
                raise ValueError(
                    "LYNN_NATIVE_GATEUP_BACKEND must be 'triton', 'triton_fast_decode', or 'cuda_tile_inter', got "
                    f"{gateup_backend!r}"
                )
            if down_backend == "cuda_tile" and _layer_selected_for_native_cuda(cfg):
                moe_out = _down_weighted_sum_native_cuda_tile(inter, expert_ids, routing_weights, w).reshape_as(h_flat)
            elif down_backend == "triton":
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
            else:
                raise ValueError("LYNN_NATIVE_DOWN_BACKEND must be 'triton' or 'cuda_tile', got " f"{down_backend!r}")
        else:
            raise ValueError(
                "LYNN_NATIVE_ACTIVE_MOE_BACKEND must be 'triton', 'cuda_scalar', "
                "'cuda_scalar_contract', or 'grouped_per16', "
                f"got {backend!r}"
            )

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
