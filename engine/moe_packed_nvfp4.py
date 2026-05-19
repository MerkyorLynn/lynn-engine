"""Packed NVFP4 MoE decode path."""
from __future__ import annotations

import os

import torch
import torch.nn.functional as F

from engine.nvfp4_runtime import _quantize_activation_to_fp4, dual_scalar_bridge
from triton_kernels.nvfp4_moe import (
    nvfp4_grouped_down_weighted_sum,
    nvfp4_grouped_down_weighted_sum_effective_scale,
    nvfp4_grouped_down_weighted_sum_prepared,
    nvfp4_grouped_gate_up_silu,
    nvfp4_grouped_gate_up_silu_fast_decode,
    nvfp4_grouped_gate_up_silu_fast_decode_effective_scale,
    nvfp4_grouped_gate_up_silu_fast_decode_prepared,
)
from triton_kernels.shared_expert_gate import (
    HAS_TRITON as HAS_SHARED_EXPERT_GATE_TRITON,
    add_shared_expert_gate_from_scalar_triton,
    apply_shared_expert_gate_triton,
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


def _w4a8_fake_quant_mode() -> str:
    mode = os.environ.get("LYNN_W4A8_FAKE_QUANT_ACTIVE", "off").lower()
    if mode in {"0", "false", "no", "off", ""}:
        return "off"
    if mode not in {"gateup", "full"}:
        raise ValueError("LYNN_W4A8_FAKE_QUANT_ACTIVE must be off, gateup, or full")
    return mode


def _use_moe_effective_scale(w: dict) -> bool:
    return (
        _env_bool("LYNN_MOE_EFFECTIVE_SCALE", False)
        and "mlp.experts._gate_up_effective_scale" in w
        and "mlp.experts._down_effective_scale" in w
    )


def _fake_quant_fp8_activation(x: torch.Tensor) -> torch.Tensor:
    """Research-only FP8 activation round-trip for W4A8 quality gates.

    This is intentionally controlled by `LYNN_W4A8_FAKE_QUANT_ACTIVE` and never
    used by default. It mirrors P104's best variant: E4M3, per-16 scaling.
    """
    fmt = os.environ.get("LYNN_W4A8_FAKE_QUANT_FORMAT", "e4m3").lower()
    granularity = os.environ.get("LYNN_W4A8_FAKE_QUANT_GRANULARITY", "per16").lower()
    if fmt == "e4m3":
        if not hasattr(torch, "float8_e4m3fn"):
            raise RuntimeError("torch.float8_e4m3fn is required for W4A8 fake quant")
        fp8_dtype = torch.float8_e4m3fn
    elif fmt == "e5m2":
        if not hasattr(torch, "float8_e5m2"):
            raise RuntimeError("torch.float8_e5m2 is required for W4A8 fake quant")
        fp8_dtype = torch.float8_e5m2
    else:
        raise ValueError("LYNN_W4A8_FAKE_QUANT_FORMAT must be e4m3 or e5m2")

    max_fp8 = float(torch.finfo(fp8_dtype).max)
    x32 = x.float()
    if granularity == "tensor":
        scale = (x32.abs().amax() / max_fp8).clamp_min(1e-8)
        return ((x32 / scale).to(fp8_dtype).float() * scale).to(x.dtype)
    if granularity == "row":
        x2 = x32.view(1, -1) if x32.ndim == 1 else x32
        scale = (x2.abs().amax(dim=-1, keepdim=True) / max_fp8).clamp_min(1e-8)
        y = ((x2 / scale).to(fp8_dtype).float() * scale).to(x.dtype)
        return y.view_as(x) if x32.ndim == 1 else y
    if granularity == "per16":
        if x32.shape[-1] % 16 != 0:
            raise ValueError(f"W4A8 per16 fake quant requires last dim divisible by 16, got {tuple(x.shape)}")
        shape = x32.shape
        grouped = x32.reshape(-1, shape[-1] // 16, 16)
        scale = (grouped.abs().amax(dim=-1, keepdim=True) / max_fp8).clamp_min(1e-8)
        return ((grouped / scale).to(fp8_dtype).float() * scale).reshape(shape).to(x.dtype)
    raise ValueError("LYNN_W4A8_FAKE_QUANT_GRANULARITY must be tensor, row, or per16")


def _topk_limit_from_env(top_k: int) -> int:
    raw = _env_first(("LYNN_MOE_TOPK_LIMIT", "LYNN_MOE_PROFILE_TOPK_LIMIT"))
    if raw is None:
        return top_k
    limit = int(raw)
    if not (1 <= limit <= top_k):
        raise ValueError(f"MoE top-k limit must be in [1, {top_k}], got {limit}")
    return limit


def _router_topk(
    router_logits: torch.Tensor,
    top_k: int,
    *,
    sorted: bool,
    scratch_owner: dict,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run router top-k, optionally reusing caller-owned output buffers.

    P163 showed `torch.topk(..., out=...)` is bit-exact for the decode shape and
    saves a small but measurable boundary cost.  Keep it opt-in because the
    scratch tensors are stored on the mutable per-layer weight dict.
    """
    if not _env_bool("LYNN_ROUTER_TOPK_OUT_BUFFER", False):
        return torch.topk(router_logits, top_k, dim=-1, sorted=sorted)
    if router_logits.ndim != 2 or router_logits.shape[0] != 1:
        return torch.topk(router_logits, top_k, dim=-1, sorted=sorted)
    values_key = "mlp.gate._topk_values_scratch"
    indices_key = "mlp.gate._topk_indices_scratch"
    values = scratch_owner.get(values_key)
    indices = scratch_owner.get(indices_key)
    expected_shape = (1, top_k)
    if (
        values is None
        or tuple(values.shape) != expected_shape
        or values.device != router_logits.device
        or values.dtype != router_logits.dtype
    ):
        values = torch.empty(expected_shape, device=router_logits.device, dtype=router_logits.dtype)
        scratch_owner[values_key] = values
    if (
        indices is None
        or tuple(indices.shape) != expected_shape
        or indices.device != router_logits.device
        or indices.dtype != torch.long
    ):
        indices = torch.empty(expected_shape, device=router_logits.device, dtype=torch.long)
        scratch_owner[indices_key] = indices
    torch.topk(router_logits, top_k, dim=-1, sorted=sorted, out=(values, indices))
    return values, indices


def _router_linear(h_flat: torch.Tensor, w: dict) -> torch.Tensor:
    """Run the MoE router projection, optionally with caller-owned output.

    P177 showed `torch.mm(hidden, gate_weight.t(), out=logits_scratch)` is
    bit-exact for the single-token decode router and removes a small allocation
    boundary. The transposed weight and scratch are attached at load time by the
    resident runner.
    """
    if not _env_bool("LYNN_ROUTER_LINEAR_OUT_BUFFER", False):
        return F.linear(h_flat, w["mlp.gate.weight"])
    weight_t = w.get("mlp.gate.weight_t")
    logits = w.get("mlp.gate._logits_scratch")
    if (
        weight_t is None
        or logits is None
        or h_flat.ndim != 2
        or h_flat.shape[0] != 1
        or logits.shape != (1, w["mlp.gate.weight"].shape[0])
        or logits.device != h_flat.device
        or logits.dtype != h_flat.dtype
    ):
        return F.linear(h_flat, w["mlp.gate.weight"])
    torch.mm(h_flat, weight_t, out=logits)
    return logits


def _router_softmax(
    routing_logits: torch.Tensor,
    *,
    scratch_owner: dict,
) -> torch.Tensor:
    """Run router softmax, optionally reusing a caller-owned float32 buffer."""
    if not _env_bool("LYNN_ROUTER_SOFTMAX_OUT_BUFFER", False):
        return F.softmax(routing_logits, dim=-1, dtype=torch.float32)[0].contiguous()
    if routing_logits.ndim != 2 or routing_logits.shape[0] != 1:
        return F.softmax(routing_logits, dim=-1, dtype=torch.float32)[0].contiguous()
    values_key = "mlp.gate._softmax_values_scratch"
    expected_shape = tuple(routing_logits.shape)
    values = scratch_owner.get(values_key)
    if (
        values is None
        or tuple(values.shape) != expected_shape
        or values.device != routing_logits.device
        or values.dtype != torch.float32
    ):
        values = torch.empty(expected_shape, device=routing_logits.device, dtype=torch.float32)
        scratch_owner[values_key] = values
    torch.softmax(routing_logits, dim=-1, dtype=torch.float32, out=values)
    return values[0]


def _skip_shared_from_env() -> bool:
    raw = _env_first(("LYNN_MOE_SKIP_SHARED", "LYNN_MOE_PROFILE_SKIP_SHARED"))
    if raw is None:
        return False
    return raw.lower() not in {"0", "false", "no", "off"}


def _apply_shared_expert_gate(h_flat: torch.Tensor, shared: torch.Tensor, w: dict) -> torch.Tensor:
    if "mlp.shared_expert_gate.weight" not in w:
        return shared
    backend = os.environ.get("LYNN_SHARED_EXPERT_GATE_BACKEND", "torch")
    if backend == "torch":
        return shared * torch.sigmoid(F.linear(h_flat, w["mlp.shared_expert_gate.weight"]))
    if backend == "torch_inplace":
        shared.mul_(torch.sigmoid(F.linear(h_flat, w["mlp.shared_expert_gate.weight"])))
        return shared
    if backend == "triton":
        if not HAS_SHARED_EXPERT_GATE_TRITON:
            raise RuntimeError("LYNN_SHARED_EXPERT_GATE_BACKEND=triton requires Triton")
        return apply_shared_expert_gate_triton(shared, h_flat, w["mlp.shared_expert_gate.weight"])
    raise ValueError("LYNN_SHARED_EXPERT_GATE_BACKEND must be 'torch', 'torch_inplace', or 'triton', got " f"{backend!r}")


def _add_shared_expert_output(moe_out: torch.Tensor, shared: torch.Tensor) -> torch.Tensor:
    if _env_bool("LYNN_MOE_ADD_SHARED_INPLACE", False):
        moe_out.add_(shared)
        return moe_out
    return moe_out + shared


def _finalize_shared_expert_output(h_flat: torch.Tensor, moe_out: torch.Tensor, shared: torch.Tensor, w: dict) -> torch.Tensor:
    backend = os.environ.get("LYNN_SHARED_EXPERT_GATE_BACKEND", "torch")
    if backend == "torch_scalar_add_triton" and "mlp.shared_expert_gate.weight" in w:
        if not HAS_SHARED_EXPERT_GATE_TRITON:
            raise RuntimeError("LYNN_SHARED_EXPERT_GATE_BACKEND=torch_scalar_add_triton requires Triton")
        gate = torch.sigmoid(F.linear(h_flat, w["mlp.shared_expert_gate.weight"]))
        return add_shared_expert_gate_from_scalar_triton(moe_out, shared, gate)
    shared = _apply_shared_expert_gate(h_flat, shared, w)
    return _add_shared_expert_output(moe_out, shared)


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


def _gate_up_native_split16_fp4(
    hidden: torch.Tensor,
    expert_ids: torch.Tensor,
    w: dict,
) -> torch.Tensor:
    """P98 experimental SM120a split16 FP4 MMA gate/up projection.

    This backend quantizes the current decode activation to Lynn-native E2M1
    per-16 format, then consumes the existing packed NVFP4 expert weights
    directly. P98 proved this is not production-promotable yet: the activation
    quantization is not CUDA-graph-capture-safe and changes greedy generations.
    """
    from engine.native_cuda import load_lynn_native_extension

    if (
        _env_bool("LYNN_LINEAR_BLOCK_GRAPH", False)
        or _env_bool("LYNN_LINEAR_BLOCK_GRAPH_REUSE", False)
        or _env_bool("LYNN_LINEAR_BLOCK_GRAPH_PREWARM", False)
    ):
        raise RuntimeError(
            "LYNN_NATIVE_GATEUP_BACKEND=split16_fp4 is an experimental P98 research backend and is not "
            "CUDA-graph-capture-safe. Set LYNN_LINEAR_BLOCK_GRAPH=0, LYNN_LINEAR_BLOCK_GRAPH_REUSE=0, "
            "and LYNN_LINEAR_BLOCK_GRAPH_PREWARM=0 for isolated probes; do not use it in production."
        )

    ext = load_lynn_native_extension(verbose=_env_bool("LYNN_NATIVE_CUDA_VERBOSE", False))
    act_packed, act_scale = _quantize_activation_to_fp4(hidden.view(1, -1))
    return ext.gate_up_silu_split16_topk_fp4(
        act_packed[0].contiguous(),
        act_scale[0].float().contiguous(),
        expert_ids,
        w["mlp.experts._gate_up_packed"],
        w["mlp.experts._gate_up_scale"],
        w["mlp.experts._gate_up_global_scale"].float(),
        _env_int("LYNN_NATIVE_FP4_MMA_SCALE_BYTE", 127),
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
    from engine.native_cuda import load_lynn_native_extension

    ext = load_lynn_native_extension(verbose=_env_bool("LYNN_NATIVE_CUDA_VERBOSE", False))
    return ext.active_moe_grouped_per16_contract(
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


def _active_moe_native_grouped_per16_fused(
    hidden: torch.Tensor,
    expert_ids: torch.Tensor,
    routing_weights: torch.Tensor,
    w: dict,
) -> torch.Tensor:
    """Fail-loud ABI for the true one-boundary grouped per-16 active kernel."""
    from engine.native_cuda import load_lynn_native_extension

    ext = load_lynn_native_extension(verbose=_env_bool("LYNN_NATIVE_CUDA_VERBOSE", False))
    return ext.active_moe_grouped_per16_fused_contract(
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


def _active_moe_native_grouped_per16_nonatomic(
    hidden: torch.Tensor,
    expert_ids: torch.Tensor,
    routing_weights: torch.Tensor,
    w: dict,
) -> torch.Tensor:
    """Native-owned scratch, non-atomic grouped per-16 reference backend."""
    from engine.native_cuda import load_lynn_native_extension

    ext = load_lynn_native_extension(verbose=_env_bool("LYNN_NATIVE_CUDA_VERBOSE", False))
    return ext.active_moe_grouped_per16_nonatomic_reference(
        hidden,
        expert_ids,
        routing_weights,
        w["mlp.experts._gate_up_packed"],
        w["mlp.experts._gate_up_scale"],
        w["mlp.experts._gate_up_global_scale"],
        w["mlp.experts._down_packed"],
        w["mlp.experts._down_scale"],
        w["mlp.experts._down_global_scale"],
        _env_int("LYNN_NATIVE_GATEUP_TILE_INTER", 2),
        _env_int("LYNN_NATIVE_DOWN_TILE_HIDDEN", 2),
    )


def _active_moe_packed_pretransposed_graphsafe_v31(
    hidden: torch.Tensor,
    expert_ids: torch.Tensor,
    routing_weights: torch.Tensor,
    w: dict,
) -> torch.Tensor:
    """V3.1 graph-safe pretransposed MoE: true zero-allocation hot path.

    Load-time prep (lazy, first call only):
      - Dequant packed NVFP4 gate_up/down weights to BF16
      - Pretranspose: W_fused_T [2048, 8192], W_down_T [8, 512, 2048]
      - Preallocate scratch: gate_up[1,8192], inter[8,1,512], down[8,1,2048], out[2048]

    Hot path: mm_out + silu_out + mul_ + bmm_out + custom CUDA reduce kernel.
    No torch::empty/zeros/to/sum/new tensor in hot path.
    """
    from engine.native_cuda import load_lynn_native_extension

    # ── Lazy one-time weight prep (first call per layer) ──
    if "_graphsafe_v31_W_fused_T" not in w:
        _prepare_graphsafe_v31_weights(w, hidden.device)

    ext = load_lynn_native_extension(verbose=_env_bool("LYNN_NATIVE_CUDA_VERBOSE", False))

    # Gather experts into pretransposed scratch (pre-graph region)
    _gather_and_pretranspose_v31(w, expert_ids)

    x_2d = hidden.view(1, 2048)
    return ext.moe_packed_pretransposed_graphsafe_v3(
        x_2d,
        routing_weights,
        w["_graphsafe_v31_W_fused_T"],
        w["_graphsafe_v31_W_down_T"],
        w["_graphsafe_v31_gate_up_scratch"],
        w["_graphsafe_v31_inter_scratch"],
        w["_graphsafe_v31_down_scratch"],
        w["_graphsafe_v31_out_scratch"],
    )


def _prepare_graphsafe_v32_weights(w: dict, device) -> None:
    """One-time scratch allocation for V3.2 graph-safe exact scalar path.

    Does NOT pre-dequant weights. Keeps original packed NVFP4 weights.
    Only allocates caller-owned scratch buffers for intermediate and output.
    """
    top_k = 8
    w["_graphsafe_v32_inter_scratch"] = torch.empty(top_k, 512, device=device, dtype=torch.bfloat16)
    w["_graphsafe_v32_out_scratch"] = torch.empty(2048, device=device, dtype=torch.bfloat16)


def _active_moe_packed_pretransposed_graphsafe_v32_ordered(
    hidden: torch.Tensor,
    expert_ids: torch.Tensor,
    routing_weights: torch.Tensor,
    w: dict,
) -> torch.Tensor:
    """V3.2 graph-safe exact scalar MoE: caller-owned scratch, native FP4 dequant.

    Load-time prep (lazy, first call only):
      - Preallocate scratch: inter[8,512], out[2048]
      - Weights stay in original packed NVFP4 format (no pre-dequant)

    Hot path: native FP4→FP32 scalar gate/up + native FP4→FP32 scalar down.
    No torch::empty/zeros/to/sum/new tensor in hot path.
    Route weights kept in FP32 (no BF16 truncation).
    Slot order strictly follows expert_ids.

    Designed to match Triton reference exactly (P37 exact 3/3 target).
    Slower than V3.1 cuBLAS path.
    """
    from engine.native_cuda import load_lynn_native_extension

    if "_graphsafe_v32_inter_scratch" not in w:
        _prepare_graphsafe_v32_weights(w, hidden.device)

    ext = load_lynn_native_extension(verbose=_env_bool("LYNN_NATIVE_CUDA_VERBOSE", False))

    ext.active_moe_scalar_out_reference(
        hidden,
        expert_ids,
        routing_weights,
        w["mlp.experts._gate_up_packed"],
        w["mlp.experts._gate_up_scale"],
        w["mlp.experts._gate_up_global_scale"],
        w["mlp.experts._down_packed"],
        w["mlp.experts._down_scale"],
        w["mlp.experts._down_global_scale"],
        w["_graphsafe_v32_inter_scratch"],
        w["_graphsafe_v32_out_scratch"],
    )
    return w["_graphsafe_v32_out_scratch"]


def _prepare_graphsafe_v31_weights(w: dict, device) -> None:
    """One-time scratch allocation for V3.1 graph-safe path.

    Does NOT pre-dequant all 256 experts (too much memory).
    Instead, preallocates scratch buffers. Per-call dequant of 8 selected
    experts happens in _gather_and_pretranspose_v31.
    """
    top_k = 8
    w["_graphsafe_v31_W_fused_T"] = torch.empty(2048, top_k * 1024, device=device, dtype=torch.bfloat16)
    w["_graphsafe_v31_W_down_T"] = torch.empty(top_k, 512, 2048, device=device, dtype=torch.bfloat16)
    w["_graphsafe_v31_gate_up_scratch"] = torch.empty(1, top_k * 1024, device=device, dtype=torch.bfloat16)
    w["_graphsafe_v31_inter_scratch"] = torch.empty(top_k, 1, 512, device=device, dtype=torch.bfloat16)
    w["_graphsafe_v31_down_scratch"] = torch.empty(top_k, 1, 2048, device=device, dtype=torch.bfloat16)
    w["_graphsafe_v31_out_scratch"] = torch.empty(2048, device=device, dtype=torch.bfloat16)


def _dequant_nvfp4_slot(packed, scale, global_scale, device):
    """Dequant a single slot [rows, cols/2] → [rows, cols] BF16."""
    E2M1_TABLE = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32, device=device)
    low = (packed & 0x0F).int()
    high = ((packed >> 4) & 0x0F).int()
    low_val = E2M1_TABLE[low & 7] * (1 - 2 * ((low >> 3) & 1).float())
    high_val = E2M1_TABLE[high & 7] * (1 - 2 * ((high >> 3) & 1).float())
    K = packed.shape[-1] * 2
    result = torch.zeros(*packed.shape[:-1], K, device=device, dtype=torch.float32)
    result[..., 0::2] = low_val
    result[..., 1::2] = high_val
    inv_g = 1.0 / global_scale.float().item()
    se = scale.float().unsqueeze(-1).expand(*scale.shape, 16).reshape(*packed.shape[:-1], K)
    return (result * se * inv_g).to(torch.bfloat16)


def _gather_and_pretranspose_v31(w: dict, expert_ids: torch.Tensor) -> None:
    """Dequant + gather 8 selected experts into pretransposed scratch."""
    device = expert_ids.device
    gu_packed = w["mlp.experts._gate_up_packed"]       # [256, 1024, 1024]
    gu_scale = w["mlp.experts._gate_up_scale"]         # [256, 1024, 128]
    gu_global = w["mlp.experts._gate_up_global_scale"] # scalar
    d_packed = w["mlp.experts._down_packed"]            # [256, 2048, 256]
    d_scale = w["mlp.experts._down_scale"]              # [256, 2048, 32]
    d_global = w["mlp.experts._down_global_scale"]      # scalar

    W_fused_T = w["_graphsafe_v31_W_fused_T"]
    W_down_T = w["_graphsafe_v31_W_down_T"]

    ids = expert_ids.long()
    # Dequant only the 8 selected experts
    slot_gu_packed = gu_packed[ids]   # [8, 1024, 1024]
    slot_gu_scale = gu_scale[ids]     # [8, 1024, 128]
    slot_d_packed = d_packed[ids]     # [8, 2048, 256]
    slot_d_scale = d_scale[ids]       # [8, 2048, 32]

    slot_gu_bf16 = _dequant_nvfp4_slot(slot_gu_packed, slot_gu_scale, gu_global, device)  # [8, 1024, 2048]
    slot_d_bf16 = _dequant_nvfp4_slot(slot_d_packed, slot_d_scale, d_global, device)      # [8, 2048, 512]

    # Pretranspose into scratch
    W_fused_T.copy_(slot_gu_bf16.reshape(8 * 1024, 2048).t())
    W_down_T.copy_(slot_d_bf16.transpose(1, 2))


def _active_moe_native_grouped_per16_nonatomic_out(
    hidden: torch.Tensor,
    expert_ids: torch.Tensor,
    routing_weights: torch.Tensor,
    w: dict,
) -> torch.Tensor:
    """Caller-owned scratch variant for CUDA graph capture probes."""
    from engine.native_cuda import load_lynn_native_extension

    inter_scratch = w.get("mlp.experts._active_inter_scratch")
    out_scratch = w.get("mlp.experts._active_out_scratch")
    if inter_scratch is None or out_scratch is None:
        raise RuntimeError(
            "LYNN_NATIVE_ACTIVE_MOE_BACKEND=grouped_per16_nonatomic_out requires "
            "LYNN_MOE_ACTIVE_SCRATCH=1 so resident_runner preallocates MoE scratch."
        )

    ext = load_lynn_native_extension(verbose=_env_bool("LYNN_NATIVE_CUDA_VERBOSE", False))
    return ext.active_moe_grouped_per16_nonatomic_out_reference(
        hidden,
        expert_ids,
        routing_weights,
        w["mlp.experts._gate_up_packed"],
        w["mlp.experts._gate_up_scale"],
        w["mlp.experts._gate_up_global_scale"],
        w["mlp.experts._down_packed"],
        w["mlp.experts._down_scale"],
        w["mlp.experts._down_global_scale"],
        inter_scratch,
        out_scratch,
        _env_int("LYNN_NATIVE_GATEUP_TILE_INTER", 2),
        _env_int("LYNN_NATIVE_DOWN_TILE_HIDDEN", 2),
    )


def _moe_forward_decode_packed_nvfp4_fixed_triton(h: torch.Tensor, w: dict, cfg: dict) -> torch.Tensor:
    """Fixed-config production fast path for the current R6000 best profile."""
    h_flat = h.reshape(-1, h.shape[-1])
    router_logits = _router_linear(h_flat, w)
    top_k = int(cfg["num_experts_per_tok"])
    routing_weights, expert_indices = _router_topk(
        router_logits,
        top_k,
        sorted=False,
        scratch_owner=w,
    )
    routing_weights = _router_softmax(routing_weights, scratch_owner=w)
    expert_ids = expert_indices[0].to(torch.int32).contiguous()
    limit = _topk_limit_from_env(top_k)
    if limit != top_k:
        expert_ids = expert_ids[:limit].contiguous()
        routing_weights = routing_weights[:limit].contiguous()
        if _env_bool("LYNN_MOE_TOPK_RENORMALIZE", True):
            routing_weights = routing_weights / routing_weights.sum().clamp_min(1e-20)
    hidden = h_flat[0]
    w4a8_mode = _w4a8_fake_quant_mode()
    if w4a8_mode in {"gateup", "full"}:
        hidden = _fake_quant_fp8_activation(hidden)
    gateup_backend = os.environ.get("LYNN_NATIVE_GATEUP_BACKEND", "triton")
    prepared_triton = _env_bool("LYNN_MOE_TRITON_PREPARED", False)
    if prepared_triton:
        if gateup_backend != "triton_fast_decode":
            raise RuntimeError("LYNN_MOE_TRITON_PREPARED requires LYNN_NATIVE_GATEUP_BACKEND=triton_fast_decode")
        inter_scratch = w.get("mlp.experts._active_inter_scratch")
        if inter_scratch is None:
            raise RuntimeError("LYNN_MOE_TRITON_PREPARED requires LYNN_MOE_ACTIVE_SCRATCH=1")
        inter = nvfp4_grouped_gate_up_silu_fast_decode_prepared(
            hidden,
            expert_ids,
            w["mlp.experts._gate_up_packed"],
            w["mlp.experts._gate_up_effective_scale"] if _use_moe_effective_scale(w) else w["mlp.experts._gate_up_scale"],
            w["mlp.experts._gate_up_global_scale"],
            inter_scratch,
            scale_effective=_use_moe_effective_scale(w),
        )
    elif gateup_backend == "split16_fp4" and _layer_selected_for_native_cuda(cfg):
        inter = _gate_up_native_split16_fp4(hidden, expert_ids, w)
    elif gateup_backend == "cuda_tile_inter" and _layer_selected_for_native_cuda(cfg):
        inter = _gate_up_native_cuda_tile_inter(hidden, expert_ids, w)
    elif gateup_backend == "triton_fast_decode":
        gateup_fn = (
            nvfp4_grouped_gate_up_silu_fast_decode_effective_scale
            if _use_moe_effective_scale(w)
            else nvfp4_grouped_gate_up_silu_fast_decode
        )
        inter = gateup_fn(
            hidden,
            expert_ids,
            w["mlp.experts._gate_up_packed"],
            w["mlp.experts._gate_up_effective_scale"]
            if _use_moe_effective_scale(w)
            else w["mlp.experts._gate_up_scale"],
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
    elif gateup_backend in {"cuda_tile_inter", "split16_fp4"}:
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
            "LYNN_NATIVE_GATEUP_BACKEND must be 'triton', 'triton_fast_decode', 'cuda_tile_inter', "
            "or 'split16_fp4', got "
            f"{gateup_backend!r}"
        )
    if w4a8_mode == "full":
        inter = _fake_quant_fp8_activation(inter)
    down_backend = os.environ.get("LYNN_NATIVE_DOWN_BACKEND", "triton")
    if down_backend == "cuda_tile" and _layer_selected_for_native_cuda(cfg):
        moe_out = _down_weighted_sum_native_cuda_tile(inter, expert_ids, routing_weights, w).reshape_as(h_flat)
    elif prepared_triton:
        if down_backend != "triton":
            raise RuntimeError("LYNN_MOE_TRITON_PREPARED requires LYNN_NATIVE_DOWN_BACKEND=triton")
        out_scratch = w.get("mlp.experts._active_out_scratch")
        if out_scratch is None:
            raise RuntimeError("LYNN_MOE_TRITON_PREPARED requires LYNN_MOE_ACTIVE_SCRATCH=1")
        moe_out = nvfp4_grouped_down_weighted_sum_prepared(
            inter,
            expert_ids,
            routing_weights,
            w["mlp.experts._down_packed"],
            w["mlp.experts._down_effective_scale"] if _use_moe_effective_scale(w) else w["mlp.experts._down_scale"],
            w["mlp.experts._down_global_scale"],
            out_scratch,
            scale_effective=_use_moe_effective_scale(w),
        ).reshape_as(h_flat)
    elif down_backend == "triton":
        down_fn = (
            nvfp4_grouped_down_weighted_sum_effective_scale
            if _use_moe_effective_scale(w)
            else nvfp4_grouped_down_weighted_sum
        )
        moe_out = down_fn(
            inter,
            expert_ids,
            routing_weights,
            w["mlp.experts._down_packed"],
            w["mlp.experts._down_effective_scale"] if _use_moe_effective_scale(w) else w["mlp.experts._down_scale"],
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
        moe_out = _finalize_shared_expert_output(h_flat, moe_out, shared, w)
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
            "split16_fp4",
        }:
            raise RuntimeError(
                "LYNN_MOE_FAST_FIXED requires LYNN_NATIVE_GATEUP_BACKEND=triton, "
                "triton_fast_decode, cuda_tile_inter, or split16_fp4"
            )
        if os.environ.get("LYNN_NATIVE_DOWN_BACKEND", "triton") not in {"triton", "cuda_tile"}:
            raise RuntimeError("LYNN_MOE_FAST_FIXED requires LYNN_NATIVE_DOWN_BACKEND=triton or cuda_tile")
        if _env_bool("LYNN_MOE_TRITON_PREPARED", False):
            if os.environ.get("LYNN_NATIVE_GATEUP_BACKEND", "triton") != "triton_fast_decode":
                raise RuntimeError("LYNN_MOE_TRITON_PREPARED requires LYNN_NATIVE_GATEUP_BACKEND=triton_fast_decode")
            if os.environ.get("LYNN_NATIVE_DOWN_BACKEND", "triton") != "triton":
                raise RuntimeError("LYNN_MOE_TRITON_PREPARED requires LYNN_NATIVE_DOWN_BACKEND=triton")
            if not _env_bool("LYNN_MOE_ACTIVE_SCRATCH", False):
                raise RuntimeError("LYNN_MOE_TRITON_PREPARED requires LYNN_MOE_ACTIVE_SCRATCH=1")
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

    router_logits = _router_linear(h_flat, w)
    routing_weights, expert_indices = _router_topk(
        router_logits,
        int(cfg["num_experts_per_tok"]),
        sorted=_env_bool("LYNN_ROUTER_TOPK_SORTED", False),
        scratch_owner=w,
    )
    routing_weights = _router_softmax(routing_weights, scratch_owner=w)
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
    w4a8_mode = _w4a8_fake_quant_mode()
    if w4a8_mode in {"gateup", "full"}:
        hidden = _fake_quant_fp8_activation(hidden)

    if os.environ.get("LYNN_MOE_PROFILE_SKIP_ACTIVE", "0") == "1":
        moe_out = torch.zeros_like(h_flat)
    else:
        backend = os.environ.get("LYNN_NATIVE_ACTIVE_MOE_BACKEND", "triton")
        down_backend = os.environ.get("LYNN_NATIVE_DOWN_BACKEND", "triton")
        if backend == "packed_pretransposed_graphsafe_v31":
            moe_out = _active_moe_packed_pretransposed_graphsafe_v31(hidden, expert_ids, routing_weights, w).reshape_as(h_flat)
        elif backend == "packed_pretransposed_graphsafe_v32_ordered":
            moe_out = _active_moe_packed_pretransposed_graphsafe_v32_ordered(hidden, expert_ids, routing_weights, w).reshape_as(h_flat)
        elif backend == "grouped_per16_nonatomic_out" and _layer_selected_for_native_cuda(cfg):
            moe_out = _active_moe_native_grouped_per16_nonatomic_out(hidden, expert_ids, routing_weights, w).reshape_as(h_flat)
        elif backend == "grouped_per16_nonatomic" and _layer_selected_for_native_cuda(cfg):
            moe_out = _active_moe_native_grouped_per16_nonatomic(hidden, expert_ids, routing_weights, w).reshape_as(h_flat)
        elif backend == "grouped_per16_fused" and _layer_selected_for_native_cuda(cfg):
            moe_out = _active_moe_native_grouped_per16_fused(hidden, expert_ids, routing_weights, w).reshape_as(h_flat)
        elif backend == "grouped_per16" and _layer_selected_for_native_cuda(cfg):
            moe_out = _active_moe_native_grouped_per16(hidden, expert_ids, routing_weights, w).reshape_as(h_flat)
        elif backend == "cuda_scalar_contract" and _layer_selected_for_native_cuda(cfg):
            moe_out = _active_moe_native_cuda_scalar_contract(hidden, expert_ids, routing_weights, w).reshape_as(h_flat)
        elif backend == "cuda_scalar" and _layer_selected_for_native_cuda(cfg):
            moe_out = _active_moe_native_cuda_scalar(hidden, expert_ids, routing_weights, w).reshape_as(h_flat)
        elif backend in {
            "triton",
            "cuda_scalar",
            "cuda_scalar_contract",
            "grouped_per16",
            "grouped_per16_fused",
            "grouped_per16_nonatomic",
            "grouped_per16_nonatomic_out",
        }:
            gateup_backend = os.environ.get("LYNN_NATIVE_GATEUP_BACKEND", "triton")
            if gateup_backend == "split16_fp4" and _layer_selected_for_native_cuda(cfg):
                inter = _gate_up_native_split16_fp4(hidden, expert_ids, w)
            elif gateup_backend == "cuda_tile_inter" and _layer_selected_for_native_cuda(cfg):
                inter = _gate_up_native_cuda_tile_inter(hidden, expert_ids, w)
            elif gateup_backend == "triton_fast_decode":
                inter_scratch = (
                    w.get("mlp.experts._active_inter_scratch")
                    if _env_bool("LYNN_MOE_ACTIVE_SCRATCH", False)
                    else None
                )
                gateup_fn = (
                    nvfp4_grouped_gate_up_silu_fast_decode_effective_scale
                    if _use_moe_effective_scale(w)
                    else nvfp4_grouped_gate_up_silu_fast_decode
                )
                inter = gateup_fn(
                    hidden,
                    expert_ids,
                    w["mlp.experts._gate_up_packed"],
                    w["mlp.experts._gate_up_effective_scale"]
                    if _use_moe_effective_scale(w)
                    else w["mlp.experts._gate_up_scale"],
                    w["mlp.experts._gate_up_global_scale"],
                    block_inter=_env_int("LYNN_MOE_GATE_BLOCK_INTER", 8),
                    block_hidden=_env_int("LYNN_MOE_GATE_BLOCK_HIDDEN", 256),
                    num_warps=_env_int("LYNN_MOE_GATE_NUM_WARPS", 4),
                    out=inter_scratch,
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
            elif gateup_backend in {"cuda_tile_inter", "split16_fp4"}:
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
                    "LYNN_NATIVE_GATEUP_BACKEND must be 'triton', 'triton_fast_decode', 'cuda_tile_inter', "
                    "or 'split16_fp4', got "
                    f"{gateup_backend!r}"
                )
            if w4a8_mode == "full":
                inter = _fake_quant_fp8_activation(inter)
            if down_backend == "cuda_tile" and _layer_selected_for_native_cuda(cfg):
                moe_out = _down_weighted_sum_native_cuda_tile(inter, expert_ids, routing_weights, w).reshape_as(h_flat)
            elif down_backend == "triton":
                out_scratch = (
                    w.get("mlp.experts._active_out_scratch")
                    if _env_bool("LYNN_MOE_ACTIVE_SCRATCH", False)
                    else None
                )
                down_fn = (
                    nvfp4_grouped_down_weighted_sum_effective_scale
                    if _use_moe_effective_scale(w)
                    else nvfp4_grouped_down_weighted_sum
                )
                moe_out = down_fn(
                    inter,
                    expert_ids,
                    routing_weights,
                    w["mlp.experts._down_packed"],
                    w["mlp.experts._down_effective_scale"]
                    if _use_moe_effective_scale(w)
                    else w["mlp.experts._down_scale"],
                    w["mlp.experts._down_global_scale"],
                    block_hidden=_env_int("LYNN_MOE_DOWN_BLOCK_HIDDEN", 8),
                    block_inter=_env_int("LYNN_MOE_DOWN_BLOCK_INTER", 512),
                    num_warps=_env_int("LYNN_MOE_DOWN_NUM_WARPS", 8),
                    out=out_scratch,
                ).reshape_as(h_flat)
            else:
                raise ValueError("LYNN_NATIVE_DOWN_BACKEND must be 'triton' or 'cuda_tile', got " f"{down_backend!r}")
        else:
            raise ValueError(
                "LYNN_NATIVE_ACTIVE_MOE_BACKEND must be 'triton', 'cuda_scalar', "
                "'cuda_scalar_contract', 'grouped_per16', 'grouped_per16_fused', "
                "'grouped_per16_nonatomic', 'grouped_per16_nonatomic_out', "
                "'packed_pretransposed_graphsafe_v31', "
                "'packed_pretransposed_graphsafe_v32_ordered', "
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
        moe_out = _finalize_shared_expert_output(h_flat, moe_out, shared, w)

    return moe_out.to(h.dtype).reshape_as(h)
