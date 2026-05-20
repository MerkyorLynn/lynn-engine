"""
Lynn Engine · Phase 3.2 · MoE expert FFN optimization.

Step 1 (this file): Replace `for e in range(256): if mask.any():` Python loop
with unique-active-experts iteration. For decode T=1, drops 256 → ~8 iter
per layer.

Step 2 (future): Triton-fused expert FFN — single kernel call per layer.

Step 3 (future): CUTLASS NVFP4 grouped GEMM — exploits NVFP4 tensor cores.
"""
from __future__ import annotations

import os
import torch
import torch.nn.functional as F


def _w4a8_fake_quant_mode() -> str:
    mode = os.environ.get("LYNN_W4A8_FAKE_QUANT_ACTIVE", "off").lower()
    if mode not in {"off", "gateup", "full"}:
        raise ValueError("LYNN_W4A8_FAKE_QUANT_ACTIVE must be off, gateup, or full")
    return mode


def _fake_quant_fp8_activation(x: torch.Tensor) -> torch.Tensor:
    """Research-only FP8 activation round-trip for W4A8 decode gates."""
    fmt = os.environ.get("LYNN_W4A8_FAKE_QUANT_FORMAT", "e4m3").lower()
    granularity = os.environ.get("LYNN_W4A8_FAKE_QUANT_GRANULARITY", "per16").lower()
    fp8_dtype = torch.float8_e4m3fn if fmt == "e4m3" else torch.float8_e5m2
    x_float = x.float()

    if granularity == "tensor":
        scale = x_float.abs().amax().clamp_min(1.0e-6) / 448.0
        return (x_float / scale).to(fp8_dtype).to(torch.float32).mul(scale).to(x.dtype)
    if granularity == "row":
        scale = x_float.abs().amax(dim=-1, keepdim=True).clamp_min(1.0e-6) / 448.0
        return (x_float / scale).to(fp8_dtype).to(torch.float32).mul(scale).to(x.dtype)
    if granularity == "per16":
        if x.shape[-1] % 16 != 0:
            raise ValueError(f"W4A8 per16 fake quant requires last dim divisible by 16, got {tuple(x.shape)}")
        groups = x_float.reshape(*x_float.shape[:-1], x_float.shape[-1] // 16, 16)
        scale = groups.abs().amax(dim=-1, keepdim=True).clamp_min(1.0e-6) / 448.0
        rounded = (groups / scale).to(fp8_dtype).to(torch.float32).mul(scale)
        return rounded.reshape_as(x_float).to(x.dtype)
    raise ValueError("LYNN_W4A8_FAKE_QUANT_GRANULARITY must be tensor, row, or per16")


def _expert_ffn(x: torch.Tensor, w: dict, expert_id: int, *, w4a8_mode: str = "off") -> torch.Tensor:
    """Run one expert FFN for either legacy per-expert or fused expert layout."""
    if w4a8_mode in {"gateup", "full"}:
        x = _fake_quant_fp8_activation(x)
    if "mlp.experts.gate_up_proj" in w and "mlp.experts.down_proj" in w:
        gate_up = F.linear(x, w["mlp.experts.gate_up_proj"][expert_id])
        gate, up = gate_up.chunk(2, dim=-1)
        inter = F.silu(gate) * up
        if w4a8_mode == "full":
            inter = _fake_quant_fp8_activation(inter)
        return F.linear(inter, w["mlp.experts.down_proj"][expert_id])

    gate = F.linear(x, w[f"mlp.experts.{expert_id}.gate_proj.weight"])
    up = F.linear(x, w[f"mlp.experts.{expert_id}.up_proj.weight"])
    inter = F.silu(gate) * up
    if w4a8_mode == "full":
        inter = _fake_quant_fp8_activation(inter)
    return F.linear(inter, w[f"mlp.experts.{expert_id}.down_proj.weight"])


def moe_forward_decode_fp8(h, w, cfg):
    """Decode-time MoE forward using the V2 Spark FP8 path.

    Auto-selected by ``moe_forward_decode_optimized`` when V2 FP8 keys
    (``mlp.experts.gate_up_proj.weight_fp8`` etc.) are present in the
    weight dict. Mirrors the prefill FP8 branch in
    ``engine/full_forward.py::_moe_forward`` but iterates only the
    active experts (decode optimization).

    Uses ``fp8_moe_expert_gate_up_silu_fused`` (Triton MoE-expert
    variant kernel from codebuddy, 1.82-2.10x BF16 at N=1408) for the
    gate/up + SwiGLU and ``torch._scaled_mm`` for the down projection.
    Shared expert routed through ``fp8_gate_up_silu_fused`` (dense
    variant, autotune block dispatch via ``auto_block=True``).
    """
    from triton_kernels.spark_fp8_moe_expert_fused import fp8_moe_expert_gate_up_silu_fused
    from triton_kernels.spark_fp8_gate_up_fused import fp8_gate_up_silu_fused

    B, T, D = h.shape
    K = cfg["num_experts_per_tok"]
    h_flat = h.view(B * T, D)

    router_logits = F.linear(h_flat, w["mlp.gate.weight"])
    routing_weights, expert_indices = torch.topk(router_logits, K, dim=-1)
    routing_weights = F.softmax(routing_weights, dim=-1, dtype=torch.float32).to(h.dtype)
    active_experts = torch.unique(expert_indices).tolist()

    w_gate_up_fp8 = w["mlp.experts.gate_up_proj.weight_fp8"]                      # [E, 2I, K]
    w_gate_up_scale = w["mlp.experts.gate_up_proj.weight_fp8_scale"]              # [E, 2I]
    intermediate = w_gate_up_fp8.shape[1] // 2
    w_gate_fp8 = w_gate_up_fp8[:, :intermediate, :].contiguous()
    w_up_fp8 = w_gate_up_fp8[:, intermediate:, :].contiguous()
    w_gate_scale = w_gate_up_scale[:, :intermediate].contiguous().to(torch.float32)
    w_up_scale = w_gate_up_scale[:, intermediate:].contiguous().to(torch.float32)
    w_down_fp8 = w["mlp.experts.down_proj.weight_fp8"]                            # [E, K, I]
    w_down_scale = w["mlp.experts.down_proj.weight_fp8_scale"]                    # [E, K]

    moe_out = torch.zeros_like(h_flat)
    for e in active_experts:
        mask = (expert_indices == e)
        token_idx, slot_idx = mask.nonzero(as_tuple=True)
        x_e = h_flat[token_idx]
        inter_bf16 = fp8_moe_expert_gate_up_silu_fused(
            x_e, w_gate_fp8, w_up_fp8, w_gate_scale, w_up_scale, expert_id=e,
        )
        inter_max = inter_bf16.abs().amax(dim=-1, keepdim=True).clamp_min(1.0e-12)
        inter_scale = (inter_max / 448.0).to(torch.float32)
        inter_fp8 = (inter_bf16.to(torch.float32) / inter_scale).to(torch.float8_e4m3fn)
        ffn_e = torch._scaled_mm(
            inter_fp8,
            w_down_fp8[e].t(),
            scale_a=inter_scale,
            scale_b=w_down_scale[e].view(1, -1).to(torch.float32),
            out_dtype=torch.bfloat16,
        )
        weight_e = routing_weights[token_idx, slot_idx].unsqueeze(-1)
        moe_out.index_add_(0, token_idx, ffn_e * weight_e)

    if "mlp.shared_expert.gate_proj.weight_fp8" in w:
        shared_inter = fp8_gate_up_silu_fused(
            h_flat,
            w["mlp.shared_expert.gate_proj.weight_fp8"],
            w["mlp.shared_expert.up_proj.weight_fp8"],
            w["mlp.shared_expert.gate_proj.weight_fp8_scale"].to(torch.float32),
            w["mlp.shared_expert.up_proj.weight_fp8_scale"].to(torch.float32),
            auto_block=True,
        )
        inter_max_s = shared_inter.abs().amax(dim=-1, keepdim=True).clamp_min(1.0e-12)
        inter_scale_s = (inter_max_s / 448.0).to(torch.float32)
        inter_fp8_s = (shared_inter.to(torch.float32) / inter_scale_s).to(torch.float8_e4m3fn)
        shared_ffn = torch._scaled_mm(
            inter_fp8_s,
            w["mlp.shared_expert.down_proj.weight_fp8"].t(),
            scale_a=inter_scale_s,
            scale_b=w["mlp.shared_expert.down_proj.weight_fp8_scale"].view(1, -1).to(torch.float32),
            out_dtype=torch.bfloat16,
        )
        if "mlp.shared_expert_gate.weight" in w:
            shared_gate = torch.sigmoid(F.linear(h_flat, w["mlp.shared_expert_gate.weight"]))
            shared_ffn = shared_ffn * shared_gate
        moe_out = moe_out + shared_ffn

    return moe_out.view(B, T, D)


def moe_forward_decode_optimized(h, w, cfg):
    """MoE forward for the decode path (T=1).

    Skips the empty-mask Python loop overhead. Iterates only the up-to-K
    unique active experts.

    Auto-dispatches to ``moe_forward_decode_fp8`` if V2 Spark FP8 weight
    keys are present (Phase 2 FP8 path). Otherwise falls through to the
    BF16 / NVFP4 / W4A8-fake-quant path below.

    h: [B=1, T=1, hidden]
    Returns: [B=1, T=1, hidden]
    """
    # Phase 2 V2 FP8 decode dispatch.
    fp8_disabled = os.environ.get("LYNN_DISABLE_W4A8_FP8_PATH", "0").lower() in {"1", "true", "yes"}
    if (not fp8_disabled
            and "mlp.experts.gate_up_proj.weight_fp8" in w
            and "mlp.experts.down_proj.weight_fp8" in w):
        return moe_forward_decode_fp8(h, w, cfg)

    B, T, D = h.shape
    K = cfg["num_experts_per_tok"]

    h_flat = h.view(B * T, D)            # [N, hidden] where N = B*T

    # Router
    router_logits = F.linear(h_flat, w["mlp.gate.weight"])
    routing_weights, expert_indices = torch.topk(router_logits, K, dim=-1)
    # softmax over top-K (mathematically equivalent to softmax-all → topk → renorm)
    routing_weights = F.softmax(routing_weights, dim=-1, dtype=torch.float32).to(h.dtype)

    # Which experts actually got hit (up to K unique entries for B=T=1)
    active_experts = torch.unique(expert_indices).tolist()
    w4a8_mode = _w4a8_fake_quant_mode()

    moe_out = torch.zeros_like(h_flat)
    for e in active_experts:
        # mask: which (token, slot) positions selected this expert
        mask = (expert_indices == e)
        token_idx, slot_idx = mask.nonzero(as_tuple=True)
        x_e = h_flat[token_idx]
        ffn_e = _expert_ffn(x_e, w, e, w4a8_mode=w4a8_mode)
        weight_e = routing_weights[token_idx, slot_idx].unsqueeze(-1)
        moe_out.index_add_(0, token_idx, ffn_e * weight_e)

    # Shared expert (always active)
    if "mlp.shared_expert.gate_proj.weight" in w:
        gate_s = F.linear(h_flat, w["mlp.shared_expert.gate_proj.weight"])
        up_s = F.linear(h_flat, w["mlp.shared_expert.up_proj.weight"])
        shared_ffn = F.linear(F.silu(gate_s) * up_s, w["mlp.shared_expert.down_proj.weight"])
        if "mlp.shared_expert_gate.weight" in w:
            shared_gate = torch.sigmoid(F.linear(h_flat, w["mlp.shared_expert_gate.weight"]))
            shared_ffn = shared_ffn * shared_gate
        moe_out = moe_out + shared_ffn

    return moe_out.view(B, T, D)


def moe_forward_prefill_optimized(h, w, cfg):
    """MoE forward for the prefill path (T can be large).

    Same optimization: iterate only active experts. Token grouping uses
    the same mask pattern — for prefill with diverse tokens, more experts
    are typically hit (closer to all 256 for T >= 32).
    """
    B, T, D = h.shape
    K = cfg["num_experts_per_tok"]

    h_flat = h.view(B * T, D)
    router_logits = F.linear(h_flat, w["mlp.gate.weight"])
    routing_weights, expert_indices = torch.topk(router_logits, K, dim=-1)
    routing_weights = F.softmax(routing_weights, dim=-1, dtype=torch.float32).to(h.dtype)

    active_experts = torch.unique(expert_indices).tolist()

    moe_out = torch.zeros_like(h_flat)
    for e in active_experts:
        mask = (expert_indices == e)
        token_idx, slot_idx = mask.nonzero(as_tuple=True)
        x_e = h_flat[token_idx]
        ffn_e = _expert_ffn(x_e, w, e)
        weight_e = routing_weights[token_idx, slot_idx].unsqueeze(-1)
        moe_out.index_add_(0, token_idx, ffn_e * weight_e)

    if "mlp.shared_expert.gate_proj.weight" in w:
        gate_s = F.linear(h_flat, w["mlp.shared_expert.gate_proj.weight"])
        up_s = F.linear(h_flat, w["mlp.shared_expert.up_proj.weight"])
        shared_ffn = F.linear(F.silu(gate_s) * up_s, w["mlp.shared_expert.down_proj.weight"])
        if "mlp.shared_expert_gate.weight" in w:
            shared_gate = torch.sigmoid(F.linear(h_flat, w["mlp.shared_expert_gate.weight"]))
            shared_ffn = shared_ffn * shared_gate
        moe_out = moe_out + shared_ffn

    return moe_out.view(B, T, D)


def moe_forward_decode_bmm(h, w, cfg):
    """MoE forward using batched-matmul (bmm) on stacked active expert weights.

    Phase 3.2.2: Single batched-GEMM call per stage (gate / up / down)
    instead of K=8 sequential F.linear calls. Uses cuBLAS via torch.bmm.

    Trade-off: requires 3 torch.stack calls per layer (gathering 8 expert
    weights into a [K, intermediate, hidden] tensor). At ~16 MB stack ×
    3 stacks × 40 layers = 2 GB memory traffic per token. Stacking cost
    on Spark (~270 GB/s effective) ≈ 7 ms per token total. Net win:
    cuBLAS bmm should be ~5-10x faster than 8 sequential matmul kernels,
    overcoming the stacking overhead.

    Estimated gain over moe_forward_decode_optimized (Python loop):
      ~30 ms / token (40 layers × ~0.7 ms saved/layer)

    h: [B=1, T=1, hidden]
    Returns: [B=1, T=1, hidden]
    """
    B, T, D = h.shape
    K = cfg["num_experts_per_tok"]

    h_flat = h.view(B * T, D)            # [N, hidden]
    router_logits = F.linear(h_flat, w["mlp.gate.weight"])
    routing_weights, expert_indices = torch.topk(router_logits, K, dim=-1)
    routing_weights = F.softmax(routing_weights, dim=-1, dtype=torch.float32).to(h.dtype)

    # For T=1, expert_indices[0] is a vector of K expert ids (may have duplicates,
    # but typically all K unique). For T>1 we'd need to handle batching.
    if h_flat.shape[0] != 1:
        # Fallback to optimized loop for multi-token (prefill); bmm path needs work.
        return moe_forward_decode_optimized(h, w, cfg)

    expert_ids = expert_indices[0].tolist()   # length K, may have dups
    weights_per_slot = routing_weights[0]     # [K]
    w4a8_mode = _w4a8_fake_quant_mode()

    # Stack weights for selected experts. CRITICAL: maintain slot order so
    # we can apply per-slot routing_weights directly.
    if "mlp.experts.gate_up_proj" in w and "mlp.experts.down_proj" in w:
        gate_up_stack = torch.stack([w["mlp.experts.gate_up_proj"][e] for e in expert_ids])
        gate_stack, up_stack = gate_up_stack.chunk(2, dim=1)
        down_stack = torch.stack([w["mlp.experts.down_proj"][e] for e in expert_ids])
    else:
        gate_stack = torch.stack([w[f"mlp.experts.{e}.gate_proj.weight"] for e in expert_ids])
        up_stack = torch.stack([w[f"mlp.experts.{e}.up_proj.weight"] for e in expert_ids])
        down_stack = torch.stack([w[f"mlp.experts.{e}.down_proj.weight"] for e in expert_ids])
    # Each stack: [K, intermediate=512, hidden=2048] for gate/up
    #             [K, hidden=2048, intermediate=512] for down

    # bmm: h_flat [1, hidden] → [K, 1, hidden] → bmm with [K, hidden, intermediate]
    active_h = _fake_quant_fp8_activation(h_flat) if w4a8_mode in {"gateup", "full"} else h_flat
    h_broadcast = active_h.unsqueeze(0).expand(K, -1, -1)   # [K, 1, hidden]
    gate_out = torch.bmm(h_broadcast, gate_stack.transpose(-1, -2))   # [K, 1, intermediate]
    up_out = torch.bmm(h_broadcast, up_stack.transpose(-1, -2))       # [K, 1, intermediate]
    inter = F.silu(gate_out) * up_out                                 # [K, 1, intermediate]
    if w4a8_mode == "full":
        inter = _fake_quant_fp8_activation(inter)
    ffn_out = torch.bmm(inter, down_stack.transpose(-1, -2))          # [K, 1, hidden]

    # Weighted sum across slots
    ffn_out_2d = ffn_out.squeeze(1)                                   # [K, hidden]
    weighted = ffn_out_2d * weights_per_slot.unsqueeze(-1)            # [K, hidden]
    moe_out = weighted.sum(dim=0, keepdim=True)                       # [1, hidden]

    # Shared expert (always active)
    if "mlp.shared_expert.gate_proj.weight" in w:
        gate_s = F.linear(h_flat, w["mlp.shared_expert.gate_proj.weight"])
        up_s = F.linear(h_flat, w["mlp.shared_expert.up_proj.weight"])
        shared_ffn = F.linear(F.silu(gate_s) * up_s, w["mlp.shared_expert.down_proj.weight"])
        if "mlp.shared_expert_gate.weight" in w:
            shared_gate = torch.sigmoid(F.linear(h_flat, w["mlp.shared_expert_gate.weight"]))
            shared_ffn = shared_ffn * shared_gate
        moe_out = moe_out + shared_ffn

    return moe_out.view(B, T, D)


def moe_forward_decode_slot_sorted(h, w, cfg):
    """One-token MoE path that preserves expert-id accumulation order.

    This sits between `decode_optimized` and `decode_bmm`: it avoids the
    mask/nonzero/index_add overhead for B=T=1, while still accumulating experts
    in ascending expert-id order like the generic full-forward implementation.
    """
    B, T, D = h.shape
    K = cfg["num_experts_per_tok"]
    h_flat = h.view(B * T, D)
    if h_flat.shape[0] != 1:
        return moe_forward_decode_optimized(h, w, cfg)

    router_logits = F.linear(h_flat, w["mlp.gate.weight"])
    routing_weights, expert_indices = torch.topk(router_logits, K, dim=-1)
    routing_weights = F.softmax(routing_weights, dim=-1, dtype=torch.float32).to(h.dtype)
    expert_ids = [int(x) for x in expert_indices[0].tolist()]
    slot_order = sorted(range(K), key=lambda idx: (expert_ids[idx], idx))
    w4a8_mode = _w4a8_fake_quant_mode()

    moe_out = torch.zeros_like(h_flat)
    for slot_idx in slot_order:
        expert_id = expert_ids[slot_idx]
        ffn_e = _expert_ffn(h_flat, w, expert_id, w4a8_mode=w4a8_mode)
        moe_out = moe_out + ffn_e * routing_weights[0, slot_idx].view(1, 1)

    if "mlp.shared_expert.gate_proj.weight" in w:
        gate_s = F.linear(h_flat, w["mlp.shared_expert.gate_proj.weight"])
        up_s = F.linear(h_flat, w["mlp.shared_expert.up_proj.weight"])
        shared_ffn = F.linear(F.silu(gate_s) * up_s, w["mlp.shared_expert.down_proj.weight"])
        if "mlp.shared_expert_gate.weight" in w:
            shared_gate = torch.sigmoid(F.linear(h_flat, w["mlp.shared_expert_gate.weight"]))
            shared_ffn = shared_ffn * shared_gate
        moe_out = moe_out + shared_ffn

    return moe_out.view(B, T, D)


__all__ = [
    "moe_forward_decode_optimized",      # Phase 3.2.1 — active-experts only
    "moe_forward_prefill_optimized",
    "moe_forward_decode_bmm",            # Phase 3.2.2 — batched matmul
    "moe_forward_decode_slot_sorted",
]
