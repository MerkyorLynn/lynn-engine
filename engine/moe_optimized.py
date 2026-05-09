"""
Lynn Engine · Phase 3.2 · MoE expert FFN optimization.

Step 1 (this file): Replace `for e in range(256): if mask.any():` Python loop
with unique-active-experts iteration. For decode T=1, drops 256 → ~8 iter
per layer.

Step 2 (future): Triton-fused expert FFN — single kernel call per layer.

Step 3 (future): CUTLASS NVFP4 grouped GEMM — exploits NVFP4 tensor cores.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def moe_forward_decode_optimized(h, w, cfg):
    """MoE forward for the decode path (T=1).

    Skips the empty-mask Python loop overhead. Iterates only the up-to-K
    unique active experts.

    h: [B=1, T=1, hidden]
    Returns: [B=1, T=1, hidden]
    """
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

    moe_out = torch.zeros_like(h_flat)
    for e in active_experts:
        # mask: which (token, slot) positions selected this expert
        mask = (expert_indices == e)
        token_idx, slot_idx = mask.nonzero(as_tuple=True)
        x_e = h_flat[token_idx]
        gate_e = F.linear(x_e, w[f"mlp.experts.{e}.gate_proj.weight"])
        up_e = F.linear(x_e, w[f"mlp.experts.{e}.up_proj.weight"])
        ffn_e = F.linear(F.silu(gate_e) * up_e, w[f"mlp.experts.{e}.down_proj.weight"])
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
        gate_e = F.linear(x_e, w[f"mlp.experts.{e}.gate_proj.weight"])
        up_e = F.linear(x_e, w[f"mlp.experts.{e}.up_proj.weight"])
        ffn_e = F.linear(F.silu(gate_e) * up_e, w[f"mlp.experts.{e}.down_proj.weight"])
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


__all__ = ["moe_forward_decode_optimized", "moe_forward_prefill_optimized"]
