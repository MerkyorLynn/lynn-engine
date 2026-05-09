"""
Lynn Engine · Phase 3.2.3 · Triton-fused MoE expert FFN kernel.

Target: single Triton kernel that does the FULL MoE expert FFN for one token
across K=8 active experts in a single kernel launch.

Math (per active expert e in routing slots 0..K-1):
    gate_e = silu(W_gate[e] @ h)        # [intermediate]
    up_e   = W_up[e] @ h                  # [intermediate]
    inter  = gate_e * up_e
    ffn_e  = W_down[e] @ inter            # [hidden]
    out   += routing_weights[e] * ffn_e

Aggregated across K experts → out [hidden]

Design (skeleton, NOT YET KERNEL-OPTIMIZED):
- Pre-stack gate / up / down weights at load time (via load_stacked_layer below):
    W_gate_up_stacked: [E=256, 2*intermediate, hidden]    # gate cat'd with up
    W_down_stacked:    [E=256, hidden, intermediate]
- Kernel loads h once into shared memory.
- Loops over K slots inside the kernel — for each, looks up expert_id, fetches
  weights via direct pointer arithmetic on the stacked tensor (no copy).
- Computes gate_e, up_e, fused silu*up, down product, accumulates into out.

Status: kernel structure scaffolded but the inner matmul tile loops are
TODO — will fill in when DGX SSH is restored and we can test correctness
against the bmm reference.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


# Qwen 3.6 fixed dims
HIDDEN_SIZE = 2048
INTERMEDIATE_SIZE = 512
NUM_EXPERTS = 256
TOP_K = 8


# ----------------------------------------------------------------------------
# Helper: stack per-expert weights into grouped tensors
# ----------------------------------------------------------------------------

def stack_expert_weights(layer_weights: dict, num_experts: int = 256) -> dict:
    """Convert per-expert weight dict to grouped tensors (in-place augment).

    Adds three new keys to layer_weights:
      mlp.experts._gate_stacked: [E, intermediate, hidden]   BF16
      mlp.experts._up_stacked:   [E, intermediate, hidden]
      mlp.experts._down_stacked: [E, hidden, intermediate]

    Memory cost: same as keeping 256 per-expert tensors (just different layout).
    Original per-expert keys remain (caller can choose path).
    """
    gate_list = [layer_weights[f"mlp.experts.{e}.gate_proj.weight"] for e in range(num_experts)]
    up_list = [layer_weights[f"mlp.experts.{e}.up_proj.weight"] for e in range(num_experts)]
    down_list = [layer_weights[f"mlp.experts.{e}.down_proj.weight"] for e in range(num_experts)]

    layer_weights["mlp.experts._gate_stacked"] = torch.stack(gate_list, dim=0).contiguous()
    layer_weights["mlp.experts._up_stacked"] = torch.stack(up_list, dim=0).contiguous()
    layer_weights["mlp.experts._down_stacked"] = torch.stack(down_list, dim=0).contiguous()
    return layer_weights


# ----------------------------------------------------------------------------
# Reference torch implementation using stacked weights + indexed bmm
# (Phase 3.2.2.5 — between bmm and Triton; uses pre-stacked layout to skip
# per-call torch.stack)
# ----------------------------------------------------------------------------

def moe_forward_decode_indexed_bmm(h, w, cfg):
    """MoE forward using pre-stacked grouped weights + indexed bmm.

    Requires `stack_expert_weights(w)` to have been called once at load.
    Saves ~7 ms / token (no per-call torch.stack) over Phase 3.2.2.

    h: [B=1, T=1, hidden]
    Returns: [B=1, T=1, hidden]
    """
    B, T, D = h.shape
    K = cfg["num_experts_per_tok"]
    h_flat = h.view(B * T, D)

    router_logits = F.linear(h_flat, w["mlp.gate.weight"])
    routing_weights, expert_indices = torch.topk(router_logits, K, dim=-1)
    routing_weights = F.softmax(routing_weights, dim=-1, dtype=torch.float32).to(h.dtype)

    if h_flat.shape[0] != 1:
        raise NotImplementedError("indexed_bmm currently single-token only")

    expert_ids = expert_indices[0]    # [K] long
    weights_per_slot = routing_weights[0]   # [K]

    # Index into grouped tensors — fancy indexing creates [K, ...] views
    # (PyTorch will copy under the hood, but it's a single contiguous copy
    # per matmul instead of K separate matmul kernel launches)
    gate_w_active = w["mlp.experts._gate_stacked"][expert_ids]   # [K, inter, hidden]
    up_w_active = w["mlp.experts._up_stacked"][expert_ids]       # [K, inter, hidden]
    down_w_active = w["mlp.experts._down_stacked"][expert_ids]   # [K, hidden, inter]

    h_b = h_flat.unsqueeze(0).expand(K, -1, -1)                  # [K, 1, hidden]
    gate_out = torch.bmm(h_b, gate_w_active.transpose(-1, -2))    # [K, 1, inter]
    up_out = torch.bmm(h_b, up_w_active.transpose(-1, -2))        # [K, 1, inter]
    inter = F.silu(gate_out) * up_out                            # [K, 1, inter]
    ffn_out = torch.bmm(inter, down_w_active.transpose(-1, -2))  # [K, 1, hidden]

    moe_out = (ffn_out.squeeze(1) * weights_per_slot.unsqueeze(-1)).sum(dim=0, keepdim=True)

    # Shared expert
    if "mlp.shared_expert.gate_proj.weight" in w:
        gate_s = F.linear(h_flat, w["mlp.shared_expert.gate_proj.weight"])
        up_s = F.linear(h_flat, w["mlp.shared_expert.up_proj.weight"])
        shared_ffn = F.linear(F.silu(gate_s) * up_s, w["mlp.shared_expert.down_proj.weight"])
        if "mlp.shared_expert_gate.weight" in w:
            shared_gate = torch.sigmoid(F.linear(h_flat, w["mlp.shared_expert_gate.weight"]))
            shared_ffn = shared_ffn * shared_gate
        moe_out = moe_out + shared_ffn

    return moe_out.view(B, T, D)


# ----------------------------------------------------------------------------
# Triton kernel skeleton (TODO: fill in matmul tile loops)
# ----------------------------------------------------------------------------

if HAS_TRITON:
    @triton.jit
    def _moe_expert_ffn_kernel(
        # Pointers
        h_ptr,                  # [hidden]
        out_ptr,                # [hidden]
        expert_ids_ptr,         # [K] int32
        routing_weights_ptr,    # [K] BF16
        gate_w_ptr,             # [E, intermediate, hidden]
        up_w_ptr,               # [E, intermediate, hidden]
        down_w_ptr,             # [E, hidden, intermediate]
        # Strides (in elements, not bytes)
        gate_stride_e, gate_stride_i, gate_stride_h,
        up_stride_e, up_stride_i, up_stride_h,
        down_stride_e, down_stride_h, down_stride_i,
        # Constants
        K: tl.constexpr,
        HIDDEN: tl.constexpr,
        INTERMEDIATE: tl.constexpr,
        BLOCK_HIDDEN: tl.constexpr,
        BLOCK_INTER: tl.constexpr,
    ):
        """Kernel for per-token MoE FFN across K active experts.

        Grid: (HIDDEN // BLOCK_HIDDEN,)  — one program per output hidden tile

        Each program block:
          1. Load h tile (HIDDEN size) into shared / registers
          2. For each slot k in [0..K-1]:
             a. Load expert_id k, routing weight k
             b. Compute gate_e tile: BLOCK_INTER elements at a time
                = sum over hidden of gate_w[expert_id, inter_block, :] * h
             c. Compute up_e tile similarly
             d. silu(gate_e) * up_e
             e. Accumulate into output via down_w
          3. Write accumulated output tile

        TODO: fill in the actual tile loops. Reference: triton_kernels/attention.py
        for the matmul accumulation pattern.
        """
        # --- TODO: implement ---
        # For now, this kernel is a placeholder. The bmm reference path
        # (moe_forward_decode_indexed_bmm) provides the correctness baseline.
        pass


def moe_forward_decode_triton(h, w, cfg):
    """Triton-fused MoE forward — wraps _moe_expert_ffn_kernel.

    Requires `stack_expert_weights(w)` called at load time.

    Until the inner matmul loops in _moe_expert_ffn_kernel are filled in,
    this just falls back to the indexed_bmm reference (so callers can wire
    it in early without breakage).
    """
    if not HAS_TRITON:
        return moe_forward_decode_indexed_bmm(h, w, cfg)

    # TODO: actually launch the kernel once filled in
    # For now, fall back to bmm reference (bit-equivalent target)
    return moe_forward_decode_indexed_bmm(h, w, cfg)


__all__ = [
    "stack_expert_weights",
    "moe_forward_decode_indexed_bmm",
    "moe_forward_decode_triton",
    "HAS_TRITON",
]
