"""
Lynn Engine · Phase 3.2.3 · Triton-fused MoE expert FFN.

Two-kernel design (no redundant work, single intermediate buffer reuse):

  Kernel A `gate_up_silu_kernel`
    Grid (K, INTERMEDIATE / BLOCK_INTER)
    Each program computes BLOCK_INTER values of inter[k] for one slot k:
        inter[k, i_block] = silu(gate_w[e_k, i_block, :] @ h)
                          * (up_w[e_k, i_block, :] @ h)
    Outputs to a [K, INTERMEDIATE] BF16 buffer.

  Kernel B `down_weighted_sum_kernel`
    Grid (HIDDEN / BLOCK_HIDDEN,)
    Each program computes BLOCK_HIDDEN output values:
        out[h_block] = sum over k of routing_w[k] *
                       (down_w[e_k, h_block, :] @ inter[k, :])

Total memory traffic per layer ~ 40 MB. At Spark 270 GB/s effective
~ 150 us per layer × 40 layers = 6 ms — vs current 200 ms decode path.
Target: 20-30 t/s on Spark single-stream.

Status (2026-05-09): kernels written and Python-side wrapper done, but
NOT YET VALIDATED on GPU (DGX SSH down until 2026-05-11).
Correctness reference: moe_forward_decode_indexed_bmm in this same file.
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
      mlp.experts._gate_stacked: [E, intermediate, hidden]   BF16  contiguous
      mlp.experts._up_stacked:   [E, intermediate, hidden]   BF16  contiguous
      mlp.experts._down_stacked: [E, hidden, intermediate]   BF16  contiguous

    Same total memory as keeping 256 per-expert tensors, but contiguous
    layout enables direct indexed pointer arithmetic in Triton kernels
    without per-call torch.stack copies.
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
# (correctness baseline for the Triton kernels below)
# ----------------------------------------------------------------------------

def moe_forward_decode_indexed_bmm(h, w, cfg):
    """MoE forward using pre-stacked grouped weights + indexed bmm.

    Requires `stack_expert_weights(w)` to have been called once at load.
    """
    B, T, D = h.shape
    K = cfg["num_experts_per_tok"]
    h_flat = h.view(B * T, D)

    router_logits = F.linear(h_flat, w["mlp.gate.weight"])
    routing_weights, expert_indices = torch.topk(router_logits, K, dim=-1)
    routing_weights = F.softmax(routing_weights, dim=-1, dtype=torch.float32).to(h.dtype)

    if h_flat.shape[0] != 1:
        raise NotImplementedError("indexed_bmm currently single-token only")

    expert_ids = expert_indices[0]
    weights_per_slot = routing_weights[0]

    gate_w_active = w["mlp.experts._gate_stacked"][expert_ids]
    up_w_active = w["mlp.experts._up_stacked"][expert_ids]
    down_w_active = w["mlp.experts._down_stacked"][expert_ids]

    h_b = h_flat.unsqueeze(0).expand(K, -1, -1)
    gate_out = torch.bmm(h_b, gate_w_active.transpose(-1, -2))
    up_out = torch.bmm(h_b, up_w_active.transpose(-1, -2))
    inter = F.silu(gate_out) * up_out
    ffn_out = torch.bmm(inter, down_w_active.transpose(-1, -2))

    moe_out = (ffn_out.squeeze(1) * weights_per_slot.unsqueeze(-1)).sum(dim=0, keepdim=True)

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
# Triton kernels
# ----------------------------------------------------------------------------

if HAS_TRITON:
    @triton.jit
    def gate_up_silu_kernel(
        h_ptr,
        expert_ids_ptr,
        gate_w_ptr,
        up_w_ptr,
        inter_ptr,
        # Strides (in elements)
        gate_stride_e, gate_stride_i, gate_stride_h,
        up_stride_e, up_stride_i, up_stride_h,
        inter_stride_k, inter_stride_i,
        # Constants
        HIDDEN: tl.constexpr,
        INTERMEDIATE: tl.constexpr,
        BLOCK_INTER: tl.constexpr,
        BLOCK_HIDDEN: tl.constexpr,
    ):
        """Compute inter[k, i_block] = silu(gate_w[e_k] @ h) * (up_w[e_k] @ h).

        Grid: (K, INTERMEDIATE // BLOCK_INTER)
        """
        slot = tl.program_id(0)
        inter_block = tl.program_id(1)

        # Lookup expert id for this slot
        expert_id = tl.load(expert_ids_ptr + slot)

        # Intermediate-dim offsets for this block
        inter_start = inter_block * BLOCK_INTER
        inter_offsets = inter_start + tl.arange(0, BLOCK_INTER)
        inter_mask = inter_offsets < INTERMEDIATE

        # FP32 accumulators
        gate_acc = tl.zeros([BLOCK_INTER], dtype=tl.float32)
        up_acc = tl.zeros([BLOCK_INTER], dtype=tl.float32)

        # Tile over the hidden dim
        for h_start in range(0, HIDDEN, BLOCK_HIDDEN):
            h_offsets = h_start + tl.arange(0, BLOCK_HIDDEN)
            h_mask = h_offsets < HIDDEN

            # Load h tile [BLOCK_HIDDEN]
            h_tile = tl.load(h_ptr + h_offsets, mask=h_mask, other=0.0)

            # Compute pointers to gate_w[expert_id, inter_offsets, h_offsets]
            # Using strides (general, in case tensor isn't tightly packed)
            gate_ptrs = (gate_w_ptr
                         + expert_id * gate_stride_e
                         + inter_offsets[:, None] * gate_stride_i
                         + h_offsets[None, :] * gate_stride_h)
            gate_tile = tl.load(
                gate_ptrs,
                mask=inter_mask[:, None] & h_mask[None, :],
                other=0.0,
            )

            up_ptrs = (up_w_ptr
                       + expert_id * up_stride_e
                       + inter_offsets[:, None] * up_stride_i
                       + h_offsets[None, :] * up_stride_h)
            up_tile = tl.load(
                up_ptrs,
                mask=inter_mask[:, None] & h_mask[None, :],
                other=0.0,
            )

            # Accumulate matmul (BLOCK_INTER, BLOCK_HIDDEN) × (BLOCK_HIDDEN,) → (BLOCK_INTER,)
            gate_acc += tl.sum(gate_tile.to(tl.float32) * h_tile[None, :].to(tl.float32),
                               axis=1)
            up_acc += tl.sum(up_tile.to(tl.float32) * h_tile[None, :].to(tl.float32),
                             axis=1)

        # silu(gate) * up
        # tl.sigmoid is supported; silu(x) = x * sigmoid(x)
        gate_silu = gate_acc * tl.sigmoid(gate_acc)
        inter_val = gate_silu * up_acc

        # Store inter[slot, inter_offsets] in BF16
        inter_ptrs = inter_ptr + slot * inter_stride_k + inter_offsets * inter_stride_i
        tl.store(inter_ptrs, inter_val.to(tl.bfloat16), mask=inter_mask)


    @triton.jit
    def down_weighted_sum_kernel(
        inter_ptr,
        expert_ids_ptr,
        routing_weights_ptr,
        down_w_ptr,
        out_ptr,
        # Strides
        inter_stride_k, inter_stride_i,
        down_stride_e, down_stride_h, down_stride_i,
        # Constants
        K: tl.constexpr,
        HIDDEN: tl.constexpr,
        INTERMEDIATE: tl.constexpr,
        BLOCK_HIDDEN: tl.constexpr,
        BLOCK_INTER: tl.constexpr,
    ):
        """Compute out[h_block] = sum over k of w_k * (down_w[e_k, h_block, :] @ inter[k, :]).

        Grid: (HIDDEN // BLOCK_HIDDEN,)
        """
        out_block = tl.program_id(0)
        out_start = out_block * BLOCK_HIDDEN
        out_offsets = out_start + tl.arange(0, BLOCK_HIDDEN)
        out_mask = out_offsets < HIDDEN

        # FP32 output accumulator
        out_acc = tl.zeros([BLOCK_HIDDEN], dtype=tl.float32)

        # Loop over the K slots
        for k in tl.static_range(K):
            expert_id = tl.load(expert_ids_ptr + k)
            weight = tl.load(routing_weights_ptr + k).to(tl.float32)

            # Per-slot contribution accumulator
            slot_contrib = tl.zeros([BLOCK_HIDDEN], dtype=tl.float32)

            # Tile over intermediate dim
            for inter_start in range(0, INTERMEDIATE, BLOCK_INTER):
                inter_offsets = inter_start + tl.arange(0, BLOCK_INTER)
                inter_mask = inter_offsets < INTERMEDIATE

                # Load inter[k, inter_offsets] [BLOCK_INTER]
                inter_ptrs = inter_ptr + k * inter_stride_k + inter_offsets * inter_stride_i
                inter_tile = tl.load(inter_ptrs, mask=inter_mask, other=0.0)

                # Load down_w[expert_id, out_offsets, inter_offsets] [BLOCK_HIDDEN, BLOCK_INTER]
                down_ptrs = (down_w_ptr
                             + expert_id * down_stride_e
                             + out_offsets[:, None] * down_stride_h
                             + inter_offsets[None, :] * down_stride_i)
                down_tile = tl.load(
                    down_ptrs,
                    mask=out_mask[:, None] & inter_mask[None, :],
                    other=0.0,
                )

                # Matmul (BLOCK_HIDDEN, BLOCK_INTER) × (BLOCK_INTER,) → (BLOCK_HIDDEN,)
                slot_contrib += tl.sum(down_tile.to(tl.float32) * inter_tile[None, :].to(tl.float32),
                                       axis=1)

            out_acc += weight * slot_contrib

        # Write output in BF16
        tl.store(out_ptr + out_offsets, out_acc.to(tl.bfloat16), mask=out_mask)


def moe_forward_decode_triton(h, w, cfg):
    """Triton-fused MoE forward — wraps gate_up_silu + down_weighted_sum kernels.

    Requires `stack_expert_weights(w)` called at load time.

    Falls back to indexed_bmm if Triton not available OR for prefill (T>1).
    """
    if not HAS_TRITON:
        return moe_forward_decode_indexed_bmm(h, w, cfg)

    B, T, D = h.shape
    if B * T != 1:
        # Prefill or multi-batch — punt to bmm path
        return moe_forward_decode_indexed_bmm(h, w, cfg)

    K = cfg["num_experts_per_tok"]
    INTERMEDIATE = INTERMEDIATE_SIZE   # 512
    HIDDEN = HIDDEN_SIZE               # 2048

    h_flat = h.view(D)

    # Router (still in PyTorch — small + uses cuBLAS)
    router_logits = F.linear(h_flat.unsqueeze(0), w["mlp.gate.weight"])
    routing_weights, expert_indices = torch.topk(router_logits, K, dim=-1)
    routing_weights = F.softmax(routing_weights, dim=-1, dtype=torch.float32).to(h.dtype)

    expert_ids = expert_indices.view(K).to(torch.int32).contiguous()
    routing_w = routing_weights.view(K).contiguous()

    gate_w = w["mlp.experts._gate_stacked"]
    up_w = w["mlp.experts._up_stacked"]
    down_w = w["mlp.experts._down_stacked"]

    # Allocate inter buffer
    inter = torch.empty((K, INTERMEDIATE), device=h.device, dtype=torch.bfloat16)

    # Kernel A: gate + up + silu*up
    BLOCK_INTER_A = 64
    BLOCK_HIDDEN_A = 64
    grid_a = (K, INTERMEDIATE // BLOCK_INTER_A)
    gate_up_silu_kernel[grid_a](
        h_flat, expert_ids, gate_w, up_w, inter,
        gate_w.stride(0), gate_w.stride(1), gate_w.stride(2),
        up_w.stride(0), up_w.stride(1), up_w.stride(2),
        inter.stride(0), inter.stride(1),
        HIDDEN=HIDDEN,
        INTERMEDIATE=INTERMEDIATE,
        BLOCK_INTER=BLOCK_INTER_A,
        BLOCK_HIDDEN=BLOCK_HIDDEN_A,
    )

    # Allocate output
    out = torch.empty((HIDDEN,), device=h.device, dtype=torch.bfloat16)

    # Kernel B: down + weighted sum
    BLOCK_HIDDEN_B = 64
    BLOCK_INTER_B = 64
    grid_b = (HIDDEN // BLOCK_HIDDEN_B,)
    down_weighted_sum_kernel[grid_b](
        inter, expert_ids, routing_w, down_w, out,
        inter.stride(0), inter.stride(1),
        down_w.stride(0), down_w.stride(1), down_w.stride(2),
        K=K,
        HIDDEN=HIDDEN,
        INTERMEDIATE=INTERMEDIATE,
        BLOCK_HIDDEN=BLOCK_HIDDEN_B,
        BLOCK_INTER=BLOCK_INTER_B,
    )

    moe_out = out.view(1, 1, HIDDEN)

    # Shared expert (kept in PyTorch — single FFN)
    h_flat_2d = h.view(1, D)
    if "mlp.shared_expert.gate_proj.weight" in w:
        gate_s = F.linear(h_flat_2d, w["mlp.shared_expert.gate_proj.weight"])
        up_s = F.linear(h_flat_2d, w["mlp.shared_expert.up_proj.weight"])
        shared_ffn = F.linear(F.silu(gate_s) * up_s, w["mlp.shared_expert.down_proj.weight"])
        if "mlp.shared_expert_gate.weight" in w:
            shared_gate = torch.sigmoid(F.linear(h_flat_2d, w["mlp.shared_expert_gate.weight"]))
            shared_ffn = shared_ffn * shared_gate
        moe_out = moe_out + shared_ffn.view(1, 1, HIDDEN)

    return moe_out.view(B, T, D)


__all__ = [
    "stack_expert_weights",
    "moe_forward_decode_indexed_bmm",
    "moe_forward_decode_triton",
    "HAS_TRITON",
]
