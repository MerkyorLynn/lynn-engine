"""Packed NVFP4 MoE kernels for Lynn variable-expert decode."""
from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:  # pragma: no cover
    triton = None
    tl = None
    HAS_TRITON = False


HIDDEN_SIZE = 2048
INTERMEDIATE_SIZE = 512


def _require_triton() -> None:
    if not HAS_TRITON:
        raise RuntimeError("Triton is required for packed NVFP4 MoE kernels")


if HAS_TRITON:

    @triton.jit
    def _e2m1_from_nibble(nibble):
        mag = nibble & 0x07
        sign = (nibble & 0x08) != 0
        val = tl.where(
            mag == 0,
            0.0,
            tl.where(
                mag == 1,
                0.5,
                tl.where(
                    mag == 2,
                    1.0,
                    tl.where(
                        mag == 3,
                        1.5,
                        tl.where(mag == 4, 2.0, tl.where(mag == 5, 3.0, tl.where(mag == 6, 4.0, 6.0))),
                    ),
                ),
            ),
        )
        return tl.where(sign, -val, val)

    @triton.jit
    def _grouped_gate_up_silu_kernel(
        x_ptr,
        expert_ids_ptr,
        gate_up_packed_ptr,
        gate_up_scale_ptr,
        global_scale_ptr,
        inter_ptr,
        PACKED_STRIDE_E: tl.constexpr,
        PACKED_STRIDE_M: tl.constexpr,
        PACKED_STRIDE_N: tl.constexpr,
        SCALE_STRIDE_E: tl.constexpr,
        SCALE_STRIDE_M: tl.constexpr,
        SCALE_STRIDE_G: tl.constexpr,
        INTER_STRIDE_K: tl.constexpr,
        INTER_STRIDE_I: tl.constexpr,
        HIDDEN: tl.constexpr,
        INTERMEDIATE: tl.constexpr,
        BLOCK_INTER: tl.constexpr,
        BLOCK_HIDDEN: tl.constexpr,
    ):
        slot = tl.program_id(0)
        block_i = tl.program_id(1)
        expert = tl.load(expert_ids_ptr + slot)
        inter_offsets = block_i * BLOCK_INTER + tl.arange(0, BLOCK_INTER)
        inter_mask = inter_offsets < INTERMEDIATE
        h_offsets = tl.arange(0, BLOCK_HIDDEN)
        global_scale = tl.load(global_scale_ptr).to(tl.float32)

        gate_acc = tl.zeros((BLOCK_INTER,), dtype=tl.float32)
        up_acc = tl.zeros((BLOCK_INTER,), dtype=tl.float32)

        for h0 in range(0, HIDDEN, BLOCK_HIDDEN):
            cols = h0 + h_offsets
            col_mask = cols < HIDDEN
            packed_cols = cols // 2
            scale_cols = cols // 16
            x = tl.load(x_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)

            gate_rows = inter_offsets
            up_rows = INTERMEDIATE + inter_offsets
            gate_packed_offsets = (
                expert * PACKED_STRIDE_E
                + gate_rows[:, None] * PACKED_STRIDE_M
                + packed_cols[None, :] * PACKED_STRIDE_N
            )
            up_packed_offsets = (
                expert * PACKED_STRIDE_E
                + up_rows[:, None] * PACKED_STRIDE_M
                + packed_cols[None, :] * PACKED_STRIDE_N
            )
            gate_scale_offsets = (
                expert * SCALE_STRIDE_E
                + gate_rows[:, None] * SCALE_STRIDE_M
                + scale_cols[None, :] * SCALE_STRIDE_G
            )
            up_scale_offsets = (
                expert * SCALE_STRIDE_E
                + up_rows[:, None] * SCALE_STRIDE_M
                + scale_cols[None, :] * SCALE_STRIDE_G
            )

            gate_packed = tl.load(
                gate_up_packed_ptr + gate_packed_offsets,
                mask=inter_mask[:, None] & col_mask[None, :],
                other=0,
            )
            up_packed = tl.load(
                gate_up_packed_ptr + up_packed_offsets,
                mask=inter_mask[:, None] & col_mask[None, :],
                other=0,
            )
            gate_nibble = tl.where((cols[None, :] & 1) == 0, gate_packed & 0x0F, (gate_packed >> 4) & 0x0F)
            up_nibble = tl.where((cols[None, :] & 1) == 0, up_packed & 0x0F, (up_packed >> 4) & 0x0F)
            gate_w = _e2m1_from_nibble(gate_nibble)
            up_w = _e2m1_from_nibble(up_nibble)
            gate_scale = tl.load(
                gate_up_scale_ptr + gate_scale_offsets,
                mask=inter_mask[:, None] & col_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            up_scale = tl.load(
                gate_up_scale_ptr + up_scale_offsets,
                mask=inter_mask[:, None] & col_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            gate_acc += tl.sum(gate_w * (gate_scale / global_scale) * x[None, :], axis=1)
            up_acc += tl.sum(up_w * (up_scale / global_scale) * x[None, :], axis=1)

        gate_silu = gate_acc * tl.sigmoid(gate_acc)
        inter = gate_silu * up_acc
        tl.store(inter_ptr + slot * INTER_STRIDE_K + inter_offsets * INTER_STRIDE_I, inter.to(tl.bfloat16), mask=inter_mask)

    @triton.jit
    def _grouped_down_weighted_sum_kernel(
        inter_ptr,
        expert_ids_ptr,
        routing_weights_ptr,
        down_packed_ptr,
        down_scale_ptr,
        global_scale_ptr,
        out_ptr,
        PACKED_STRIDE_E: tl.constexpr,
        PACKED_STRIDE_M: tl.constexpr,
        PACKED_STRIDE_N: tl.constexpr,
        SCALE_STRIDE_E: tl.constexpr,
        SCALE_STRIDE_M: tl.constexpr,
        SCALE_STRIDE_G: tl.constexpr,
        INTER_STRIDE_K: tl.constexpr,
        INTER_STRIDE_I: tl.constexpr,
        TOP_K: tl.constexpr,
        HIDDEN: tl.constexpr,
        INTERMEDIATE: tl.constexpr,
        BLOCK_HIDDEN: tl.constexpr,
        BLOCK_INTER: tl.constexpr,
    ):
        hidden_block = tl.program_id(0)
        rows = hidden_block * BLOCK_HIDDEN + tl.arange(0, BLOCK_HIDDEN)
        row_mask = rows < HIDDEN
        inter_offsets = tl.arange(0, BLOCK_INTER)
        global_scale = tl.load(global_scale_ptr).to(tl.float32)
        acc = tl.zeros((BLOCK_HIDDEN,), dtype=tl.float32)

        for slot in range(0, TOP_K):
            expert = tl.load(expert_ids_ptr + slot)
            route = tl.load(routing_weights_ptr + slot).to(tl.float32)
            slot_acc = tl.zeros((BLOCK_HIDDEN,), dtype=tl.float32)
            for i0 in range(0, INTERMEDIATE, BLOCK_INTER):
                cols = i0 + inter_offsets
                col_mask = cols < INTERMEDIATE
                packed_cols = cols // 2
                scale_cols = cols // 16
                inter = tl.load(
                    inter_ptr + slot * INTER_STRIDE_K + cols * INTER_STRIDE_I,
                    mask=col_mask,
                    other=0.0,
                ).to(tl.float32)
                packed_offsets = (
                    expert * PACKED_STRIDE_E
                    + rows[:, None] * PACKED_STRIDE_M
                    + packed_cols[None, :] * PACKED_STRIDE_N
                )
                scale_offsets = (
                    expert * SCALE_STRIDE_E
                    + rows[:, None] * SCALE_STRIDE_M
                    + scale_cols[None, :] * SCALE_STRIDE_G
                )
                packed = tl.load(
                    down_packed_ptr + packed_offsets,
                    mask=row_mask[:, None] & col_mask[None, :],
                    other=0,
                )
                nibble = tl.where((cols[None, :] & 1) == 0, packed & 0x0F, (packed >> 4) & 0x0F)
                w = _e2m1_from_nibble(nibble)
                scale = tl.load(
                    down_scale_ptr + scale_offsets,
                    mask=row_mask[:, None] & col_mask[None, :],
                    other=0.0,
                ).to(tl.float32)
                slot_acc += tl.sum(w * (scale / global_scale) * inter[None, :], axis=1)
            acc += route * slot_acc

        tl.store(out_ptr + rows, acc.to(tl.bfloat16), mask=row_mask)


def nvfp4_grouped_gate_up_silu(
    x: torch.Tensor,
    expert_ids: torch.Tensor,
    gate_up_packed: torch.Tensor,
    gate_up_scale: torch.Tensor,
    gate_up_global_scale: torch.Tensor,
    *,
    block_inter: int = 64,
    block_hidden: int = 64,
    num_warps: int = 4,
) -> torch.Tensor:
    """Compute top-k `silu(gate(x))*up(x)` from grouped packed NVFP4 weights."""
    _require_triton()
    if x.ndim != 1 or x.numel() != HIDDEN_SIZE:
        raise ValueError(f"x must be [2048], got {tuple(x.shape)}")
    if gate_up_packed.ndim != 3 or gate_up_scale.ndim != 3:
        raise ValueError(
            f"expected grouped 3D tensors, got packed={tuple(gate_up_packed.shape)} scale={tuple(gate_up_scale.shape)}"
        )
    expert_ids = expert_ids.to(device=x.device, dtype=torch.int32).contiguous()
    inter = torch.empty((expert_ids.numel(), INTERMEDIATE_SIZE), device=x.device, dtype=torch.bfloat16)
    grid = (expert_ids.numel(), triton.cdiv(INTERMEDIATE_SIZE, block_inter))
    _grouped_gate_up_silu_kernel[grid](
        x.contiguous(),
        expert_ids,
        gate_up_packed.contiguous(),
        gate_up_scale.contiguous(),
        gate_up_global_scale.to(device=x.device).contiguous(),
        inter,
        gate_up_packed.stride(0),
        gate_up_packed.stride(1),
        gate_up_packed.stride(2),
        gate_up_scale.stride(0),
        gate_up_scale.stride(1),
        gate_up_scale.stride(2),
        inter.stride(0),
        inter.stride(1),
        HIDDEN=HIDDEN_SIZE,
        INTERMEDIATE=INTERMEDIATE_SIZE,
        BLOCK_INTER=block_inter,
        BLOCK_HIDDEN=block_hidden,
        num_warps=num_warps,
    )
    return inter


def nvfp4_grouped_down_weighted_sum(
    inter: torch.Tensor,
    expert_ids: torch.Tensor,
    routing_weights: torch.Tensor,
    down_packed: torch.Tensor,
    down_scale: torch.Tensor,
    down_global_scale: torch.Tensor,
    *,
    block_hidden: int = 16,
    block_inter: int = 128,
    num_warps: int = 4,
) -> torch.Tensor:
    """Compute weighted top-k down projection from grouped packed NVFP4 weights."""
    _require_triton()
    if inter.ndim != 2 or inter.shape[1] != INTERMEDIATE_SIZE:
        raise ValueError(f"inter must be [top_k, 512], got {tuple(inter.shape)}")
    if down_packed.ndim != 3 or down_scale.ndim != 3:
        raise ValueError(
            f"expected grouped 3D tensors, got packed={tuple(down_packed.shape)} scale={tuple(down_scale.shape)}"
        )
    expert_ids = expert_ids.to(device=inter.device, dtype=torch.int32).contiguous()
    routing_weights = routing_weights.to(device=inter.device, dtype=torch.float32).contiguous()
    if expert_ids.numel() != inter.shape[0] or routing_weights.numel() != inter.shape[0]:
        raise ValueError("expert_ids/routing_weights must match inter top_k")
    out = torch.empty((HIDDEN_SIZE,), device=inter.device, dtype=torch.bfloat16)
    grid = (triton.cdiv(HIDDEN_SIZE, block_hidden),)
    _grouped_down_weighted_sum_kernel[grid](
        inter.contiguous(),
        expert_ids,
        routing_weights,
        down_packed.contiguous(),
        down_scale.contiguous(),
        down_global_scale.to(device=inter.device).contiguous(),
        out,
        down_packed.stride(0),
        down_packed.stride(1),
        down_packed.stride(2),
        down_scale.stride(0),
        down_scale.stride(1),
        down_scale.stride(2),
        inter.stride(0),
        inter.stride(1),
        TOP_K=inter.shape[0],
        HIDDEN=HIDDEN_SIZE,
        INTERMEDIATE=INTERMEDIATE_SIZE,
        BLOCK_HIDDEN=block_hidden,
        BLOCK_INTER=block_inter,
        num_warps=num_warps,
    )
    return out


__all__ = ["HAS_TRITON", "nvfp4_grouped_down_weighted_sum", "nvfp4_grouped_gate_up_silu"]
