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
    def _e2m1_from_nibble_fast(nibble):
        """Same E2M1 table with a shallower expression tree.

        This keeps the exact compressed-tensors value table:
        [0, .5, 1, 1.5, 2, 3, 4, 6].
        """
        mag = nibble & 0x07
        sign = (nibble & 0x08) != 0
        mag_f = mag.to(tl.float32)
        val = tl.where(
            mag <= 4,
            mag_f * 0.5,
            tl.where(mag == 5, 3.0, tl.where(mag == 6, 4.0, 6.0)),
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
    def _grouped_gate_up_silu_scale_hoist_kernel(
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
        group_offsets = tl.arange(0, 16)
        global_scale = tl.load(global_scale_ptr).to(tl.float32)

        gate_acc = tl.zeros((BLOCK_INTER,), dtype=tl.float32)
        up_acc = tl.zeros((BLOCK_INTER,), dtype=tl.float32)

        for h0 in tl.static_range(0, HIDDEN, BLOCK_HIDDEN):
            for hg in tl.static_range(0, BLOCK_HIDDEN, 16):
                cols = h0 + hg + group_offsets
                col_mask = cols < HIDDEN
                packed_cols = cols // 2
                scale_col = (h0 + hg) // 16
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
                gate_w = _e2m1_from_nibble_fast(gate_nibble)
                up_w = _e2m1_from_nibble_fast(up_nibble)
                gate_scale = tl.load(
                    gate_up_scale_ptr
                    + expert * SCALE_STRIDE_E
                    + gate_rows * SCALE_STRIDE_M
                    + scale_col * SCALE_STRIDE_G,
                    mask=inter_mask,
                    other=0.0,
                ).to(tl.float32)
                up_scale = tl.load(
                    gate_up_scale_ptr
                    + expert * SCALE_STRIDE_E
                    + up_rows * SCALE_STRIDE_M
                    + scale_col * SCALE_STRIDE_G,
                    mask=inter_mask,
                    other=0.0,
                ).to(tl.float32)
                gate_acc += tl.sum(gate_w * (gate_scale[:, None] / global_scale) * x[None, :], axis=1)
                up_acc += tl.sum(up_w * (up_scale[:, None] / global_scale) * x[None, :], axis=1)

        gate_silu = gate_acc * tl.sigmoid(gate_acc)
        inter = gate_silu * up_acc
        tl.store(inter_ptr + slot * INTER_STRIDE_K + inter_offsets * INTER_STRIDE_I, inter.to(tl.bfloat16), mask=inter_mask)

    @triton.jit
    def _grouped_gate_up_silu_fast_decode_kernel(
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
        SCALE_EFFECTIVE: tl.constexpr,
    ):
        slot = tl.program_id(0)
        block_i = tl.program_id(1)
        expert = tl.load(expert_ids_ptr + slot)
        inter_offsets = block_i * BLOCK_INTER + tl.arange(0, BLOCK_INTER)
        inter_mask = inter_offsets < INTERMEDIATE
        h_offsets = tl.arange(0, BLOCK_HIDDEN)
        if SCALE_EFFECTIVE:
            global_scale = 1.0
        else:
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
            gate_w = _e2m1_from_nibble_fast(gate_nibble)
            up_w = _e2m1_from_nibble_fast(up_nibble)
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
            if SCALE_EFFECTIVE:
                gate_acc += tl.sum(gate_w * gate_scale * x[None, :], axis=1)
                up_acc += tl.sum(up_w * up_scale * x[None, :], axis=1)
            else:
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
        SCALE_EFFECTIVE: tl.constexpr,
    ):
        hidden_block = tl.program_id(0)
        rows = hidden_block * BLOCK_HIDDEN + tl.arange(0, BLOCK_HIDDEN)
        row_mask = rows < HIDDEN
        inter_offsets = tl.arange(0, BLOCK_INTER)
        if SCALE_EFFECTIVE:
            global_scale = 1.0
        else:
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
                if SCALE_EFFECTIVE:
                    slot_acc += tl.sum(w * scale * inter[None, :], axis=1)
                else:
                    slot_acc += tl.sum(w * (scale / global_scale) * inter[None, :], axis=1)
            acc += route * slot_acc

        tl.store(out_ptr + rows, acc.to(tl.bfloat16), mask=row_mask)

    @triton.jit
    def _grouped_down_weighted_sum_scale_hoist_kernel(
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
        group_offsets = tl.arange(0, 16)
        global_scale = tl.load(global_scale_ptr).to(tl.float32)
        acc = tl.zeros((BLOCK_HIDDEN,), dtype=tl.float32)

        for slot in range(0, TOP_K):
            expert = tl.load(expert_ids_ptr + slot)
            route = tl.load(routing_weights_ptr + slot).to(tl.float32)
            slot_acc = tl.zeros((BLOCK_HIDDEN,), dtype=tl.float32)
            for i0 in tl.static_range(0, INTERMEDIATE, 16):
                cols = i0 + group_offsets
                col_mask = cols < INTERMEDIATE
                packed_cols = cols // 2
                scale_col = i0 // 16
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
                packed = tl.load(
                    down_packed_ptr + packed_offsets,
                    mask=row_mask[:, None] & col_mask[None, :],
                    other=0,
                )
                nibble = tl.where((cols[None, :] & 1) == 0, packed & 0x0F, (packed >> 4) & 0x0F)
                w = _e2m1_from_nibble_fast(nibble)
                scale = tl.load(
                    down_scale_ptr
                    + expert * SCALE_STRIDE_E
                    + rows * SCALE_STRIDE_M
                    + scale_col * SCALE_STRIDE_G,
                    mask=row_mask,
                    other=0.0,
                ).to(tl.float32)
                slot_acc += tl.sum(w * (scale[:, None] / global_scale) * inter[None, :], axis=1)
            acc += route * slot_acc

        tl.store(out_ptr + rows, acc.to(tl.bfloat16), mask=row_mask)

    @triton.jit
    def _grouped_gate_up_silu_merged_topk_kernel(
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
        INTER_STRIDE_I: tl.constexpr,
        HIDDEN: tl.constexpr,
        INTERMEDIATE: tl.constexpr,
        TOP_K: tl.constexpr,
        BLOCK_INTER: tl.constexpr,
        BLOCK_HIDDEN: tl.constexpr,
    ):
        block_i = tl.program_id(0)
        inter_offsets = block_i * BLOCK_INTER + tl.arange(0, BLOCK_INTER)
        inter_mask = inter_offsets < INTERMEDIATE
        h_offsets = tl.arange(0, BLOCK_HIDDEN)
        global_scale = tl.load(global_scale_ptr).to(tl.float32)

        for slot in range(0, TOP_K):
            expert = tl.load(expert_ids_ptr + slot)
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
            tl.store(
                inter_ptr + slot * INTERMEDIATE + inter_offsets * INTER_STRIDE_I,
                inter.to(tl.bfloat16),
                mask=inter_mask,
            )


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


def nvfp4_grouped_gate_up_silu_fast_decode_effective_scale(
    x: torch.Tensor,
    expert_ids: torch.Tensor,
    gate_up_packed: torch.Tensor,
    gate_up_effective_scale: torch.Tensor,
    gate_up_global_scale: torch.Tensor,
    *,
    block_inter: int = 8,
    block_hidden: int = 256,
    num_warps: int = 4,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fast gate/up path that consumes precomputed `scale / global_scale`.

    This is an opt-in MoE repack probe.  It keeps the same packed weight layout
    and output contract as `nvfp4_grouped_gate_up_silu_fast_decode`, but removes
    the per-element division from the decode kernel when the runner has attached
    effective scale tensors.
    """
    _require_triton()
    if x.ndim != 1 or x.numel() != HIDDEN_SIZE:
        raise ValueError(f"x must be [2048], got {tuple(x.shape)}")
    if gate_up_packed.ndim != 3 or gate_up_effective_scale.ndim != 3:
        raise ValueError(
            "expected grouped 3D tensors, got "
            f"packed={tuple(gate_up_packed.shape)} scale={tuple(gate_up_effective_scale.shape)}"
        )
    expert_ids = expert_ids.to(device=x.device, dtype=torch.int32).contiguous()
    if out is None:
        inter = torch.empty((expert_ids.numel(), INTERMEDIATE_SIZE), device=x.device, dtype=torch.bfloat16)
    else:
        if out.ndim != 2 or out.shape[0] < expert_ids.numel() or out.shape[1] != INTERMEDIATE_SIZE:
            raise ValueError(
                f"out must be at least [top_k, {INTERMEDIATE_SIZE}], got {tuple(out.shape)} "
                f"for top_k={expert_ids.numel()}"
            )
        if out.device != x.device or out.dtype != torch.bfloat16:
            raise ValueError("out must be a bfloat16 tensor on the same device as x")
        inter = out[: expert_ids.numel()]
    grid = (expert_ids.numel(), triton.cdiv(INTERMEDIATE_SIZE, block_inter))
    _grouped_gate_up_silu_fast_decode_kernel[grid](
        x.contiguous(),
        expert_ids,
        gate_up_packed.contiguous(),
        gate_up_effective_scale.contiguous(),
        gate_up_global_scale.to(device=x.device).contiguous(),
        inter,
        gate_up_packed.stride(0),
        gate_up_packed.stride(1),
        gate_up_packed.stride(2),
        gate_up_effective_scale.stride(0),
        gate_up_effective_scale.stride(1),
        gate_up_effective_scale.stride(2),
        inter.stride(0),
        inter.stride(1),
        HIDDEN=HIDDEN_SIZE,
        INTERMEDIATE=INTERMEDIATE_SIZE,
        BLOCK_INTER=block_inter,
        BLOCK_HIDDEN=block_hidden,
        SCALE_EFFECTIVE=True,
        num_warps=num_warps,
    )
    return inter


def nvfp4_grouped_gate_up_silu_scale_hoist(
    x: torch.Tensor,
    expert_ids: torch.Tensor,
    gate_up_packed: torch.Tensor,
    gate_up_scale: torch.Tensor,
    gate_up_global_scale: torch.Tensor,
    *,
    block_inter: int = 8,
    block_hidden: int = 256,
    num_warps: int = 4,
) -> torch.Tensor:
    """P53 scale-hoisted gate/up probe.

    This variant loads each per-16 scale once per group instead of repeating it
    across the 16 columns. It is intentionally opt-in until full-generate gates
    prove both speed and numerical stability.
    """
    _require_triton()
    if block_hidden % 16 != 0:
        raise ValueError(f"block_hidden must be divisible by 16, got {block_hidden}")
    if x.ndim != 1 or x.numel() != HIDDEN_SIZE:
        raise ValueError(f"x must be [2048], got {tuple(x.shape)}")
    if gate_up_packed.ndim != 3 or gate_up_scale.ndim != 3:
        raise ValueError(
            f"expected grouped 3D tensors, got packed={tuple(gate_up_packed.shape)} scale={tuple(gate_up_scale.shape)}"
        )
    expert_ids = expert_ids.to(device=x.device, dtype=torch.int32).contiguous()
    inter = torch.empty((expert_ids.numel(), INTERMEDIATE_SIZE), device=x.device, dtype=torch.bfloat16)
    grid = (expert_ids.numel(), triton.cdiv(INTERMEDIATE_SIZE, block_inter))
    _grouped_gate_up_silu_scale_hoist_kernel[grid](
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


def nvfp4_grouped_gate_up_silu_fast_decode(
    x: torch.Tensor,
    expert_ids: torch.Tensor,
    gate_up_packed: torch.Tensor,
    gate_up_scale: torch.Tensor,
    gate_up_global_scale: torch.Tensor,
    *,
    block_inter: int = 8,
    block_hidden: int = 256,
    num_warps: int = 4,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """P53 lightweight probe: same kernel shape, faster E2M1 expression only."""
    _require_triton()
    if x.ndim != 1 or x.numel() != HIDDEN_SIZE:
        raise ValueError(f"x must be [2048], got {tuple(x.shape)}")
    if gate_up_packed.ndim != 3 or gate_up_scale.ndim != 3:
        raise ValueError(
            f"expected grouped 3D tensors, got packed={tuple(gate_up_packed.shape)} scale={tuple(gate_up_scale.shape)}"
        )
    expert_ids = expert_ids.to(device=x.device, dtype=torch.int32).contiguous()
    if out is None:
        inter = torch.empty((expert_ids.numel(), INTERMEDIATE_SIZE), device=x.device, dtype=torch.bfloat16)
    else:
        if out.ndim != 2 or out.shape[0] < expert_ids.numel() or out.shape[1] != INTERMEDIATE_SIZE:
            raise ValueError(
                f"out must be at least [top_k, {INTERMEDIATE_SIZE}], got {tuple(out.shape)} "
                f"for top_k={expert_ids.numel()}"
            )
        if out.device != x.device or out.dtype != torch.bfloat16:
            raise ValueError("out must be a bfloat16 tensor on the same device as x")
        inter = out[: expert_ids.numel()]
    grid = (expert_ids.numel(), triton.cdiv(INTERMEDIATE_SIZE, block_inter))
    _grouped_gate_up_silu_fast_decode_kernel[grid](
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
        SCALE_EFFECTIVE=False,
        num_warps=num_warps,
    )
    return inter


def nvfp4_grouped_gate_up_silu_fast_decode_prepared(
    x: torch.Tensor,
    expert_ids: torch.Tensor,
    gate_up_packed: torch.Tensor,
    gate_up_scale: torch.Tensor,
    gate_up_global_scale: torch.Tensor,
    out: torch.Tensor,
    *,
    scale_effective: bool = False,
) -> torch.Tensor:
    """Prepared-shape gate/up entrypoint for the resident decode hot path.

    The public wrappers defensively cast and contiguous-copy inputs. The fixed
    resident W4A16 path already prepares these tensors. This exact wrapper keeps
    the same Triton kernel and constants, but avoids redundant boundary ops.
    """
    _require_triton()
    top_k = expert_ids.numel()
    if x.ndim != 1 or x.numel() != HIDDEN_SIZE:
        raise ValueError(f"x must be [2048], got {tuple(x.shape)}")
    if expert_ids.ndim != 1 or expert_ids.dtype != torch.int32 or expert_ids.device != x.device:
        raise ValueError("expert_ids must be a contiguous int32 tensor on x.device")
    if not expert_ids.is_contiguous():
        raise ValueError("expert_ids must be contiguous")
    if gate_up_packed.ndim != 3 or gate_up_scale.ndim != 3:
        raise ValueError(
            f"expected grouped 3D tensors, got packed={tuple(gate_up_packed.shape)} scale={tuple(gate_up_scale.shape)}"
        )
    if not (
        x.is_contiguous()
        and gate_up_packed.is_contiguous()
        and gate_up_scale.is_contiguous()
        and gate_up_global_scale.is_contiguous()
    ):
        raise ValueError("prepared gate/up tensors must be contiguous")
    if out.ndim != 2 or out.shape[0] < top_k or out.shape[1] != INTERMEDIATE_SIZE:
        raise ValueError(f"out must be at least [top_k, {INTERMEDIATE_SIZE}], got {tuple(out.shape)}")
    if out.device != x.device or out.dtype != torch.bfloat16:
        raise ValueError("out must be a bfloat16 tensor on x.device")
    inter = out[:top_k]
    grid = (top_k, triton.cdiv(INTERMEDIATE_SIZE, 8))
    _grouped_gate_up_silu_fast_decode_kernel[grid](
        x,
        expert_ids,
        gate_up_packed,
        gate_up_scale,
        gate_up_global_scale,
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
        BLOCK_INTER=8,
        BLOCK_HIDDEN=256,
        SCALE_EFFECTIVE=scale_effective,
        num_warps=4,
    )
    return inter


def nvfp4_grouped_gate_up_silu_merged_topk(
    x: torch.Tensor,
    expert_ids: torch.Tensor,
    gate_up_packed: torch.Tensor,
    gate_up_scale: torch.Tensor,
    gate_up_global_scale: torch.Tensor,
    *,
    block_inter: int = 8,
    block_hidden: int = 256,
    num_warps: int = 4,
) -> torch.Tensor:
    """Gate/up variant with one program per inter block and a top-k inner loop.

    This is an opt-in P26 probe, not the production default. It tests whether
    reducing kernel program count from `[top_k, inter_blocks]` to
    `[inter_blocks]` helps launch/scheduling overhead while preserving the
    current per-16 scalar contract.
    """
    _require_triton()
    if x.ndim != 1 or x.numel() != HIDDEN_SIZE:
        raise ValueError(f"x must be [2048], got {tuple(x.shape)}")
    if gate_up_packed.ndim != 3 or gate_up_scale.ndim != 3:
        raise ValueError(
            f"expected grouped 3D tensors, got packed={tuple(gate_up_packed.shape)} scale={tuple(gate_up_scale.shape)}"
        )
    expert_ids = expert_ids.to(device=x.device, dtype=torch.int32).contiguous()
    inter = torch.empty((expert_ids.numel(), INTERMEDIATE_SIZE), device=x.device, dtype=torch.bfloat16)
    grid = (triton.cdiv(INTERMEDIATE_SIZE, block_inter),)
    _grouped_gate_up_silu_merged_topk_kernel[grid](
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
        inter.stride(1),
        HIDDEN=HIDDEN_SIZE,
        INTERMEDIATE=INTERMEDIATE_SIZE,
        TOP_K=expert_ids.numel(),
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
    out: torch.Tensor | None = None,
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
    if out is None:
        out = torch.empty((HIDDEN_SIZE,), device=inter.device, dtype=torch.bfloat16)
    else:
        if out.shape != (HIDDEN_SIZE,):
            raise ValueError(f"out must be [{HIDDEN_SIZE}], got {tuple(out.shape)}")
        if out.device != inter.device or out.dtype != torch.bfloat16:
            raise ValueError("out must be a bfloat16 tensor on the same device as inter")
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
        SCALE_EFFECTIVE=False,
        num_warps=num_warps,
    )
    return out


def nvfp4_grouped_down_weighted_sum_effective_scale(
    inter: torch.Tensor,
    expert_ids: torch.Tensor,
    routing_weights: torch.Tensor,
    down_packed: torch.Tensor,
    down_effective_scale: torch.Tensor,
    down_global_scale: torch.Tensor,
    *,
    block_hidden: int = 16,
    block_inter: int = 128,
    num_warps: int = 4,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Down projection consuming precomputed `scale / global_scale`."""
    _require_triton()
    if inter.ndim != 2 or inter.shape[1] != INTERMEDIATE_SIZE:
        raise ValueError(f"inter must be [top_k, 512], got {tuple(inter.shape)}")
    if down_packed.ndim != 3 or down_effective_scale.ndim != 3:
        raise ValueError(
            "expected grouped 3D tensors, got "
            f"packed={tuple(down_packed.shape)} scale={tuple(down_effective_scale.shape)}"
        )
    expert_ids = expert_ids.to(device=inter.device, dtype=torch.int32).contiguous()
    routing_weights = routing_weights.to(device=inter.device, dtype=torch.float32).contiguous()
    if expert_ids.numel() != inter.shape[0] or routing_weights.numel() != inter.shape[0]:
        raise ValueError("expert_ids/routing_weights must match inter top_k")
    if out is None:
        out = torch.empty((HIDDEN_SIZE,), device=inter.device, dtype=torch.bfloat16)
    else:
        if out.shape != (HIDDEN_SIZE,):
            raise ValueError(f"out must be [{HIDDEN_SIZE}], got {tuple(out.shape)}")
        if out.device != inter.device or out.dtype != torch.bfloat16:
            raise ValueError("out must be a bfloat16 tensor on the same device as inter")
    grid = (triton.cdiv(HIDDEN_SIZE, block_hidden),)
    _grouped_down_weighted_sum_kernel[grid](
        inter.contiguous(),
        expert_ids,
        routing_weights,
        down_packed.contiguous(),
        down_effective_scale.contiguous(),
        down_global_scale.to(device=inter.device).contiguous(),
        out,
        down_packed.stride(0),
        down_packed.stride(1),
        down_packed.stride(2),
        down_effective_scale.stride(0),
        down_effective_scale.stride(1),
        down_effective_scale.stride(2),
        inter.stride(0),
        inter.stride(1),
        TOP_K=inter.shape[0],
        HIDDEN=HIDDEN_SIZE,
        INTERMEDIATE=INTERMEDIATE_SIZE,
        BLOCK_HIDDEN=block_hidden,
        BLOCK_INTER=block_inter,
        SCALE_EFFECTIVE=True,
        num_warps=num_warps,
    )
    return out


def nvfp4_grouped_down_weighted_sum_prepared(
    inter: torch.Tensor,
    expert_ids: torch.Tensor,
    routing_weights: torch.Tensor,
    down_packed: torch.Tensor,
    down_scale: torch.Tensor,
    down_global_scale: torch.Tensor,
    out: torch.Tensor,
    *,
    scale_effective: bool = False,
) -> torch.Tensor:
    """Prepared-shape down entrypoint for the resident decode hot path."""
    _require_triton()
    if inter.ndim != 2 or inter.shape[1] != INTERMEDIATE_SIZE:
        raise ValueError(f"inter must be [top_k, 512], got {tuple(inter.shape)}")
    if expert_ids.ndim != 1 or expert_ids.dtype != torch.int32 or expert_ids.device != inter.device:
        raise ValueError("expert_ids must be a contiguous int32 tensor on inter.device")
    if routing_weights.ndim != 1 or routing_weights.dtype != torch.float32 or routing_weights.device != inter.device:
        raise ValueError("routing_weights must be a contiguous float32 tensor on inter.device")
    if not expert_ids.is_contiguous() or not routing_weights.is_contiguous():
        raise ValueError("expert_ids and routing_weights must be contiguous")
    if expert_ids.numel() != inter.shape[0] or routing_weights.numel() != inter.shape[0]:
        raise ValueError("expert_ids/routing_weights must match inter top_k")
    if down_packed.ndim != 3 or down_scale.ndim != 3:
        raise ValueError(
            f"expected grouped 3D tensors, got packed={tuple(down_packed.shape)} scale={tuple(down_scale.shape)}"
        )
    if not (inter.is_contiguous() and down_packed.is_contiguous() and down_scale.is_contiguous() and down_global_scale.is_contiguous()):
        raise ValueError("prepared down tensors must be contiguous")
    if out.shape != (HIDDEN_SIZE,):
        raise ValueError(f"out must be [{HIDDEN_SIZE}], got {tuple(out.shape)}")
    if out.device != inter.device or out.dtype != torch.bfloat16:
        raise ValueError("out must be a bfloat16 tensor on inter.device")
    grid = (triton.cdiv(HIDDEN_SIZE, 8),)
    _grouped_down_weighted_sum_kernel[grid](
        inter,
        expert_ids,
        routing_weights,
        down_packed,
        down_scale,
        down_global_scale,
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
        BLOCK_HIDDEN=8,
        BLOCK_INTER=512,
        SCALE_EFFECTIVE=scale_effective,
        num_warps=8,
    )
    return out


def nvfp4_grouped_down_weighted_sum_scale_hoist(
    inter: torch.Tensor,
    expert_ids: torch.Tensor,
    routing_weights: torch.Tensor,
    down_packed: torch.Tensor,
    down_scale: torch.Tensor,
    down_global_scale: torch.Tensor,
    *,
    block_hidden: int = 8,
    block_inter: int = 512,
    num_warps: int = 8,
) -> torch.Tensor:
    """P53 scale-hoisted down weighted-sum probe."""
    _require_triton()
    if block_inter != INTERMEDIATE_SIZE:
        raise ValueError("scale-hoist down currently expects block_inter=512")
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
    _grouped_down_weighted_sum_scale_hoist_kernel[grid](
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


__all__ = [
    "HAS_TRITON",
    "nvfp4_grouped_down_weighted_sum",
    "nvfp4_grouped_down_weighted_sum_effective_scale",
    "nvfp4_grouped_down_weighted_sum_prepared",
    "nvfp4_grouped_down_weighted_sum_scale_hoist",
    "nvfp4_grouped_gate_up_silu",
    "nvfp4_grouped_gate_up_silu_fast_decode",
    "nvfp4_grouped_gate_up_silu_fast_decode_effective_scale",
    "nvfp4_grouped_gate_up_silu_fast_decode_prepared",
    "nvfp4_grouped_gate_up_silu_merged_topk",
    "nvfp4_grouped_gate_up_silu_scale_hoist",
]
