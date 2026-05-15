#!/usr/bin/env python3
"""P24 spike: use tl.dot inside packed NVFP4 gate/up.

This is a research probe for the custom active expert kernel path. It keeps the
current per-16 scale contract, dequantizes packed E2M1 values inside Triton, and
then uses `tl.dot` for the small `[BLOCK_INTER, BLOCK_HIDDEN] @ [BLOCK_HIDDEN, 1]`
reduction instead of explicit `tl.sum`.

It is not promoted unless it is both faster and close to the current scalar
gate/up reference.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Callable

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("Triton is required for this probe") from exc

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.p10e_packed_active_expert_probe import _load_grouped, _prefill_to_layer_input  # noqa: E402
from engine.full_forward import _rms_norm  # noqa: E402
from engine.resident_runner import LynnIncrementalRunner  # noqa: E402
from triton_kernels.nvfp4_moe import nvfp4_grouped_gate_up_silu  # noqa: E402


HIDDEN_SIZE = 2048
INTERMEDIATE_SIZE = 512


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
def _gateup_tl_dot_kernel(
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

    gate_acc = tl.zeros((BLOCK_INTER, 1), dtype=tl.float32)
    up_acc = tl.zeros((BLOCK_INTER, 1), dtype=tl.float32)

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
        x_col = x[:, None]
        gate_scaled = (gate_w * (gate_scale / global_scale)).to(tl.float16)
        up_scaled = (up_w * (up_scale / global_scale)).to(tl.float16)
        x_scaled = x_col.to(tl.float16)
        gate_acc += tl.dot(gate_scaled, x_scaled)
        up_acc += tl.dot(up_scaled, x_scaled)

    gate_vec = tl.reshape(gate_acc, (BLOCK_INTER,))
    up_vec = tl.reshape(up_acc, (BLOCK_INTER,))
    gate_silu = gate_vec * tl.sigmoid(gate_vec)
    inter = gate_silu * up_vec
    tl.store(inter_ptr + slot * INTER_STRIDE_K + inter_offsets * INTER_STRIDE_I, inter.to(tl.bfloat16), mask=inter_mask)


def gateup_tl_dot(
    x: torch.Tensor,
    expert_ids: torch.Tensor,
    gate_up_packed: torch.Tensor,
    gate_up_scale: torch.Tensor,
    gate_up_global_scale: torch.Tensor,
    *,
    block_inter: int,
    block_hidden: int,
    num_warps: int,
) -> torch.Tensor:
    expert_ids = expert_ids.to(device=x.device, dtype=torch.int32).contiguous()
    inter = torch.empty((expert_ids.numel(), INTERMEDIATE_SIZE), device=x.device, dtype=torch.bfloat16)
    grid = (expert_ids.numel(), triton.cdiv(INTERMEDIATE_SIZE, block_inter))
    _gateup_tl_dot_kernel[grid](
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
        num_stages=3,
    )
    return inter


def _bench(fn: Callable[[], torch.Tensor], warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end) / iters)


def _diff(a: torch.Tensor, b: torch.Tensor) -> dict:
    af = a.float().reshape(-1)
    bf = b.float().reshape(-1)
    return {
        "max_abs": float((af - bf).abs().max().item()),
        "mean_abs": float((af - bf).abs().mean().item()),
        "rel_l2": float(torch.linalg.vector_norm(af - bf).item() / torch.linalg.vector_norm(af).item()),
        "cosine": float(F.cosine_similarity(af, bf, dim=0).item()),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layer", type=int, default=28)
    ap.add_argument("--prompt", default="用一句话解释 MoE active parameters")
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--iters", type=int, default=80)
    args = ap.parse_args()

    model_dir = Path(args.model)
    runner = LynnIncrementalRunner(args.model, device="cuda", dtype=torch.bfloat16, verbose=False)
    h_layer, _ = _prefill_to_layer_input(runner, args.layer, args.prompt)
    w = runner.layer_weights[args.layer]
    cfg = runner.layer_cfgs[args.layer]
    h_moe = _rms_norm(h_layer, w["post_attention_layernorm.weight"])
    h_flat = h_moe.view(-1, h_moe.shape[-1])
    hidden = h_flat[0]
    top_k = int(cfg["num_experts_per_tok"])
    router_logits = F.linear(h_flat, w["mlp.gate.weight"])
    _, expert_indices = torch.topk(router_logits, top_k, dim=-1, sorted=False)
    expert_ids = expert_indices[0].to(torch.int32).contiguous()
    gate_up_packed, gate_up_scale, gate_up_global = _load_grouped(
        model_dir,
        f"model.language_model.layers.{args.layer}.mlp.experts.gate_up_proj",
        runner.device,
    )

    ref = nvfp4_grouped_gate_up_silu(
        hidden,
        expert_ids,
        gate_up_packed,
        gate_up_scale,
        gate_up_global,
        block_inter=8,
        block_hidden=256,
        num_warps=4,
    )
    rows = []
    for block_inter in (4, 8, 16):
        for block_hidden in (32, 64, 128):
            for num_warps in (4, 8):
                def cand(
                    block_inter=block_inter,
                    block_hidden=block_hidden,
                    num_warps=num_warps,
                ) -> torch.Tensor:
                    return gateup_tl_dot(
                        hidden,
                        expert_ids,
                        gate_up_packed,
                        gate_up_scale,
                        gate_up_global,
                        block_inter=block_inter,
                        block_hidden=block_hidden,
                        num_warps=num_warps,
                    )

                out = cand()
                rows.append({
                    "block_inter": block_inter,
                    "block_hidden": block_hidden,
                    "num_warps": num_warps,
                    "ms": _bench(cand, args.warmup, args.iters),
                    "diff_vs_scalar": _diff(ref, out),
                })

    def ref_fn() -> torch.Tensor:
        return nvfp4_grouped_gate_up_silu(
            hidden,
            expert_ids,
            gate_up_packed,
            gate_up_scale,
            gate_up_global,
            block_inter=8,
            block_hidden=256,
            num_warps=4,
        )

    result = {
        "schema_version": "lynn-engine-p24-gateup-tl-dot-probe-v1",
        "model": args.model,
        "layer": args.layer,
        "expert_ids": [int(x) for x in expert_ids.tolist()],
        "reference_scalar_ms": _bench(ref_fn, args.warmup, args.iters),
        "candidates": rows,
        "best_by_speed": sorted(rows, key=lambda r: r["ms"])[:5],
        "best_passing_cosine_999": [
            r for r in sorted(rows, key=lambda r: r["ms"])
            if r["diff_vs_scalar"]["cosine"] >= 0.999
        ][:5],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
