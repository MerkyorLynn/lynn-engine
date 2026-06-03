#!/usr/bin/env python3
"""Stage 6 Phase 2-KB: gated-delta block-kernel PoC.

P2-KA proved that the existing single-token Triton recurrent kernel matches the
linear-attention chunk reference, but one host launch per prefill token regresses
badly as T grows. P2-KB moves that recurrent loop inside one Triton launch.

This is still a lower-bound kernel: it uses the recurrent recurrence, not HF's
chunk triangular algorithm, and each program owns one value-head/v-block. The
question is whether one-launch block recurrence is numerically acceptable and
meaningfully faster than the P2-KA host loop.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.loader import load_qwen36_layer  # noqa: E402
from scripts.spark_stage6_p1_dense_projection_poc import _diff_stats  # noqa: E402
from scripts.spark_stage6_p2k_gated_delta_native_loop_poc import (  # noqa: E402
    DEFAULT_MODEL,
    _model_cfg,
    _native_recurrent_loop,
    _parse_ints,
    _prepare_linear_attn_inputs,
    _reference_chunk,
)
from triton_kernels.gated_delta import recurrent_gated_delta_block_gqa  # noqa: E402


def _time_wall_ms(fn: Callable[[], Any]) -> tuple[Any, float]:
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = fn()
    torch.cuda.synchronize()
    return out, (time.perf_counter() - t0) * 1000.0


def _bench_wall_ms(fn: Callable[[], Any], *, repeats: int) -> dict[str, Any]:
    times: list[float] = []
    for _ in range(repeats):
        out, ms = _time_wall_ms(fn)
        times.append(ms)
        del out
        torch.cuda.empty_cache()
    times_sorted = sorted(times)
    return {
        "repeats": repeats,
        "median_ms": float(times_sorted[len(times_sorted) // 2]),
        "min_ms": float(min(times)),
        "max_ms": float(max(times)),
        "all_ms": times,
    }


def _block_kernel(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    inplace: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    old = os.environ.get("LYNN_LINEAR_ATTN_RECURRENT_INPLACE")
    os.environ["LYNN_LINEAR_ATTN_RECURRENT_INPLACE"] = "1" if inplace else "0"
    try:
        state = torch.zeros((1, 32, 128, 128), device=q.device, dtype=torch.float32)
        return recurrent_gated_delta_block_gqa(q, k, v, g, beta, state)
    finally:
        if old is None:
            os.environ.pop("LYNN_LINEAR_ATTN_RECURRENT_INPLACE", None)
        else:
            os.environ["LYNN_LINEAR_ATTN_RECURRENT_INPLACE"] = old


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--seq-lens", default="16,64,128,256,512")
    ap.add_argument("--chunk-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=20260604)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--json-out", default="")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = "cuda"
    model_dir = Path(args.model)
    text_cfg = _model_cfg(model_dir)
    num_experts = int(text_cfg.get("num_experts", 256))
    torch.manual_seed(args.seed)

    w, _cfg = load_qwen36_layer(
        str(model_dir),
        args.layer,
        num_experts=num_experts,
        device=device,
        dequant_dtype=torch.bfloat16,
    )
    hidden = int(w["linear_attn.in_proj_qkv.weight"].shape[1])
    seq_lens = _parse_ints(args.seq_lens)

    print("=============== STAGE 6 PHASE 2-KB GATED-DELTA BLOCK KERNEL POC ===============", flush=True)
    print(f"model      : {model_dir}", flush=True)
    print(f"layer      : {args.layer}", flush=True)
    print(f"seq_lens   : {seq_lens}", flush=True)
    print(f"chunk_size : {args.chunk_size}", flush=True)

    rows: list[dict[str, Any]] = []
    numeric: dict[str, Any] = {}
    for seq_len in seq_lens:
        h = (torch.randn((1, seq_len, hidden), device=device, dtype=torch.float32) * 0.35).to(torch.bfloat16)
        q, k, v, g, beta = _prepare_linear_attn_inputs(h, w)
        ref_out, ref_state = _reference_chunk(q, k, v, g, beta, chunk_size=args.chunk_size)
        loop_out, loop_state = _native_recurrent_loop(q, k, v, g, beta, inplace=True)
        block_out, block_state = _block_kernel(q, k, v, g, beta, inplace=True)

        numeric[f"T{seq_len}_loop_out_vs_chunk"] = _diff_stats(loop_out, ref_out)
        numeric[f"T{seq_len}_loop_state_vs_chunk"] = _diff_stats(loop_state, ref_state)
        numeric[f"T{seq_len}_block_out_vs_chunk"] = _diff_stats(block_out, ref_out)
        numeric[f"T{seq_len}_block_state_vs_chunk"] = _diff_stats(block_state, ref_state)
        numeric[f"T{seq_len}_block_out_vs_loop"] = _diff_stats(block_out, loop_out)
        numeric[f"T{seq_len}_block_state_vs_loop"] = _diff_stats(block_state, loop_state)

        def ref_fn():
            return _reference_chunk(q, k, v, g, beta, chunk_size=args.chunk_size)

        def loop_fn():
            return _native_recurrent_loop(q, k, v, g, beta, inplace=True)

        def block_fn():
            return _block_kernel(q, k, v, g, beta, inplace=True)

        ref_bench = _bench_wall_ms(ref_fn, repeats=args.repeats)
        loop_bench = _bench_wall_ms(loop_fn, repeats=args.repeats)
        block_bench = _bench_wall_ms(block_fn, repeats=args.repeats)
        row = {
            "seq_len": seq_len,
            "chunk_reference_ms": ref_bench["median_ms"],
            "host_loop_inplace_ms": loop_bench["median_ms"],
            "block_kernel_inplace_ms": block_bench["median_ms"],
            "block_vs_chunk": ref_bench["median_ms"] / block_bench["median_ms"] if block_bench["median_ms"] else None,
            "block_vs_host_loop": loop_bench["median_ms"] / block_bench["median_ms"] if block_bench["median_ms"] else None,
            "estimated_kernel_launches": {
                "chunk_reference": "torch graph",
                "host_loop": seq_len,
                "block_kernel": 1,
            },
            "ref_bench": ref_bench,
            "loop_bench": loop_bench,
            "block_bench": block_bench,
        }
        rows.append(row)
        print(
            f"[T={seq_len}] chunk={row['chunk_reference_ms']:.2f}ms "
            f"host_loop={row['host_loop_inplace_ms']:.2f}ms "
            f"block={row['block_kernel_inplace_ms']:.2f}ms "
            f"block/loop={row['block_vs_host_loop']:.3f}x "
            f"block/chunk={row['block_vs_chunk']:.3f}x",
            flush=True,
        )
        print(
            f"  block-vs-chunk cos={numeric[f'T{seq_len}_block_out_vs_chunk']['cosine']:.9f} "
            f"rel_l2={numeric[f'T{seq_len}_block_out_vs_chunk']['rel_l2']:.3e}; "
            f"block-vs-loop cos={numeric[f'T{seq_len}_block_out_vs_loop']['cosine']:.9f} "
            f"rel_l2={numeric[f'T{seq_len}_block_out_vs_loop']['rel_l2']:.3e}",
            flush=True,
        )

    numeric_pass = all(v["cosine"] > 0.999 and v["argmax_match"] for v in numeric.values())
    launch_cut_pass = all((r["block_vs_host_loop"] or 0.0) >= 1.0 for r in rows)
    speed_vs_chunk_pass = all((r["block_vs_chunk"] or 0.0) >= 1.0 for r in rows)
    result = {
        "schema": "lynn-stage6-p2kb-gated-delta-block-kernel-poc-v1",
        "model": str(model_dir),
        "layer": args.layer,
        "seed": args.seed,
        "seq_lens": seq_lens,
        "chunk_size": args.chunk_size,
        "rows": rows,
        "numeric": numeric,
        "passes": {
            "numeric": bool(numeric_pass),
            "launch_cut_vs_host_loop": bool(launch_cut_pass),
            "speed_vs_chunk_reference": bool(speed_vs_chunk_pass),
            "all": bool(numeric_pass and launch_cut_pass and speed_vs_chunk_pass),
        },
        "notes": [
            "P2-KB moves the P2-KA recurrent loop inside one Triton launch.",
            "It is a lower-bound block recurrent kernel, not the HF chunk triangular algorithm.",
            "Promotion still requires full prefill integration plus RC quality.",
        ],
    }
    print("=============== RESULT JSON ===============", flush=True)
    print(json.dumps(result, indent=2), flush=True)
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
