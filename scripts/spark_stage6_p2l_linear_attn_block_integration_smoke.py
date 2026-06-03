#!/usr/bin/env python3
"""Stage 6 Phase 2-L: opt-in linear-attn block-kernel integration smoke."""
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

from engine.incremental_decode import prefill_linear_attn  # noqa: E402
from engine.loader import load_qwen36_layer  # noqa: E402
from scripts.spark_stage6_p1_dense_projection_poc import _diff_stats  # noqa: E402
from scripts.spark_stage6_p2k_gated_delta_native_loop_poc import DEFAULT_MODEL, _model_cfg, _parse_ints  # noqa: E402


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


def _prefill_with_flag(h: torch.Tensor, w: dict[str, Any], *, enabled: bool, chunk_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    old = os.environ.get("LYNN_LINEAR_ATTN_PREFILL_BLOCK_GQA")
    os.environ["LYNN_LINEAR_ATTN_PREFILL_BLOCK_GQA"] = "1" if enabled else "0"
    try:
        return prefill_linear_attn(h, w, chunk_size=chunk_size)
    finally:
        if old is None:
            os.environ.pop("LYNN_LINEAR_ATTN_PREFILL_BLOCK_GQA", None)
        else:
            os.environ["LYNN_LINEAR_ATTN_PREFILL_BLOCK_GQA"] = old


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

    print("=============== STAGE 6 PHASE 2-L LINEAR-ATTN BLOCK INTEGRATION SMOKE ===============", flush=True)
    print(f"model      : {model_dir}", flush=True)
    print(f"layer      : {args.layer}", flush=True)
    print(f"seq_lens   : {seq_lens}", flush=True)
    print(f"chunk_size : {args.chunk_size}", flush=True)

    rows: list[dict[str, Any]] = []
    numeric: dict[str, Any] = {}
    for seq_len in seq_lens:
        h = (torch.randn((1, seq_len, hidden), device=device, dtype=torch.float32) * 0.35).to(torch.bfloat16)

        ref_out, ref_state, ref_conv = _prefill_with_flag(h, w, enabled=False, chunk_size=args.chunk_size)
        block_out, block_state, block_conv = _prefill_with_flag(h, w, enabled=True, chunk_size=args.chunk_size)

        numeric[f"T{seq_len}_out"] = _diff_stats(block_out, ref_out)
        numeric[f"T{seq_len}_recurrent_state"] = _diff_stats(block_state, ref_state)
        numeric[f"T{seq_len}_conv_state"] = _diff_stats(block_conv, ref_conv)

        def ref_fn():
            return _prefill_with_flag(h, w, enabled=False, chunk_size=args.chunk_size)

        def block_fn():
            return _prefill_with_flag(h, w, enabled=True, chunk_size=args.chunk_size)

        ref_bench = _bench_wall_ms(ref_fn, repeats=args.repeats)
        block_bench = _bench_wall_ms(block_fn, repeats=args.repeats)
        row = {
            "seq_len": seq_len,
            "reference_ms": ref_bench["median_ms"],
            "block_optin_ms": block_bench["median_ms"],
            "block_vs_reference": ref_bench["median_ms"] / block_bench["median_ms"] if block_bench["median_ms"] else None,
            "ref_bench": ref_bench,
            "block_bench": block_bench,
        }
        rows.append(row)
        print(
            f"[T={seq_len}] ref={row['reference_ms']:.2f}ms "
            f"block={row['block_optin_ms']:.2f}ms "
            f"ratio={row['block_vs_reference']:.3f}x "
            f"out_cos={numeric[f'T{seq_len}_out']['cosine']:.9f} "
            f"out_rel_l2={numeric[f'T{seq_len}_out']['rel_l2']:.3e}",
            flush=True,
        )

    numeric_pass = all(v["cosine"] > 0.999 and v["argmax_match"] for v in numeric.values())
    speed_pass = all((r["block_vs_reference"] or 0.0) >= 1.0 for r in rows)
    result = {
        "schema": "lynn-stage6-p2l-linear-attn-block-integration-smoke-v1",
        "model": str(model_dir),
        "layer": args.layer,
        "seed": args.seed,
        "seq_lens": seq_lens,
        "chunk_size": args.chunk_size,
        "rows": rows,
        "numeric": numeric,
        "passes": {
            "numeric": bool(numeric_pass),
            "speed_vs_reference": bool(speed_pass),
            "all": bool(numeric_pass and speed_pass),
        },
        "notes": [
            "This smoke only exercises prefill_linear_attn with LYNN_LINEAR_ATTN_PREFILL_BLOCK_GQA=1.",
            "Default path remains unchanged when the flag is unset.",
            "Promotion still requires selected-layer/full-prefill and RC quality gates.",
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
