#!/usr/bin/env python3
"""Stage 6 Phase 1-A: batched dense projection packed-NVFP4 PoC.

P1 proved a single real projection can run from packed NVFP4 without a BF16
shadow. P1-A asks the next question for prefill: can one kernel launch cover
multiple token rows, avoiding the slow Python row loop while preserving the same
packed weight contract?
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.dequant import dequantize_nvfp4_v8_rtn_weight  # noqa: E402
from scripts.spark_stage6_p1_dense_projection_poc import (  # noqa: E402
    DEFAULT_BASE_KEY,
    DEFAULT_MODEL,
    GIB,
    _bench_cuda,
    _diff_stats,
    _load_projection,
    _nbytes,
)
from triton_kernels.nvfp4_linear import nvfp4_batched_matmul_packed  # noqa: E402


def _parse_batches(text: str) -> list[int]:
    vals = [int(x.strip()) for x in text.split(",") if x.strip()]
    if not vals or any(v <= 0 for v in vals):
        raise argparse.ArgumentTypeError("--batches must be a comma-separated list of positive integers")
    return vals


def _summary(vals: list[float]) -> dict[str, float]:
    return {
        "median": float(statistics.median(vals)),
        "min": float(min(vals)),
        "max": float(max(vals)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--base-key", default=DEFAULT_BASE_KEY)
    ap.add_argument("--batches", type=_parse_batches, default=[1, 4, 16, 64])
    ap.add_argument("--seed", type=int, default=20260604)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=80)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--block-m", type=int, default=16)
    ap.add_argument("--block-n", type=int, default=128)
    ap.add_argument("--json-out", default="")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = "cuda"
    model_dir = Path(args.model)
    torch.manual_seed(args.seed)

    loaded = _load_projection(model_dir, args.base_key, device=device)
    rec = loaded["record"]
    packed: torch.Tensor = loaded["packed"]
    scale: torch.Tensor = loaded["scale"]
    global_scale: torch.Tensor = loaded["global_scale"]
    out_features, in_features = map(int, rec["original_shape"])
    bf16_shadow_bytes = out_features * in_features * 2
    packed_weight_bytes = _nbytes(packed) + _nbytes(scale) + _nbytes(global_scale)

    print("=============== STAGE 6 PHASE 1-A BATCHED PROJECTION POC ===============", flush=True)
    print(f"model        : {model_dir}", flush=True)
    print(f"base_key     : {args.base_key}", flush=True)
    print(f"shape        : out={out_features} in={in_features}", flush=True)
    print(f"batches      : {args.batches}", flush=True)
    print(f"scale dtype  : {scale.dtype}", flush=True)
    print(f"BF16 shadow  : {bf16_shadow_bytes / 1024**2:.2f} MiB", flush=True)
    print(f"packed+scale : {packed_weight_bytes / 1024**2:.2f} MiB", flush=True)

    w_fp32 = dequantize_nvfp4_v8_rtn_weight(
        packed,
        scale,
        global_scale,
        output_dtype=torch.float32,
    )
    w_bf16 = w_fp32.to(torch.bfloat16)

    xs: dict[int, torch.Tensor] = {}
    numeric: dict[str, Any] = {}
    bench_bf16: dict[str, Any] = {}
    for batch in args.batches:
        x = (torch.randn((batch, in_features), device=device, dtype=torch.float32) * 0.35).to(torch.bfloat16)
        xs[batch] = x
        ref_fp32 = x.float() @ w_fp32.t()
        ref_bf16 = F.linear(x, w_bf16).float()
        y = nvfp4_batched_matmul_packed(
            x,
            packed,
            scale,
            global_scale,
            block_m=args.block_m,
            block_n=args.block_n,
        )
        torch.cuda.synchronize()
        numeric[str(batch)] = {
            "vs_fp32_dequant": _diff_stats(y, ref_fp32),
            "vs_bf16_shadow": _diff_stats(y, ref_bf16),
        }
        bench_bf16[str(batch)] = _bench_cuda(
            lambda x=x: F.linear(x, w_bf16),
            warmup=args.warmup,
            iters=args.iters,
            repeats=args.repeats,
        )
        print(
            f"[numeric M={batch}] cos={numeric[str(batch)]['vs_fp32_dequant']['cosine']:.9f} "
            f"rel_l2={numeric[str(batch)]['vs_fp32_dequant']['rel_l2']:.3e} "
            f"argmax={numeric[str(batch)]['vs_fp32_dequant']['argmax_match']}",
            flush=True,
        )

    del w_fp32, w_bf16, ref_fp32, ref_bf16, y
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    before_packed_alloc = int(torch.cuda.memory_allocated())
    bench_packed: dict[str, Any] = {}
    for batch, x in xs.items():
        bench_packed[str(batch)] = _bench_cuda(
            lambda x=x: nvfp4_batched_matmul_packed(
                x,
                packed,
                scale,
                global_scale,
                block_m=args.block_m,
                block_n=args.block_n,
            ),
            warmup=args.warmup,
            iters=args.iters,
            repeats=args.repeats,
        )
    after_packed_alloc = int(torch.cuda.memory_allocated())
    packed_peak_alloc = int(torch.cuda.max_memory_allocated())

    rows: list[dict[str, Any]] = []
    for batch in args.batches:
        bp = bench_packed[str(batch)]
        bb = bench_bf16[str(batch)]
        speedup = bb["median_us"] / bp["median_us"] if bp["median_us"] else math.nan
        rows.append(
            {
                "batch": batch,
                "packed_median_us": bp["median_us"],
                "bf16_median_us": bb["median_us"],
                "speedup_vs_bf16": speedup,
                "packed_us_per_token": bp["median_us"] / batch,
                "bf16_us_per_token": bb["median_us"] / batch,
            }
        )
        print(
            f"[bench M={batch}] packed={bp['median_us']:.2f}us "
            f"bf16={bb['median_us']:.2f}us speedup={speedup:.3f}x",
            flush=True,
        )

    numeric_pass = all(
        row["vs_fp32_dequant"]["cosine"] >= 0.99999
        and row["vs_fp32_dequant"]["rel_l2"] <= 2.0e-3
        and row["vs_fp32_dequant"]["argmax_match"]
        for row in numeric.values()
    )
    no_shadow_pass = packed_peak_alloc < bf16_shadow_bytes
    perf_pass_all = all(row["speedup_vs_bf16"] >= 1.0 for row in rows)
    result = {
        "schema": "lynn-stage6-phase1a-batched-projection-poc-v1",
        "model": str(model_dir),
        "base_key": args.base_key,
        "seed": args.seed,
        "batches": args.batches,
        "shape": {"out_features": out_features, "in_features": in_features},
        "dtypes": {
            "packed": str(packed.dtype),
            "scale": str(scale.dtype),
            "global_scale": str(global_scale.dtype),
        },
        "bytes": {
            "bf16_shadow": bf16_shadow_bytes,
            "packed_weight_total": packed_weight_bytes,
            "bf16_to_packed_weight_ratio": bf16_shadow_bytes / packed_weight_bytes,
        },
        "numeric": numeric,
        "bench": {
            "rows": rows,
            "packed_triton": bench_packed,
            "bf16_shadow_linear": bench_bf16,
        },
        "memory_after_deleting_bf16_refs": {
            "before_packed_bench_gib": before_packed_alloc / GIB,
            "after_packed_bench_gib": after_packed_alloc / GIB,
            "peak_packed_bench_gib": packed_peak_alloc / GIB,
        },
        "passes": {
            "numeric": bool(numeric_pass),
            "no_bf16_shadow_allocated_for_packed_bench": bool(no_shadow_pass),
            "perf_speedup_vs_bf16_all_batches": bool(perf_pass_all),
            "all": bool(numeric_pass and no_shadow_pass and perf_pass_all),
        },
        "notes": [
            "The first P1-A kernel removes the Python row loop but does not yet share activation tiles across tokens.",
            "Perf may need tiling work before promotion even if numeric/no-shadow pass.",
        ],
    }
    print("=============== RESULT JSON ===============", flush=True)
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    if not numeric_pass or not no_shadow_pass:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
