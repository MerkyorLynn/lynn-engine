#!/usr/bin/env python3
"""Verify + benchmark the Spark FP8 fused gate/up + SwiGLU Triton kernel.

Stand-alone test harness for ``triton_kernels/spark_fp8_gate_up_fused.py``.
Generates synthetic NVFP4-equivalent weights, repacks them to FP8 via the
v0 offline tool, then compares:

* correctness: cos vs BF16 reference SwiGLU
* perf: tokens/sec equivalent at decode shapes (M=1, K=2048, N=6144)
* perf: vs naive ``torch.nn.functional.linear`` BF16 path

This is a Phase 2 step 2 unit test. End-to-end TPS measurement against
the full model integration is step 3.

Usage on Spark::

    /home/merkyor/comfyui/ComfyUI/.venv/bin/python -u \
        scripts/spark_fp8_kernel_verify.py \
        --shapes 1x2048x6144 8x2048x6144 16x2048x6144 \
        --bench-iters 200
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.spark_pack_w4a8_fp8 import (  # type: ignore  # noqa: E402
    repack_nvfp4_to_fp8,
    synthetic_nvfp4,
)
from triton_kernels.spark_fp8_gate_up_fused import (  # noqa: E402
    fp8_gate_up_silu_fused,
    fp8_gate_up_silu_reference,
)


def _diff_stats(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    af = a.detach().float().flatten().double()
    bf = b.detach().float().flatten().double()
    diff = af - bf
    max_abs = float(diff.abs().max().item())
    a_norm = float(af.norm().item())
    diff_norm = float(diff.norm().item())
    rel_l2 = float(diff_norm / a_norm) if a_norm > 0 else float("nan")
    cos = float((af * bf).sum().item()) / (a_norm * float(bf.norm().item()) + 1e-12)
    return {"max_abs": max_abs, "rel_l2": rel_l2, "cosine": cos}


def benchmark_call(fn, iters: int, warmup: int = 10) -> dict[str, float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    per_iter_us = elapsed * 1e6 / iters
    return {"elapsed_seconds": elapsed, "iters": iters, "per_iter_us": per_iter_us}


def run_shape(m: int, k: int, n: int, *, bench_iters: int = 200, device: str = "cuda") -> dict:
    """Run correctness + bench for one (M, K, N) shape."""
    print(f"[verify] shape M={m} K={k} N={n}", flush=True)
    torch.manual_seed(0)
    # Synthetic activation (BF16)
    x_bf16 = torch.randn((m, k), dtype=torch.bfloat16, device=device) * 0.5

    # Synthetic NVFP4 weights for gate / up
    gate_packed_cpu, gate_scale_cpu = synthetic_nvfp4(n, k, seed=1234)
    up_packed_cpu, up_scale_cpu = synthetic_nvfp4(n, k, seed=5678)

    gate_repacked = repack_nvfp4_to_fp8(gate_packed_cpu, gate_scale_cpu, None, scale_granularity="per_row")
    up_repacked = repack_nvfp4_to_fp8(up_packed_cpu, up_scale_cpu, None, scale_granularity="per_row")

    w_gate_fp8 = gate_repacked.fp8_weight.to(device)
    w_up_fp8 = up_repacked.fp8_weight.to(device)
    w_gate_scale = gate_repacked.fp8_scale.to(device)
    w_up_scale = up_repacked.fp8_scale.to(device)

    # BF16 reference weights (from FP8 dequant of the same FP8 tensors for fair comparison)
    w_gate_bf16 = (w_gate_fp8.to(torch.float32) * w_gate_scale[:, None]).to(torch.bfloat16)
    w_up_bf16 = (w_up_fp8.to(torch.float32) * w_up_scale[:, None]).to(torch.bfloat16)

    # 1. Correctness
    fused_out = fp8_gate_up_silu_fused(
        x_bf16, w_gate_fp8, w_up_fp8, w_gate_scale, w_up_scale,
    )
    ref_out = fp8_gate_up_silu_reference(x_bf16, w_gate_bf16, w_up_bf16)
    diff = _diff_stats(fused_out, ref_out)
    print(
        f"[verify]   correctness: cos={diff['cosine']:.6f} rel_l2={diff['rel_l2']:.4e} "
        f"max_abs={diff['max_abs']:.4e}",
        flush=True,
    )
    ok = diff["cosine"] > 0.99

    # 2. Bench fused FP8
    bench_fp8 = benchmark_call(
        lambda: fp8_gate_up_silu_fused(
            x_bf16, w_gate_fp8, w_up_fp8, w_gate_scale, w_up_scale,
        ),
        iters=bench_iters,
    )
    # 3. Bench BF16 reference (silu(x @ w_gate^T) * (x @ w_up^T))
    bench_bf16 = benchmark_call(
        lambda: fp8_gate_up_silu_reference(x_bf16, w_gate_bf16, w_up_bf16),
        iters=bench_iters,
    )

    speedup = bench_bf16["per_iter_us"] / bench_fp8["per_iter_us"]
    print(
        f"[verify]   bench: fused_FP8 {bench_fp8['per_iter_us']:.1f}us  vs  "
        f"BF16_ref {bench_bf16['per_iter_us']:.1f}us  speedup={speedup:.2f}×",
        flush=True,
    )

    return {
        "shape": {"M": m, "K": k, "N": n},
        "correctness": diff,
        "correctness_ok": ok,
        "bench_fused_fp8_us": bench_fp8["per_iter_us"],
        "bench_bf16_ref_us": bench_bf16["per_iter_us"],
        "speedup_vs_bf16": speedup,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--shapes",
        nargs="+",
        default=["1x2048x6144", "8x2048x6144", "16x2048x6144"],
        help="MxKxN shapes to test (M=tokens, K=hidden, N=intermediate)",
    )
    ap.add_argument("--bench-iters", type=int, default=200)
    ap.add_argument("--out", default=None, help="optional JSON output path")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")

    results = []
    overall_ok = True
    for spec in args.shapes:
        try:
            m, k, n = (int(x) for x in spec.split("x"))
        except ValueError:
            print(f"bad shape spec {spec!r}, expected MxKxN", file=sys.stderr)
            return 2
        try:
            r = run_shape(m, k, n, bench_iters=args.bench_iters)
            results.append(r)
            if not r["correctness_ok"]:
                overall_ok = False
        except Exception as exc:  # noqa: BLE001
            print(f"[verify]   shape {spec!r} FAILED: {exc!r}", flush=True)
            results.append({"shape_spec": spec, "error": repr(exc)})
            overall_ok = False

    summary = {
        "schema_version": "lynn-spark-fp8-fused-gate-up-verify-v1",
        "results": results,
        "all_ok": overall_ok,
    }
    print("[verify] === summary ===", flush=True)
    for r in results:
        if "error" in r:
            print(f"[verify]   {r.get('shape_spec')}: ERROR {r['error']}", flush=True)
        else:
            s = r["shape"]
            print(
                f"[verify]   M={s['M']} K={s['K']} N={s['N']}: "
                f"cos={r['correctness']['cosine']:.4f} "
                f"speedup={r['speedup_vs_bf16']:.2f}×",
                flush=True,
            )

    if args.out:
        Path(args.out).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(f"[verify] wrote {args.out}", flush=True)

    return 0 if overall_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
