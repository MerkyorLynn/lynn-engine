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
    fp8_gate_up_silu_fused_fp8out,
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

    # 4. fp8out variant: native FP8 intermediate + per-row scale emitted by the
    #    kernel epilogue replaces the four-op Python rescale block in the
    #    callers.  Cosine vs the Python rescale path must be >= 0.999.
    fp8out_diff, fp8out_ok = _verify_fp8out(
        x_bf16, w_gate_fp8, w_up_fp8, w_gate_scale, w_up_scale,
    )
    print(
        f"[verify]   fp8out: bf16_inter cos={fp8out_diff['inter_bf16_cosine']:.6f} "
        f"fp8_inter cos={fp8out_diff['inter_fp8_cosine']:.6f} "
        f"scaled_mm_out cos={fp8out_diff['scaled_mm_cosine']:.6f}",
        flush=True,
    )

    return {
        "shape": {"M": m, "K": k, "N": n},
        "correctness": diff,
        "correctness_ok": ok,
        "bench_fused_fp8_us": bench_fp8["per_iter_us"],
        "bench_bf16_ref_us": bench_bf16["per_iter_us"],
        "speedup_vs_bf16": speedup,
        "fp8out_correctness": fp8out_diff,
        "fp8out_ok": fp8out_ok,
    }


def _python_rescale_to_fp8(inter_bf16: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference: the exact four-op Python rescale block we are replacing.

    Mirrors the block in ``engine/full_forward.py`` and
    ``engine/moe_optimized.py``. Returns ``(inter_fp8, inter_scale)``.
    """
    inter_max = inter_bf16.abs().amax(dim=-1, keepdim=True).clamp_min(1.0e-12)
    inter_scale = (inter_max / 448.0).to(torch.float32)
    inter_fp8 = (inter_bf16.to(torch.float32) / inter_scale).to(torch.float8_e4m3fn)
    return inter_fp8, inter_scale


def _verify_fp8out(
    x_bf16: torch.Tensor,
    w_gate_fp8: torch.Tensor,
    w_up_fp8: torch.Tensor,
    w_gate_scale: torch.Tensor,
    w_up_scale: torch.Tensor,
) -> tuple[dict[str, float], bool]:
    """Compare the new fp8out kernel against the old BF16-out + Python rescale path.

    Runs the same set of weights through both the BF16-out kernel followed by
    the Python rescale block (the path we are replacing) and the new
    ``fp8_gate_up_silu_fused_fp8out`` kernel.  Then runs a synthetic down
    projection through ``torch._scaled_mm`` with both intermediates to confirm
    that the downstream ``_scaled_mm`` result has cosine >= 0.999 — which is
    what the engine actually consumes.
    """
    # Old path: BF16-out kernel + Python rescale.
    inter_bf16_ref = fp8_gate_up_silu_fused(
        x_bf16, w_gate_fp8, w_up_fp8, w_gate_scale, w_up_scale,
    )
    inter_fp8_ref, inter_scale_ref = _python_rescale_to_fp8(inter_bf16_ref)

    # New path: fp8out kernel emits both BF16 and FP8 + per-row scale.
    inter_bf16_new, inter_fp8_new, inter_scale_new = fp8_gate_up_silu_fused_fp8out(
        x_bf16, w_gate_fp8, w_up_fp8, w_gate_scale, w_up_scale,
    )

    # Direct comparison on the FP8 intermediate (cast to F32 for cosine math).
    bf16_diff = _diff_stats(inter_bf16_new, inter_bf16_ref)
    fp8_diff = _diff_stats(
        inter_fp8_new.to(torch.float32), inter_fp8_ref.to(torch.float32),
    )
    scale_diff = _diff_stats(inter_scale_new, inter_scale_ref)

    # End-to-end check: run both through a synthetic _scaled_mm down projection
    # and compare. This is the actual quantity the engine consumes.
    M, N = inter_bf16_new.shape
    K_out = N // 2 if N >= 64 else N
    torch.manual_seed(7)
    w_down = torch.randn((K_out, N), dtype=torch.float32, device=x_bf16.device) * 0.1
    w_down_scale = (w_down.abs().amax(dim=-1).clamp_min(1.0e-12) / 448.0).to(torch.float32)
    w_down_fp8 = (w_down / w_down_scale[:, None]).to(torch.float8_e4m3fn)

    out_ref = torch._scaled_mm(
        inter_fp8_ref,
        w_down_fp8.t(),
        scale_a=inter_scale_ref,
        scale_b=w_down_scale.view(1, -1),
        out_dtype=torch.bfloat16,
    )
    out_new = torch._scaled_mm(
        inter_fp8_new,
        w_down_fp8.t(),
        scale_a=inter_scale_new,
        scale_b=w_down_scale.view(1, -1),
        out_dtype=torch.bfloat16,
    )
    scaled_mm_diff = _diff_stats(out_new, out_ref)

    diff = {
        "inter_bf16_cosine": bf16_diff["cosine"],
        "inter_bf16_rel_l2": bf16_diff["rel_l2"],
        "inter_fp8_cosine": fp8_diff["cosine"],
        "inter_fp8_rel_l2": fp8_diff["rel_l2"],
        "inter_scale_cosine": scale_diff["cosine"],
        "inter_scale_rel_l2": scale_diff["rel_l2"],
        "scaled_mm_cosine": scaled_mm_diff["cosine"],
        "scaled_mm_rel_l2": scaled_mm_diff["rel_l2"],
    }
    # Acceptance: the FP8 intermediate may differ at the LSB but the
    # downstream _scaled_mm output must hold cosine >= 0.999 (engine-level
    # equivalence). BF16 intermediate must be bit-identical (same kernel
    # computation order) — cosine 1.0 expected.
    ok = (
        diff["inter_bf16_cosine"] >= 0.9999
        and diff["scaled_mm_cosine"] >= 0.999
    )
    return diff, ok


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
            if not r.get("fp8out_ok", True):
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
            fp8out_cos = r.get("fp8out_correctness", {}).get(
                "scaled_mm_cosine", float("nan"),
            )
            print(
                f"[verify]   M={s['M']} K={s['K']} N={s['N']}: "
                f"cos={r['correctness']['cosine']:.4f} "
                f"speedup={r['speedup_vs_bf16']:.2f}× "
                f"fp8out_scaled_mm_cos={fp8out_cos:.4f}",
                flush=True,
            )

    if args.out:
        Path(args.out).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(f"[verify] wrote {args.out}", flush=True)

    return 0 if overall_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
