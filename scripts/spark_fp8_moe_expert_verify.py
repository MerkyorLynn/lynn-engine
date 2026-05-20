#!/usr/bin/env python3
"""Verify + benchmark the Spark FP8 MoE expert fused gate/up kernel.

Representative MoE expert shapes are small-M routed activation slices with
K=2048 and expert intermediate N in {1408, 768}.  This harness compares:

* correctness vs BF16 reference SwiGLU for one expert
* perf vs BF16 reference
* perf vs the dense fused FP8 kernel from spark_fp8_gate_up_fused.py

Usage on Spark::

    /home/merkyor/comfyui/ComfyUI/.venv/bin/python -u \
        scripts/spark_fp8_moe_expert_verify.py --bench-iters 200
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.spark_pack_w4a8_fp8 import (  # type: ignore  # noqa: E402
    repack_nvfp4_to_fp8,
    synthetic_nvfp4,
)
from triton_kernels.spark_fp8_gate_up_fused import fp8_gate_up_silu_fused  # noqa: E402
from triton_kernels.spark_fp8_moe_expert_fused import (  # noqa: E402
    fp8_activation_scale,
    fp8_moe_expert_gate_up_silu_fused,
    fp8_moe_expert_gate_up_silu_reference,
)


def _default_shapes() -> list[str]:
    return [f"{m}x2048x{n}" for n in (1408, 768) for m in range(1, 9)]


def _diff_stats(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    af = a.detach().float().flatten().double()
    bf = b.detach().float().flatten().double()
    diff = af - bf
    max_abs = float(diff.abs().max().item())
    a_norm = float(af.norm().item())
    b_norm = float(bf.norm().item())
    diff_norm = float(diff.norm().item())
    rel_l2 = float(diff_norm / a_norm) if a_norm > 0 else float("nan")
    cos = float((af * bf).sum().item()) / (a_norm * b_norm + 1e-12)
    return {"max_abs": max_abs, "rel_l2": rel_l2, "cosine": cos}


def benchmark_call(fn, iters: int, warmup: int = 20, repeats: int = 3) -> dict[str, float]:
    best: dict[str, float] | None = None
    for _ in range(repeats):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        per_iter_us = elapsed * 1e6 / iters
        result = {"elapsed_seconds": elapsed, "iters": iters, "per_iter_us": per_iter_us}
        if best is None or result["per_iter_us"] < best["per_iter_us"]:
            best = result
    assert best is not None
    best["repeats"] = repeats
    return best


def _make_weights(n: int, k: int, device: str) -> dict[str, torch.Tensor]:
    gate_packed_cpu, gate_scale_cpu = synthetic_nvfp4(n, k, seed=1234 + n)
    up_packed_cpu, up_scale_cpu = synthetic_nvfp4(n, k, seed=5678 + n)

    gate_repacked = repack_nvfp4_to_fp8(gate_packed_cpu, gate_scale_cpu, None, scale_granularity="per_row")
    up_repacked = repack_nvfp4_to_fp8(up_packed_cpu, up_scale_cpu, None, scale_granularity="per_row")

    w_gate_fp8 = gate_repacked.fp8_weight.to(device)
    w_up_fp8 = up_repacked.fp8_weight.to(device)
    w_gate_scale = gate_repacked.fp8_scale.to(device)
    w_up_scale = up_repacked.fp8_scale.to(device)

    w_gate_bf16 = (w_gate_fp8.to(torch.float32) * w_gate_scale[:, None]).to(torch.bfloat16)
    w_up_bf16 = (w_up_fp8.to(torch.float32) * w_up_scale[:, None]).to(torch.bfloat16)

    return {
        "w_gate_fp8": w_gate_fp8,
        "w_up_fp8": w_up_fp8,
        "w_gate_scale": w_gate_scale,
        "w_up_scale": w_up_scale,
        "w_gate_bf16": w_gate_bf16,
        "w_up_bf16": w_up_bf16,
    }


def run_shape(
    m: int,
    k: int,
    n: int,
    *,
    bench_iters: int,
    device: str,
    weight_cache: dict[tuple[int, int], dict[str, torch.Tensor]],
    require_speedup: bool,
) -> dict:
    print(f"[moe-verify] shape M={m} K={k} N={n}", flush=True)
    torch.manual_seed(1000 + m + n)
    x_bf16 = (torch.randn((m, k), dtype=torch.bfloat16, device=device) * 0.5).contiguous()
    x_scale = fp8_activation_scale(x_bf16)

    weight_key = (n, k)
    weights = weight_cache.get(weight_key)
    if weights is None:
        weights = _make_weights(n, k, device)
        weight_cache[weight_key] = weights

    fused_out = fp8_moe_expert_gate_up_silu_fused(
        x_bf16,
        weights["w_gate_fp8"],
        weights["w_up_fp8"],
        weights["w_gate_scale"],
        weights["w_up_scale"],
        x_scale=x_scale,
    )
    ref_out = fp8_moe_expert_gate_up_silu_reference(
        x_bf16,
        weights["w_gate_bf16"],
        weights["w_up_bf16"],
    )
    diff = _diff_stats(fused_out, ref_out)
    correctness_ok = diff["cosine"] > 0.99
    print(
        f"[moe-verify]   correctness: cos={diff['cosine']:.6f} "
        f"rel_l2={diff['rel_l2']:.4e} max_abs={diff['max_abs']:.4e}",
        flush=True,
    )

    bench_moe_fp8 = benchmark_call(
        lambda: fp8_moe_expert_gate_up_silu_fused(
            x_bf16,
            weights["w_gate_fp8"],
            weights["w_up_fp8"],
            weights["w_gate_scale"],
            weights["w_up_scale"],
            x_scale=x_scale,
        ),
        iters=bench_iters,
    )
    bench_bf16 = benchmark_call(
        lambda: fp8_moe_expert_gate_up_silu_reference(
            x_bf16,
            weights["w_gate_bf16"],
            weights["w_up_bf16"],
        ),
        iters=bench_iters,
    )
    bench_dense_fp8 = benchmark_call(
        lambda: fp8_gate_up_silu_fused(
            x_bf16,
            weights["w_gate_fp8"],
            weights["w_up_fp8"],
            weights["w_gate_scale"],
            weights["w_up_scale"],
        ),
        iters=bench_iters,
    )

    speedup_vs_bf16 = bench_bf16["per_iter_us"] / bench_moe_fp8["per_iter_us"]
    speedup_vs_dense = bench_dense_fp8["per_iter_us"] / bench_moe_fp8["per_iter_us"]
    perf_ok = (not require_speedup) or speedup_vs_bf16 > 1.0
    print(
        f"[moe-verify]   bench: moe_FP8 {bench_moe_fp8['per_iter_us']:.1f}us  "
        f"BF16_ref {bench_bf16['per_iter_us']:.1f}us  "
        f"dense_FP8 {bench_dense_fp8['per_iter_us']:.1f}us  "
        f"speedup_vs_bf16={speedup_vs_bf16:.2f}x  "
        f"speedup_vs_dense={speedup_vs_dense:.2f}x",
        flush=True,
    )

    return {
        "shape": {"M": m, "K": k, "N": n},
        "correctness": diff,
        "correctness_ok": correctness_ok,
        "bench_moe_fp8_us": bench_moe_fp8["per_iter_us"],
        "bench_bf16_ref_us": bench_bf16["per_iter_us"],
        "bench_dense_fp8_us": bench_dense_fp8["per_iter_us"],
        "speedup_vs_bf16": speedup_vs_bf16,
        "speedup_vs_dense_fused": speedup_vs_dense,
        "perf_ok": perf_ok,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--shapes",
        nargs="+",
        default=_default_shapes(),
        help="MxKxN shapes to test (default: M=1..8, K=2048, N=1408 and 768)",
    )
    ap.add_argument("--bench-iters", type=int, default=200)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None, help="optional JSON output path")
    ap.add_argument(
        "--no-require-speedup",
        action="store_true",
        help="do not fail the run when speedup_vs_bf16 is <= 1.0",
    )
    args = ap.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA required")

    weight_cache: dict[tuple[int, int], dict[str, torch.Tensor]] = {}
    results = []
    overall_ok = True
    for spec in args.shapes:
        try:
            m, k, n = (int(x) for x in spec.split("x"))
        except ValueError:
            print(f"bad shape spec {spec!r}, expected MxKxN", file=sys.stderr)
            return 2
        try:
            r = run_shape(
                m,
                k,
                n,
                bench_iters=args.bench_iters,
                device=args.device,
                weight_cache=weight_cache,
                require_speedup=not args.no_require_speedup,
            )
            results.append(r)
            if not (r["correctness_ok"] and r["perf_ok"]):
                overall_ok = False
        except Exception as exc:  # noqa: BLE001
            print(f"[moe-verify]   shape {spec!r} FAILED: {exc!r}", flush=True)
            results.append({"shape_spec": spec, "error": repr(exc)})
            overall_ok = False

    summary = {
        "schema_version": "lynn-spark-fp8-moe-expert-fused-verify-v1",
        "results": results,
        "all_ok": overall_ok,
    }
    print("[moe-verify] === summary ===", flush=True)
    for r in results:
        if "error" in r:
            print(f"[moe-verify]   {r.get('shape_spec')}: ERROR {r['error']}", flush=True)
        else:
            s = r["shape"]
            print(
                f"[moe-verify]   M={s['M']} K={s['K']} N={s['N']}: "
                f"cos={r['correctness']['cosine']:.4f} "
                f"speedup_vs_bf16={r['speedup_vs_bf16']:.2f}x "
                f"speedup_vs_dense={r['speedup_vs_dense_fused']:.2f}x",
                flush=True,
            )

    if args.out:
        Path(args.out).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(f"[moe-verify] wrote {args.out}", flush=True)

    return 0 if overall_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
