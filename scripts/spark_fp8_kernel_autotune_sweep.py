#!/usr/bin/env python3
"""Autotune sweep for the Spark sm_121 FP8 fused gate/up + SwiGLU kernel.

Phase 2 step 2.2 — exhaustive (BLOCK_M, BLOCK_K, BLOCK_N) sweep across
representative decode shapes to find the optimal tile size per shape class.

For each (M, K, N) × (BLOCK_M, BLOCK_K, BLOCK_N):
  1. Skip if estimated shared-memory > 228 KB (Spark sm_121 limit).
  2. Run correctness check (cos vs BF16 reference); skip failing configs.
  3. Benchmark (warmup 5, run 50 iterations).
  4. Record best config per shape.

Usage on Spark::

    /home/merkyor/comfyui/ComfyUI/.venv/bin/python -u \
        scripts/spark_fp8_kernel_autotune_sweep.py
"""
from __future__ import annotations

import argparse
import itertools
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
from triton_kernels.spark_fp8_gate_up_fused import (  # noqa: E402
    fp8_gate_up_silu_fused,
    fp8_gate_up_silu_reference,
)

# ── sweep grids ──────────────────────────────────────────────────────────
M_GRID = [1, 4, 8, 16]
K_GRID = [2048, 4096, 6144]
N_GRID = [256, 768, 1408, 2048, 6144]

BLOCK_M_CANDIDATES = [16, 32, 64]
BLOCK_K_CANDIDATES = [32, 64, 128]
BLOCK_N_CANDIDATES = [32, 64, 128, 256]

# Spark sm_121 shared memory ≈ 228 KB.
SMEM_LIMIT_BYTES = 228 * 1024


def _estimate_smem(block_m: int, block_k: int, block_n: int) -> int:
    """Conservative shared-memory estimate (bytes) for one K-loop iteration.

    Accounts for: x [BLOCK_M, BLOCK_K] + w_gate [BLOCK_N, BLOCK_K] +
    w_up [BLOCK_N, BLOCK_K] — all FP8 (1 B/elem) — plus accumulators
    and scale buffers.
    """
    # Activation tile + two weight tiles (FP8 = 1 byte)
    tile_bytes = block_m * block_k + 2 * block_n * block_k
    # Accumulators: 2 × BLOCK_M × BLOCK_N × 4 (F32)
    acc_bytes = 2 * block_m * block_n * 4
    # Scale buffers: (BLOCK_M + 2×BLOCK_N) × 4 + BLOCK_M × 4
    scale_bytes = (2 * block_m + 2 * block_n) * 4
    return tile_bytes + acc_bytes + scale_bytes


def _diff_stats(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    af = a.detach().float().flatten().double()
    bf = b.detach().float().flatten().double()
    diff = af - bf
    a_norm = float(af.norm().item())
    diff_norm = float(diff.norm().item())
    cos = float((af * bf).sum().item()) / (a_norm * float(bf.norm().item()) + 1e-12)
    return {"cosine": cos}


def _benchmark(fn, warmup: int, iters: int) -> float:
    """Return per-iteration time in microseconds."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1e6 / iters


def run_one(
    m: int,
    k: int,
    n: int,
    block_m: int,
    block_k: int,
    block_n: int,
    *,
    warmup: int,
    iters: int,
    device: str = "cuda",
) -> dict:
    """Run correctness + benchmark for one (shape, block-config) combo."""
    torch.manual_seed(0)
    x_bf16 = torch.randn((m, k), dtype=torch.bfloat16, device=device) * 0.5

    gate_packed, gate_scale = synthetic_nvfp4(n, k, seed=1234)
    up_packed, up_scale = synthetic_nvfp4(n, k, seed=5678)

    gate_repacked = repack_nvfp4_to_fp8(gate_packed, gate_scale, None, scale_granularity="per_row")
    up_repacked = repack_nvfp4_to_fp8(up_packed, up_scale, None, scale_granularity="per_row")

    w_gate_fp8 = gate_repacked.fp8_weight.to(device)
    w_up_fp8 = up_repacked.fp8_weight.to(device)
    w_gate_s = gate_repacked.fp8_scale.to(device)
    w_up_s = up_repacked.fp8_scale.to(device)

    w_gate_bf16 = (w_gate_fp8.to(torch.float32) * w_gate_s[:, None]).to(torch.bfloat16)
    w_up_bf16 = (w_up_fp8.to(torch.float32) * w_up_s[:, None]).to(torch.bfloat16)

    # Correctness
    fused = fp8_gate_up_silu_fused(
        x_bf16, w_gate_fp8, w_up_fp8, w_gate_s, w_up_s,
        block_m=block_m, block_k=block_k, block_n=block_n,
    )
    ref = fp8_gate_up_silu_reference(x_bf16, w_gate_bf16, w_up_bf16)
    diff = _diff_stats(fused, ref)

    result: dict = {
        "cosine": diff["cosine"],
        "correctness_ok": diff["cosine"] > 0.99,
    }

    if not result["correctness_ok"]:
        return result

    # Benchmark FP8 fused
    fp8_us = _benchmark(
        lambda: fp8_gate_up_silu_fused(
            x_bf16, w_gate_fp8, w_up_fp8, w_gate_s, w_up_s,
            block_m=block_m, block_k=block_k, block_n=block_n,
        ),
        warmup=warmup,
        iters=iters,
    )
    # Benchmark BF16 reference
    bf16_us = _benchmark(
        lambda: fp8_gate_up_silu_reference(x_bf16, w_gate_bf16, w_up_bf16),
        warmup=warmup,
        iters=iters,
    )

    result["fp8_us"] = fp8_us
    result["bf16_us"] = bf16_us
    result["speedup"] = bf16_us / fp8_us if fp8_us > 0 else 0.0
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--bench-iters", type=int, default=50)
    ap.add_argument("--out", default=None, help="JSON output path (auto-generated if omitted)")
    ap.add_argument(
        "--smem-limit", type=int, default=SMEM_LIMIT_BYTES,
        help="Shared memory limit in bytes (default: 228 KiB)",
    )
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")

    dev_name = torch.cuda.get_device_name(0)
    print(f"[sweep] device={dev_name}", flush=True)
    print(f"[sweep] smem_limit={args.smem_limit} bytes", flush=True)

    shapes = list(itertools.product(M_GRID, K_GRID, N_GRID))
    block_configs = list(itertools.product(BLOCK_M_CANDIDATES, BLOCK_K_CANDIDATES, BLOCK_N_CANDIDATES))

    # Filter block configs by shared-memory limit
    valid_blocks = [
        (bm, bk, bn)
        for bm, bk, bn in block_configs
        if _estimate_smem(bm, bk, bn) <= args.smem_limit
    ]
    skipped_blocks = len(block_configs) - len(valid_blocks)
    total = len(shapes) * len(valid_blocks)
    print(
        f"[sweep] {len(shapes)} shapes × {len(valid_blocks)} block configs "
        f"({skipped_blocks} skipped by smem) = {total} combos",
        flush=True,
    )

    all_results: list[dict] = []
    best_per_shape: dict[str, dict] = {}
    t_start = time.time()

    for idx, ((m, k, n), (bm, bk, bn)) in enumerate(
        itertools.product(shapes, valid_blocks), 1
    ):
        shape_key = f"M{m}_K{k}_N{n}"
        print(
            f"[{idx}/{total}] shape=({m},{k},{n}) blocks=({bm},{bk},{bn}) ... ",
            end="",
            flush=True,
        )

        try:
            r = run_one(
                m, k, n, bm, bk, bn,
                warmup=args.warmup, iters=args.bench_iters,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"FAIL ({exc!r})", flush=True)
            all_results.append({
                "shape": {"M": m, "K": k, "N": n},
                "blocks": {"BLOCK_M": bm, "BLOCK_K": bk, "BLOCK_N": bn},
                "error": repr(exc),
            })
            continue

        entry = {
            "shape": {"M": m, "K": k, "N": n},
            "blocks": {"BLOCK_M": bm, "BLOCK_K": bk, "BLOCK_N": bn},
            "cosine": r["cosine"],
            "correctness_ok": r["correctness_ok"],
        }
        if r["correctness_ok"]:
            entry["fp8_us"] = r["fp8_us"]
            entry["bf16_us"] = r["bf16_us"]
            entry["speedup"] = r["speedup"]
            status = f"cos={r['cosine']:.4f} fp8={r['fp8_us']:.1f}us bf16={r['bf16_us']:.1f}us speedup={r['speedup']:.2f}x"
        else:
            status = f"FAIL cos={r['cosine']:.4f}"

        print(status, flush=True)
        all_results.append(entry)

        # Track best per shape
        if r["correctness_ok"]:
            cur = best_per_shape.get(shape_key)
            if cur is None or r["speedup"] > cur["speedup"]:
                best_per_shape[shape_key] = {
                    "shape": {"M": m, "K": k, "N": n},
                    "blocks": {"BLOCK_M": bm, "BLOCK_K": bk, "BLOCK_N": bn},
                    "cosine": r["cosine"],
                    "fp8_us": r["fp8_us"],
                    "bf16_us": r["bf16_us"],
                    "speedup": r["speedup"],
                }

    elapsed = time.time() - t_start

    # Build output
    summary = {
        "schema_version": "lynn-spark-fp8-autotune-sweep-v1",
        "device": dev_name,
        "smem_limit_bytes": args.smem_limit,
        "warmup": args.warmup,
        "bench_iters": args.bench_iters,
        "total_combos": total,
        "elapsed_seconds": round(elapsed, 1),
        "best_per_shape": best_per_shape,
        "all_results": all_results,
    }

    out_path = args.out
    if out_path is None:
        out_path = str(ROOT / "reports" / "mtp" / "spark_fp8_autotune_sweep_TS.json")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"\n[sweep] wrote {out_path}", flush=True)

    # Print best-per-shape table
    print("\n[sweep] ═══ best config per shape ═══", flush=True)
    for key in sorted(best_per_shape):
        b = best_per_shape[key]
        print(
            f"[sweep]   {key}: ({b['blocks']['BLOCK_M']},{b['blocks']['BLOCK_K']},{b['blocks']['BLOCK_N']})  "
            f"speedup={b['speedup']:.2f}×  fp8={b['fp8_us']:.1f}us  bf16={b['bf16_us']:.1f}us  cos={b['cosine']:.4f}",
            flush=True,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
