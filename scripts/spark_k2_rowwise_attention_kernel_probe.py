#!/usr/bin/env python3
"""Validate the experimental rowwise-prefix-attention K=2 Triton kernel."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from triton_kernels.rowwise_attention import rowwise_prefix_attention  # noqa: E402


def _cmp(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    af = a.detach().float().reshape(-1)
    bf = b.detach().float().reshape(-1)
    diff = af - bf
    denom = torch.linalg.vector_norm(af).clamp_min(1e-12)
    return {
        "max_abs": float(diff.abs().max().item()),
        "mean_abs": float(diff.abs().mean().item()),
        "rel_l2": float((torch.linalg.vector_norm(diff) / denom).item()),
        "cosine": float(F.cosine_similarity(af, bf, dim=0).item()),
        "equal": bool(torch.equal(a, b)),
    }


def _time_cuda(fn, *, repeats: int):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.time()
    out = None
    for _ in range(repeats):
        out = fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return out, (time.time() - t0) * 1000.0 / repeats


def _run_case(
    *,
    seed: int,
    h_q: int,
    h_kv: int,
    n: int,
    head_dim: int,
    dtype: torch.dtype,
    device: str,
    repeats: int,
    warmup_iters: int,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    q = torch.randn(1, h_q, 2, head_dim, device=device, dtype=dtype)
    k = torch.randn(1, h_kv, n, head_dim, device=device, dtype=dtype)
    v = torch.randn(1, h_kv, n, head_dim, device=device, dtype=dtype)

    def run_k2() -> torch.Tensor:
        return rowwise_prefix_attention(q, k, v)

    def run_t1x2() -> torch.Tensor:
        return torch.cat(
            [
                rowwise_prefix_attention(q[:, :, 0:1, :].contiguous(), k[:, :, : n - 1, :], v[:, :, : n - 1, :]),
                rowwise_prefix_attention(q[:, :, 1:2, :].contiguous(), k, v),
            ],
            dim=2,
        )

    # Warm all JIT specializations and CUDA allocator paths before timing.
    for _ in range(warmup_iters):
        _ = run_k2()
        _ = run_t1x2()
    if device.startswith("cuda"):
        torch.cuda.synchronize()

    k2, k2_ms = _time_cuda(run_k2, repeats=repeats)
    t1, t1_ms = _time_cuda(run_t1x2, repeats=repeats)
    return {
        "seed": seed,
        "h_q": h_q,
        "h_kv": h_kv,
        "n": n,
        "head_dim": head_dim,
        "dtype": str(dtype).replace("torch.", ""),
        "device": device,
        "kernel_k2_ms": k2_ms,
        "kernel_t1x2_ms": t1_ms,
        "kernel_k2_vs_kernel_t1x2": _cmp(t1, k2),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--h-q", type=int, default=32)
    ap.add_argument("--h-kv", type=int, default=4)
    ap.add_argument("--n", type=int, default=2048)
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--warmup-iters", type=int, default=3)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is false")
    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.dtype]
    rows = [
        _run_case(
            seed=seed,
            h_q=args.h_q,
            h_kv=args.h_kv,
            n=args.n,
            head_dim=args.head_dim,
            dtype=dtype,
            device=args.device,
            repeats=args.repeats,
            warmup_iters=args.warmup_iters,
        )
        for seed in range(args.seeds)
    ]
    exact = [row for row in rows if row["kernel_k2_vs_kernel_t1x2"]["equal"]]
    report = {
        "schema_version": "lynn-k2-rowwise-attention-kernel-probe-v1",
        "rows": rows,
        "summary": {
            "n": len(rows),
            "kernel_k2_matches_kernel_t1x2_count": len(exact),
            "kernel_k2_matches_kernel_t1x2_rate": len(exact) / len(rows) if rows else None,
            "worst_kernel_k2_vs_kernel_t1x2": max(
                rows, key=lambda row: row["kernel_k2_vs_kernel_t1x2"]["max_abs"]
            ) if rows else None,
            "mean_kernel_k2_ms": sum(row["kernel_k2_ms"] for row in rows) / len(rows) if rows else None,
            "mean_kernel_t1x2_ms": sum(row["kernel_t1x2_ms"] for row in rows) / len(rows) if rows else None,
        },
    }
    print(json.dumps(report["summary"], indent=2))
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"[rowwise-attention-kernel] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
