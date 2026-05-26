#!/usr/bin/env python3
"""Validate the experimental rowwise-linear K=2 Triton kernel.

This compares:

* one kernel call with ``M=2``;
* two kernel calls with ``M=1`` concatenated;
* PyTorch batched and row-wise ``F.linear`` references.

The promotion gate for using this shape in the K2 verifier is that the one
``M=2`` kernel call must be bit-equal to two ``M=1`` kernel calls.
"""
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

from triton_kernels.rowwise_linear import rowwise_linear  # noqa: E402


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


def _time_cuda(fn):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.time()
    out = fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return out, (time.time() - t0) * 1000.0


def _run_case(seed: int, hidden: int, out_features: int, dtype: torch.dtype, device: str) -> dict[str, Any]:
    torch.manual_seed(seed)
    x = torch.randn(2, hidden, device=device, dtype=dtype)
    w = torch.randn(out_features, hidden, device=device, dtype=dtype)

    # Warmup.
    _ = rowwise_linear(x, w)
    _ = rowwise_linear(x[0:1].contiguous(), w)
    _ = F.linear(x, w)
    if device.startswith("cuda"):
        torch.cuda.synchronize()

    k2, k2_ms = _time_cuda(lambda: rowwise_linear(x, w))
    t1, t1_ms = _time_cuda(
        lambda: torch.cat(
            [
                rowwise_linear(x[idx:idx + 1].contiguous(), w)
                for idx in range(2)
            ],
            dim=0,
        )
    )
    torch_batched, torch_batched_ms = _time_cuda(lambda: F.linear(x, w))
    torch_t1, torch_t1_ms = _time_cuda(
        lambda: torch.cat(
            [
                F.linear(x[idx:idx + 1].contiguous(), w)
                for idx in range(2)
            ],
            dim=0,
        )
    )

    return {
        "seed": seed,
        "hidden": hidden,
        "out_features": out_features,
        "dtype": str(dtype).replace("torch.", ""),
        "device": device,
        "kernel_k2_ms": k2_ms,
        "kernel_t1x2_ms": t1_ms,
        "torch_batched_ms": torch_batched_ms,
        "torch_t1x2_ms": torch_t1_ms,
        "kernel_k2_vs_kernel_t1x2": _cmp(t1, k2),
        "kernel_k2_vs_torch_t1x2": _cmp(torch_t1, k2),
        "torch_batched_vs_torch_t1x2": _cmp(torch_t1, torch_batched),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hidden", type=int, default=4096)
    ap.add_argument("--out-features", type=int, default=4096)
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
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
        _run_case(seed, args.hidden, args.out_features, dtype, args.device)
        for seed in range(args.seeds)
    ]
    exact = [row for row in rows if row["kernel_k2_vs_kernel_t1x2"]["equal"]]
    report = {
        "schema_version": "lynn-k2-rowwise-linear-kernel-probe-v1",
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
            "mean_torch_batched_ms": sum(row["torch_batched_ms"] for row in rows) / len(rows) if rows else None,
            "mean_torch_t1x2_ms": sum(row["torch_t1x2_ms"] for row in rows) / len(rows) if rows else None,
        },
    }
    print(json.dumps(report["summary"], indent=2))
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"[rowwise-linear-kernel] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
