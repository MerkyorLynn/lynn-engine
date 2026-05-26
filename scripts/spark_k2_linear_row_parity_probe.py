#!/usr/bin/env python3
"""Probe batched-vs-rowwise Linear parity for K=2 verifier work.

The Qwen35 K2 verifier sweep isolated one full-attention drift source to
``o_proj``: ``F.linear(x[:, :2], W)`` is not always identical to two separate
``F.linear(x[:, i:i+1], W)`` calls. This tiny probe reproduces that class of
drift without loading a model, so future kernel work can validate row-wise
accumulation behavior cheaply before running the 35B smoke.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


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


def _run_case(
    *,
    seed: int,
    hidden: int,
    out_features: int,
    dtype: torch.dtype,
    device: str,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    x = torch.randn(1, 2, hidden, device=device, dtype=dtype)
    w = torch.randn(out_features, hidden, device=device, dtype=dtype)

    # Warm both dispatch shapes so timing does not include CUDA/cuBLAS setup.
    _ = F.linear(x, w)
    _ = torch.cat(
        [
            F.linear(x[:, idx:idx + 1, :].contiguous(), w)
            for idx in range(2)
        ],
        dim=1,
    )
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    t0 = time.time()
    batched = F.linear(x, w)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    batched_ms = (time.time() - t0) * 1000.0

    if device.startswith("cuda"):
        torch.cuda.synchronize()
    t0 = time.time()
    rowwise = torch.cat(
        [
            F.linear(x[:, idx:idx + 1, :].contiguous(), w)
            for idx in range(2)
        ],
        dim=1,
    )
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    rowwise_ms = (time.time() - t0) * 1000.0

    return {
        "seed": seed,
        "hidden": hidden,
        "out_features": out_features,
        "dtype": str(dtype).replace("torch.", ""),
        "device": device,
        "batched_ms": batched_ms,
        "rowwise_ms": rowwise_ms,
        "cmp": _cmp(rowwise, batched),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hidden", type=int, default=4096)
    ap.add_argument("--out-features", type=int, default=4096)
    ap.add_argument("--seeds", type=int, default=16)
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
        _run_case(
            seed=seed,
            hidden=args.hidden,
            out_features=args.out_features,
            dtype=dtype,
            device=args.device,
        )
        for seed in range(args.seeds)
    ]
    drift_rows = [row for row in rows if not row["cmp"]["equal"]]
    worst = max(rows, key=lambda row: row["cmp"]["max_abs"]) if rows else None
    report = {
        "schema_version": "lynn-k2-linear-row-parity-probe-v1",
        "rows": rows,
        "summary": {
            "n": len(rows),
            "drift_count": len(drift_rows),
            "drift_rate": len(drift_rows) / len(rows) if rows else None,
            "worst": worst,
            "mean_batched_ms": sum(row["batched_ms"] for row in rows) / len(rows) if rows else None,
            "mean_rowwise_ms": sum(row["rowwise_ms"] for row in rows) / len(rows) if rows else None,
        },
    }

    print(json.dumps(report["summary"], indent=2))
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"[linear-row-parity] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
