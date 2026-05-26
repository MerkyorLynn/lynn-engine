#!/usr/bin/env python3
"""Probe batched-vs-rowwise prefix-causal attention parity for K=2.

The Qwen35 K2 verifier has a second full-attention drift source besides
``o_proj``: batched prefix-causal attention. This fixture compares the exact
shape used by ``decode_full_attn_k2`` against two separate T=1 SDPA calls
without loading a model.
"""
from __future__ import annotations

import argparse
import json
import math
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


def _batched_prefix_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    _, h_q, _, _ = q.shape
    _, h_kv, n, _ = k.shape
    causal_mask = torch.zeros(2, n, dtype=torch.bool, device=q.device)
    causal_mask[0, : n - 1] = True
    causal_mask[1, : n] = True
    attn_mask = torch.zeros(2, n, dtype=q.dtype, device=q.device)
    attn_mask.masked_fill_(~causal_mask, float("-inf"))
    attn_mask = attn_mask.view(1, 1, 2, n)
    return F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=attn_mask,
        is_causal=False,
        enable_gqa=(h_kv != h_q),
    )


def _rowwise_prefix_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    _, h_q, _, _ = q.shape
    _, h_kv, n, _ = k.shape
    pieces = []
    for idx, end in enumerate((n - 1, n)):
        pieces.append(
            F.scaled_dot_product_attention(
                q[:, :, idx:idx + 1, :].contiguous(),
                k[:, :, :end, :],
                v[:, :, :end, :],
                is_causal=False,
                enable_gqa=(h_kv != h_q),
            )
        )
    return torch.cat(pieces, dim=2)


def _manual_batched_prefix_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Manual GQA attention used as a debugging reference, not a fast path."""
    b, h_q, m, d = q.shape
    _, h_kv, n, _ = k.shape
    group = h_q // h_kv
    q_grouped = q.view(b, h_kv, group, m, d)
    scores = torch.einsum("bhgmd,bhnd->bhgmn", q_grouped.float(), k.float()) / math.sqrt(d)
    causal_mask = torch.zeros(m, n, dtype=torch.bool, device=q.device)
    causal_mask[0, : n - 1] = True
    causal_mask[1, : n] = True
    scores = scores.masked_fill(~causal_mask.view(1, 1, 1, m, n), float("-inf"))
    probs = torch.softmax(scores, dim=-1).to(v.dtype)
    out = torch.einsum("bhgmn,bhnd->bhgmd", probs, v)
    return out.reshape(b, h_q, m, d)


def _run_case(
    *,
    seed: int,
    h_q: int,
    h_kv: int,
    n: int,
    head_dim: int,
    dtype: torch.dtype,
    device: str,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    q = torch.randn(1, h_q, 2, head_dim, device=device, dtype=dtype)
    k = torch.randn(1, h_kv, n, head_dim, device=device, dtype=dtype)
    v = torch.randn(1, h_kv, n, head_dim, device=device, dtype=dtype)

    _ = _batched_prefix_attention(q, k, v)
    _ = _rowwise_prefix_attention(q, k, v)
    if device.startswith("cuda"):
        torch.cuda.synchronize()

    t0 = time.time()
    batched = _batched_prefix_attention(q, k, v)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    batched_ms = (time.time() - t0) * 1000.0

    t0 = time.time()
    rowwise = _rowwise_prefix_attention(q, k, v)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    rowwise_ms = (time.time() - t0) * 1000.0

    manual = _manual_batched_prefix_attention(q, k, v)
    if device.startswith("cuda"):
        torch.cuda.synchronize()

    return {
        "seed": seed,
        "h_q": h_q,
        "h_kv": h_kv,
        "n": n,
        "head_dim": head_dim,
        "dtype": str(dtype).replace("torch.", ""),
        "device": device,
        "batched_ms": batched_ms,
        "rowwise_ms": rowwise_ms,
        "rowwise_vs_batched": _cmp(rowwise, batched),
        "rowwise_vs_manual": _cmp(rowwise, manual),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--h-q", type=int, default=32)
    ap.add_argument("--h-kv", type=int, default=4)
    ap.add_argument("--n", type=int, default=2048)
    ap.add_argument("--head-dim", type=int, default=128)
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
            h_q=args.h_q,
            h_kv=args.h_kv,
            n=args.n,
            head_dim=args.head_dim,
            dtype=dtype,
            device=args.device,
        )
        for seed in range(args.seeds)
    ]
    drift_rows = [row for row in rows if not row["rowwise_vs_batched"]["equal"]]
    worst = max(rows, key=lambda row: row["rowwise_vs_batched"]["max_abs"]) if rows else None
    report = {
        "schema_version": "lynn-k2-attention-row-parity-probe-v1",
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
        print(f"[attention-row-parity] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
