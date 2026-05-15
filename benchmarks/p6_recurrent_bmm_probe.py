#!/usr/bin/env python3
"""P6-B: probe a BMM-form recurrent gated-delta decode step."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.incremental_decode import _recurrent_gated_delta_rule  # noqa: E402
from engine.qwen36_linear_attn_block import HEAD_K_DIM, HEAD_V_DIM, NUM_V_HEADS, l2norm  # noqa: E402


def _bench(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end) / iters)


def _cmp(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    af = a.float().flatten()
    bf = b.float().flatten()
    diff = af - bf
    denom = torch.linalg.vector_norm(bf).clamp_min(1e-12)
    cosine = torch.dot(af, bf) / (
        torch.linalg.vector_norm(af).clamp_min(1e-12)
        * torch.linalg.vector_norm(bf).clamp_min(1e-12)
    )
    return {
        "mean_abs": float(diff.abs().mean()),
        "max_abs": float(diff.abs().max()),
        "rel_l2": float(torch.linalg.vector_norm(diff) / denom),
        "cosine": float(cosine),
    }


def recurrent_bmm(q, k, v, g, beta, s_prev):
    """Equivalent single-token recurrent step using bmm for memory reads."""
    initial_dtype = q.dtype
    q = l2norm(q, dim=-1, eps=1e-6)
    k = l2norm(k, dim=-1, eps=1e-6)
    q = q.transpose(1, 2).contiguous().to(torch.float32).squeeze(2)
    k = k.transpose(1, 2).contiguous().to(torch.float32).squeeze(2)
    v = v.transpose(1, 2).contiguous().to(torch.float32).squeeze(2)
    g = g.transpose(1, 2).contiguous().to(torch.float32).squeeze(2)
    beta = beta.transpose(1, 2).contiguous().to(torch.float32).squeeze(2)
    q = q * (1.0 / math.sqrt(q.shape[-1]))

    bsz, heads, k_dim = q.shape
    v_dim = v.shape[-1]
    s = s_prev * g.exp().unsqueeze(-1).unsqueeze(-1)
    s2 = s.reshape(bsz * heads, k_dim, v_dim)
    k2 = k.reshape(bsz * heads, 1, k_dim)
    q2 = q.reshape(bsz * heads, 1, k_dim)
    kv_mem = torch.bmm(k2, s2).reshape(bsz, heads, v_dim)
    delta = (v - kv_mem) * beta.unsqueeze(-1)
    s_new = s + k.unsqueeze(-1) * delta.unsqueeze(-2)
    out = torch.bmm(q2, s_new.reshape(bsz * heads, k_dim, v_dim)).reshape(bsz, heads, v_dim)
    return out.unsqueeze(1).to(initial_dtype), s_new


def recurrent_bmm_outer(q, k, v, g, beta, s_prev):
    """BMM for reads plus baddbmm-style outer update."""
    initial_dtype = q.dtype
    q = l2norm(q, dim=-1, eps=1e-6)
    k = l2norm(k, dim=-1, eps=1e-6)
    q = q.transpose(1, 2).contiguous().to(torch.float32).squeeze(2)
    k = k.transpose(1, 2).contiguous().to(torch.float32).squeeze(2)
    v = v.transpose(1, 2).contiguous().to(torch.float32).squeeze(2)
    g = g.transpose(1, 2).contiguous().to(torch.float32).squeeze(2)
    beta = beta.transpose(1, 2).contiguous().to(torch.float32).squeeze(2)
    q = q * (1.0 / math.sqrt(q.shape[-1]))

    bsz, heads, k_dim = q.shape
    v_dim = v.shape[-1]
    s = s_prev * g.exp().unsqueeze(-1).unsqueeze(-1)
    s2 = s.reshape(bsz * heads, k_dim, v_dim)
    k2 = k.reshape(bsz * heads, 1, k_dim)
    q2 = q.reshape(bsz * heads, 1, k_dim)
    kv_mem = torch.bmm(k2, s2).reshape(bsz, heads, v_dim)
    delta = (v - kv_mem) * beta.unsqueeze(-1)
    s_new2 = torch.baddbmm(
        s2,
        k.reshape(bsz * heads, k_dim, 1),
        delta.reshape(bsz * heads, 1, v_dim),
    )
    out = torch.bmm(q2, s_new2).reshape(bsz, heads, v_dim)
    return out.unsqueeze(1).to(initial_dtype), s_new2.reshape(bsz, heads, k_dim, v_dim)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    ap.add_argument("--iters", type=int, default=500)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--seed", type=int, default=20260515)
    args = ap.parse_args()

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    gen = torch.Generator(device=args.device)
    gen.manual_seed(args.seed)
    q = torch.randn(1, 1, NUM_V_HEADS, HEAD_K_DIM, device=args.device, dtype=dtype, generator=gen)
    k = torch.randn(1, 1, NUM_V_HEADS, HEAD_K_DIM, device=args.device, dtype=dtype, generator=gen)
    v = torch.randn(1, 1, NUM_V_HEADS, HEAD_V_DIM, device=args.device, dtype=dtype, generator=gen)
    g = torch.randn(1, 1, NUM_V_HEADS, device=args.device, dtype=dtype, generator=gen) * 0.1
    beta = torch.randn(1, 1, NUM_V_HEADS, device=args.device, dtype=dtype, generator=gen).sigmoid()
    s_prev = torch.randn(1, NUM_V_HEADS, HEAD_K_DIM, HEAD_V_DIM, device=args.device, dtype=torch.float32, generator=gen) * 0.01

    ref_out, ref_s = _recurrent_gated_delta_rule(q, k, v, g, beta, s_prev)
    bmm_out, bmm_s = recurrent_bmm(q, k, v, g, beta, s_prev)
    outer_out, outer_s = recurrent_bmm_outer(q, k, v, g, beta, s_prev)

    result = {
        "schema_version": "lynn-engine-p6-recurrent-bmm-probe-v1",
        "device": torch.cuda.get_device_name(args.device),
        "comparisons": {
            "bmm_vs_ref": {"out": _cmp(bmm_out, ref_out), "state": _cmp(bmm_s, ref_s)},
            "bmm_outer_vs_ref": {"out": _cmp(outer_out, ref_out), "state": _cmp(outer_s, ref_s)},
        },
        "latency_ms": {
            "ref_elementwise": _bench(lambda: _recurrent_gated_delta_rule(q, k, v, g, beta, s_prev), args.warmup, args.iters),
            "bmm": _bench(lambda: recurrent_bmm(q, k, v, g, beta, s_prev), args.warmup, args.iters),
            "bmm_outer": _bench(lambda: recurrent_bmm_outer(q, k, v, g, beta, s_prev), args.warmup, args.iters),
        },
    }
    result["derived"] = {
        "bmm_vs_ref_speed_ratio": result["latency_ms"]["ref_elementwise"] / result["latency_ms"]["bmm"],
        "bmm_outer_vs_ref_speed_ratio": result["latency_ms"]["ref_elementwise"] / result["latency_ms"]["bmm_outer"],
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
