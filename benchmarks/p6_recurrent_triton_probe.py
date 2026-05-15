#!/usr/bin/env python3
"""P6-C: Triton fused recurrent gated-delta single-token probe.

This probes the true hot segment found by P6-A. It keeps the numerically
sensitive l2norm/transpose preparation in PyTorch, then fuses:

  S_decay, kv_mem, delta, S_new, out

for one decode token into a Triton kernel. The probe is correctness-first and
does not change engine defaults.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import torch

try:
    import triton
    import triton.language as tl
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("Triton is required for P6 recurrent probe") from exc

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.incremental_decode import _recurrent_gated_delta_rule  # noqa: E402
from engine.qwen36_linear_attn_block import HEAD_K_DIM, HEAD_V_DIM, NUM_V_HEADS, l2norm  # noqa: E402


@triton.jit
def _recurrent_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    g_ptr,
    beta_ptr,
    s_prev_ptr,
    out_ptr,
    s_new_ptr,
    BLOCK_K: tl.constexpr,
    BLOCK_V: tl.constexpr,
    HEAD_V: tl.constexpr,
):
    head = tl.program_id(0)
    v_block = tl.program_id(1)
    offs_k = tl.arange(0, BLOCK_K)
    offs_v = v_block * BLOCK_V + tl.arange(0, BLOCK_V)

    q = tl.load(q_ptr + head * BLOCK_K + offs_k).to(tl.float32)
    k = tl.load(k_ptr + head * BLOCK_K + offs_k).to(tl.float32)
    v = tl.load(v_ptr + head * HEAD_V + offs_v).to(tl.float32)
    g_exp = tl.exp(tl.load(g_ptr + head).to(tl.float32))
    beta = tl.load(beta_ptr + head).to(tl.float32)

    s_offsets = head * BLOCK_K * HEAD_V + offs_k[:, None] * HEAD_V + offs_v[None, :]
    s_prev = tl.load(s_prev_ptr + s_offsets).to(tl.float32)
    s_decay = s_prev * g_exp
    kv_mem = tl.sum(s_decay * k[:, None], axis=0)
    delta = (v - kv_mem) * beta
    s_new = s_decay + k[:, None] * delta[None, :]
    out = tl.sum(s_new * q[:, None], axis=0)
    tl.store(s_new_ptr + s_offsets, s_new)
    tl.store(out_ptr + head * HEAD_V + offs_v, out)


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


def _prepare(q, k, v, g, beta):
    initial_dtype = q.dtype
    q = l2norm(q, dim=-1, eps=1e-6)
    k = l2norm(k, dim=-1, eps=1e-6)
    q = q.transpose(1, 2).contiguous().to(torch.float32).squeeze(2)
    k = k.transpose(1, 2).contiguous().to(torch.float32).squeeze(2)
    v = v.transpose(1, 2).contiguous().to(torch.float32).squeeze(2)
    g = g.transpose(1, 2).contiguous().to(torch.float32).squeeze(2)
    beta = beta.transpose(1, 2).contiguous().to(torch.float32).squeeze(2)
    q = q * (1.0 / math.sqrt(q.shape[-1]))
    return initial_dtype, q.reshape(NUM_V_HEADS, HEAD_K_DIM), k.reshape(NUM_V_HEADS, HEAD_K_DIM), v.reshape(NUM_V_HEADS, HEAD_V_DIM), g.reshape(NUM_V_HEADS), beta.reshape(NUM_V_HEADS)


def recurrent_triton_prepared(q2, k2, v2, g2, beta2, s_prev):
    out = torch.empty((NUM_V_HEADS, HEAD_V_DIM), device=q2.device, dtype=torch.float32)
    s_new = torch.empty_like(s_prev)
    _recurrent_kernel[(NUM_V_HEADS, 4)](
        q2,
        k2,
        v2,
        g2,
        beta2,
        s_prev.reshape(NUM_V_HEADS, HEAD_K_DIM, HEAD_V_DIM),
        out,
        s_new.reshape(NUM_V_HEADS, HEAD_K_DIM, HEAD_V_DIM),
        BLOCK_K=HEAD_K_DIM,
        BLOCK_V=32,
        HEAD_V=HEAD_V_DIM,
        num_warps=4,
    )
    return out.reshape(1, 1, NUM_V_HEADS, HEAD_V_DIM), s_new.reshape(1, NUM_V_HEADS, HEAD_K_DIM, HEAD_V_DIM)


def recurrent_triton_full(q, k, v, g, beta, s_prev):
    initial_dtype, q2, k2, v2, g2, beta2 = _prepare(q, k, v, g, beta)
    out, s_new = recurrent_triton_prepared(q2, k2, v2, g2, beta2, s_prev)
    return out.to(initial_dtype), s_new


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    ap.add_argument("--iters", type=int, default=800)
    ap.add_argument("--warmup", type=int, default=100)
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

    initial_dtype, q2, k2, v2, g2, beta2 = _prepare(q, k, v, g, beta)
    ref_out, ref_s = _recurrent_gated_delta_rule(q, k, v, g, beta, s_prev)
    tri_out, tri_s = recurrent_triton_prepared(q2, k2, v2, g2, beta2, s_prev)
    tri_out = tri_out.to(initial_dtype)

    result = {
        "schema_version": "lynn-engine-p6-recurrent-triton-probe-v1",
        "device": torch.cuda.get_device_name(args.device),
        "comparisons": {
            "triton_vs_ref": {"out": _cmp(tri_out, ref_out), "state": _cmp(tri_s, ref_s)}
        },
        "latency_ms": {
            "ref_full": _bench(lambda: _recurrent_gated_delta_rule(q, k, v, g, beta, s_prev), args.warmup, args.iters),
            "triton_prepared_kernel": _bench(lambda: recurrent_triton_prepared(q2, k2, v2, g2, beta2, s_prev), args.warmup, args.iters),
            "triton_full_with_prepare": _bench(lambda: recurrent_triton_full(q, k, v, g, beta, s_prev), args.warmup, args.iters),
        },
    }
    result["derived"] = {
        "triton_prepared_vs_ref_speed_ratio": result["latency_ms"]["ref_full"] / result["latency_ms"]["triton_prepared_kernel"],
        "triton_full_vs_ref_speed_ratio": result["latency_ms"]["ref_full"] / result["latency_ms"]["triton_full_with_prepare"],
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
