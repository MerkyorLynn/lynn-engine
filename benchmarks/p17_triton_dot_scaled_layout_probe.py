#!/usr/bin/env python3
"""P17: minimal Triton `tl.dot_scaled` FP4 layout probe.

This is intentionally synthetic. Before wiring Lynn expert tensors into a
grouped native-FP4 active kernel, first prove how Triton 3.6 expects packed
E2M1 operands to be shaped for `tl.dot_scaled`.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

import triton
import triton.language as tl


E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32)


@triton.jit
def _dot_scaled_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    K_PACKED: tl.constexpr,
    N: tl.constexpr,
    BLOCK_K_PACKED: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    block_n = tl.program_id(0)
    offs_n = block_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_kp = tl.arange(0, BLOCK_K_PACKED)
    out = tl.zeros((1, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K_PACKED, BLOCK_K_PACKED):
        k = k0 + offs_kp
        a = tl.load(a_ptr + k, mask=k < K_PACKED, other=0)
        b = tl.load(
            b_ptr + k[:, None] * N + offs_n[None, :],
            mask=(k[:, None] < K_PACKED) & (offs_n[None, :] < N),
            other=0,
        )
        # A is [1, K/2], B is [K/2, N]. Both are packed along K.
        out += tl.dot_scaled(
            a[None, :],
            None,
            "e2m1",
            b,
            None,
            "e2m1",
            lhs_k_pack=True,
            rhs_k_pack=True,
        )
    tl.store(c_ptr + offs_n, tl.reshape(out, (BLOCK_N,)), mask=offs_n < N)


def _pack_codes(codes: torch.Tensor) -> torch.Tensor:
    return (codes[..., 0::2] | (codes[..., 1::2] << 4)).contiguous()


def _unpack_fp4(packed: torch.Tensor, *, logical_k: int) -> torch.Tensor:
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    codes = torch.empty((*packed.shape[:-1], logical_k), device=packed.device, dtype=torch.uint8)
    codes[..., 0::2] = low
    codes[..., 1::2] = high
    mag = codes & 0x07
    sign = torch.where((codes & 0x08).bool(), -1.0, 1.0).to(packed.device)
    return E2M1.to(packed.device)[mag.long()] * sign


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=64)
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--block-n", type=int, default=128)
    ap.add_argument("--block-k-packed", type=int, default=256)
    args = ap.parse_args()
    if args.k % 2:
        raise ValueError("--k must be even")

    torch.manual_seed(args.seed)
    device = "cuda"
    # Random E2M1 codes including sign bit.
    a_codes = torch.randint(0, 16, (1, args.k), device=device, dtype=torch.uint8)
    # B logical is [K, N]. Pack along K into [K/2, N].
    b_codes = torch.randint(0, 16, (args.k, args.n), device=device, dtype=torch.uint8)
    a_packed = _pack_codes(a_codes)
    b_packed = (b_codes[0::2, :] | (b_codes[1::2, :] << 4)).contiguous()
    c = torch.empty((args.n,), device=device, dtype=torch.float32)

    def run_kernel() -> None:
        _dot_scaled_kernel[(triton.cdiv(args.n, args.block_n),)](
            a_packed,
            b_packed,
            c,
            K_PACKED=args.k // 2,
            N=args.n,
            BLOCK_K_PACKED=args.block_k_packed,
            BLOCK_N=args.block_n,
            num_warps=4,
        )

    run_kernel()
    torch.cuda.synchronize()

    a_ref = _unpack_fp4(a_packed, logical_k=args.k).float()
    b_ref = _unpack_fp4(b_packed.t().contiguous(), logical_k=args.k).float().t().contiguous()
    ref = (a_ref @ b_ref).reshape(-1)
    diff = c.float() - ref.float()
    result = {
        "schema_version": "lynn-engine-p17-triton-dot-scaled-layout-probe-v1",
        "k": args.k,
        "n": args.n,
        "max_abs": float(diff.abs().max().item()),
        "mean_abs": float(diff.abs().mean().item()),
        "cosine": float(F.cosine_similarity(c.float(), ref.float(), dim=0).item()),
        "out_sample": [float(x) for x in c[: min(8, args.n)].tolist()],
        "ref_sample": [float(x) for x in ref[: min(8, args.n)].tolist()],
    }
    for _ in range(args.warmup):
        run_kernel()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(args.iters):
        run_kernel()
    end.record()
    torch.cuda.synchronize()
    result["timing_ms"] = {
        "dot_scaled_raw_ms": float(start.elapsed_time(end) / args.iters),
        "logical_ops": int(2 * args.k * args.n),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
