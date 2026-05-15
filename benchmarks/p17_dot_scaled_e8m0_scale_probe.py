#!/usr/bin/env python3
"""P17: infer Triton `tl.dot_scaled` e8m0 scale behavior.

`tl.dot_scaled` documents e8m0 uint8 scales. Lynn's current NVFP4 artifact uses
per-16 e4m3-ish scales, so before designing a conversion path we need an
empirical handle on e8m0 byte encoding and rhs_scale layout.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

import triton
import triton.language as tl


@triton.jit
def _scaled_kernel(
    a_ptr,
    a_scale_ptr,
    b_ptr,
    b_scale_ptr,
    c_ptr,
    K_PACKED: tl.constexpr,
    N: tl.constexpr,
    GROUPS: tl.constexpr,
    BLOCK_K_PACKED: tl.constexpr,
    BLOCK_N: tl.constexpr,
    RHS_LAYOUT_NG: tl.constexpr,
):
    offs_kp = tl.arange(0, BLOCK_K_PACKED)
    offs_n = tl.arange(0, BLOCK_N)
    offs_g = tl.arange(0, GROUPS)
    a = tl.load(a_ptr + offs_kp, mask=offs_kp < K_PACKED, other=0)
    b = tl.load(b_ptr + offs_kp[:, None] * N + offs_n[None, :], mask=(offs_kp[:, None] < K_PACKED) & (offs_n[None, :] < N), other=0)
    a_s = tl.load(a_scale_ptr + offs_g)
    if RHS_LAYOUT_NG:
        b_s = tl.load(b_scale_ptr + offs_n[:, None] * GROUPS + offs_g[None, :], mask=offs_n[:, None] < N, other=0)
    else:
        b_s = tl.load(b_scale_ptr + offs_g[:, None] * N + offs_n[None, :], mask=offs_n[None, :] < N, other=0)
    out = tl.dot_scaled(
        a[None, :],
        a_s[None, :],
        "e2m1",
        b,
        b_s,
        "e2m1",
        lhs_k_pack=True,
        rhs_k_pack=True,
    )
    tl.store(c_ptr + offs_n, tl.reshape(out, (BLOCK_N,)), mask=offs_n < N)


def _pack_ones(shape: tuple[int, ...]) -> torch.Tensor:
    # E2M1 code 2 is +1.0. Pack pairs into low/high nibbles.
    codes = torch.full(shape, 2, device="cuda", dtype=torch.uint8)
    return (codes[..., 0::2] | (codes[..., 1::2] << 4)).contiguous()


def _run(byte: int, *, rhs_layout_ng: bool, k: int, n: int) -> torch.Tensor:
    groups = k // 32
    a = _pack_ones((1, k))
    b_codes = torch.full((k, n), 2, device="cuda", dtype=torch.uint8)
    b = (b_codes[0::2, :] | (b_codes[1::2, :] << 4)).contiguous()
    a_scale = torch.full((groups,), byte, device="cuda", dtype=torch.uint8)
    if rhs_layout_ng:
        b_scale = torch.full((n, groups), byte, device="cuda", dtype=torch.uint8)
    else:
        b_scale = torch.full((groups, n), byte, device="cuda", dtype=torch.uint8)
    c = torch.empty((n,), device="cuda", dtype=torch.float32)
    _scaled_kernel[(1,)](
        a,
        a_scale,
        b,
        b_scale,
        c,
        K_PACKED=k // 2,
        N=n,
        GROUPS=groups,
        BLOCK_K_PACKED=k // 2,
        BLOCK_N=triton.next_power_of_2(n),
        RHS_LAYOUT_NG=rhs_layout_ng,
        num_warps=4,
    )
    torch.cuda.synchronize()
    return c


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=64)
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--bytes", default="120,124,126,127,128,129,130,132")
    args = ap.parse_args()

    rows = []
    for layout in (True, False):
        for byte in [int(x) for x in args.bytes.split(",") if x.strip()]:
            try:
                out = _run(byte, rhs_layout_ng=layout, k=args.k, n=args.n)
                rows.append(
                    {
                        "rhs_layout": "N,G" if layout else "G,N",
                        "scale_byte": byte,
                        "sample0": float(out[0].item()),
                        "mean": float(out.float().mean().item()),
                    }
                )
            except Exception as exc:
                rows.append({"rhs_layout": "N,G" if layout else "G,N", "scale_byte": byte, "error": repr(exc)})

    result = {
        "schema_version": "lynn-engine-p17-dot-scaled-e8m0-scale-probe-v1",
        "k": args.k,
        "n": args.n,
        "raw_unscaled_expected": args.k,
        "rows": rows,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
