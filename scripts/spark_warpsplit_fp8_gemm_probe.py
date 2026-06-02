#!/usr/bin/env python3
"""Spark warp-split-K + small-M FP8 GEMM probe (FlashRT technique on sm_121).

FlashRT's `fp4_w4a4_mma_warpsplit_mrows_sm120` proves two things on sm_120 for
the speculative-decode VERIFY path:
  (1) small-M epilogue: the 16-row MMA tile means M<=16 verify rows cost the
      SAME weight HBM read as M=1 (the verify is "free" over the weight read);
  (2) warp-split-K: splitting the long-K reduction across warps fills the SMs
      that a single-warp GEMV underfills on long-K shapes (mlp_down K=17408).

Spark (sm_121 / GB10) has NO FP4 MMA, but it DOES have the FP8 e4m3 m16n8k32
MMA atom — also a 16-row tile. So both techniques transfer to an FP8 path.
This probe reimplements the structure in Triton (clean-room; FlashRT is Apache
but we reimplement, not copy) and measures, with NO model load:
  * correctness: cos / max-rel vs a bf16-dequant reference (token-exact intent);
  * small-M-free: latency at M=1 vs M=16 on a long-K verify shape;
  * warp-split-K: latency vs SPLIT_K in {1,2,4,8}.

Run on Spark:  python scripts/spark_warpsplit_fp8_gemm_probe.py
"""
from __future__ import annotations

import argparse
import torch

try:
    import triton
    import triton.language as tl
except Exception as e:  # pragma: no cover
    raise SystemExit(f"triton import failed: {e}")


@triton.jit
def _ws_fp8_gemm_kernel(
    A, B, Dscratch, sa, sb,
    M, N, K,
    stride_am, stride_ak, stride_bn, stride_bk, stride_dm, stride_dn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    # D[M,N] = (A[M,K] @ B[N,K]^T) * sa * sb ; A,B fp8_e4m3.  small-M: BLOCK_M=16
    # tile always (the m16n8k32 MMA computes 16 rows regardless of M<=16);
    # warp-split-K: program_id(1) splits the K reduction, partials atomic-add'd.
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    k_per = tl.cdiv(K, SPLIT_K)
    k_start = pid_k * k_per
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(k_start, k_start + k_per, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        a = tl.load(
            A + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak,
            mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0,
        )
        # load B as a [K, N] tile (B is [N, K] row-major -> transpose by indexing)
        b = tl.load(
            B + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn,
            mask=(offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0,
        )
        acc += tl.dot(a, b)  # fp8 x fp8 -> fp32 via m16n8k32 MMA
    acc = acc * sa * sb
    d_ptrs = Dscratch + offs_m[:, None] * stride_dm + offs_n[None, :] * stride_dn
    tl.atomic_add(d_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def ws_fp8_gemm(A_fp8, B_fp8, sa, sb, M, N, K, split_k, block_n=128, block_k=64):
    Dscr = torch.zeros((16, N), dtype=torch.float32, device=A_fp8.device)
    grid = (triton.cdiv(N, block_n), split_k)
    _ws_fp8_gemm_kernel[grid](
        A_fp8, B_fp8, Dscr, float(sa), float(sb), M, N, K,
        A_fp8.stride(0), A_fp8.stride(1), B_fp8.stride(0), B_fp8.stride(1),
        Dscr.stride(0), Dscr.stride(1),
        BLOCK_M=16, BLOCK_N=block_n, BLOCK_K=block_k, SPLIT_K=split_k,
        num_warps=4,
    )
    return Dscr[:M].to(torch.bfloat16)


def _bench(fn, iters=50, warmup=10):
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
    return start.elapsed_time(end) / iters  # ms


def main():
    ap = argparse.ArgumentParser()
    # mlp_down verify shape from the article: K=17408 long-K, N moderate.
    ap.add_argument("--K", type=int, default=17408)
    ap.add_argument("--N", type=int, default=4096)
    args = ap.parse_args()
    dev = "cuda"
    assert torch.cuda.is_available(), "needs Spark GPU"
    torch.manual_seed(0)
    K, N = args.K, args.N
    sa, sb = 0.03, 0.025
    cap = torch.cuda.get_device_capability()
    print(f"device cap = sm_{cap[0]}{cap[1]}  K={K} N={N}")

    # FP8 inputs (M=16 rows; we slice M=1..16). e4m3.
    A16 = (torch.randn(16, K, device=dev) * 0.5).clamp(-6, 6).to(torch.float8_e4m3fn)
    B = (torch.randn(N, K, device=dev) * 0.5).clamp(-6, 6).to(torch.float8_e4m3fn)

    # correctness vs bf16-dequant reference, at M=1 and M=16
    ok = True
    for M in (1, 8, 16):
        out = ws_fp8_gemm(A16[:M].contiguous(), B, sa, sb, M, N, K, split_k=4)
        ref = ((A16[:M].float() * sa) @ (B.float() * sb).t()).to(torch.bfloat16)
        cos = torch.nn.functional.cosine_similarity(
            out.float().flatten(), ref.float().flatten(), dim=0).item()
        denom = ref.float().abs().mean().clamp_min(1e-6)
        maxrel = ((out.float() - ref.float()).abs().max() / denom).item()
        tag = "OK" if cos > 0.999 else "FAIL"
        ok = ok and cos > 0.999
        print(f"  M={M:2d}  cos={cos:.6f}  max_rel={maxrel:.4f}  [{tag}]")

    # small-M-free: M=1 vs M=16 latency (split_k=4)
    print("small-M (latency M=1 vs M=16, split_k=4):")
    t1 = _bench(lambda: ws_fp8_gemm(A16[:1].contiguous(), B, sa, sb, 1, N, K, 4))
    t16 = _bench(lambda: ws_fp8_gemm(A16.contiguous(), B, sa, sb, 16, N, K, 4))
    print(f"  M=1 : {t1*1000:8.1f} us")
    print(f"  M=16: {t16*1000:8.1f} us   (ratio {t16/t1:.3f}x; ~1.0 = verify-free)")

    # THE task#11 money metric: per-position T=1 x16 (our current verify path)
    # vs one small-M tile (the proposed fix). This is the whole point.
    print("verify path: per-position T=1 x16  vs  small-M tile M=16 (split_k=1):")

    def per_position():
        return [ws_fp8_gemm(A16[t:t + 1].contiguous(), B, sa, sb, 1, N, K, 1)
                for t in range(16)]

    tpp = _bench(per_position, iters=20, warmup=5)
    ttile = _bench(lambda: ws_fp8_gemm(A16.contiguous(), B, sa, sb, 16, N, K, 1),
                   iters=20, warmup=5)
    print(f"  per-position x16: {tpp*1000:8.1f} us")
    print(f"  small-M tile    : {ttile*1000:8.1f} us   "
          f"({tpp/ttile:.1f}x faster  <- task #11 verify win)")

    # warp-split-K: latency vs SPLIT_K at M=16, on N=4096 (well-tiled) AND a
    # small-N shape where N-tiling underfills the SMs (FlashRT's regime).
    for Ntest in (N, 512, 256):
        Bt = B[:Ntest].contiguous()
        print(f"warp-split-K (latency vs SPLIT_K, M=16, N={Ntest}):")
        base = None
        for sk in (1, 2, 4, 8):
            t = _bench(lambda sk=sk, Bt=Bt, Ntest=Ntest:
                       ws_fp8_gemm(A16.contiguous(), Bt, sa, sb, 16, Ntest, K, sk))
            if base is None:
                base = t
            print(f"  SPLIT_K={sk}: {t*1000:8.1f} us   ({base/t:.2f}x vs SPLIT_K=1)")

    print("RESULT:", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
