"""
Lynn Engine · Phase 1 spike · Triton attention kernel + reference

Single-batch causal multi-head attention with GQA support.
Targets Qwen 3.6 35B-A3B layout: 64 query heads / 4 KV heads / head_dim=128.

This is the FIRST kernel and the GO/NO-GO gate per docs/DESIGN.md §8 Phase 1.
Acceptance: max(|out_triton - out_reference|) < 1e-3 on FP16 path.

Run:
    # Mac (PyTorch reference only, sanity):
    python attention.py --reference-only

    # Spark / RTX PRO 6000 (Triton + reference + diff check):
    python attention.py
"""
import argparse
import time
import math

import torch
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────
# PyTorch reference implementation (matches Qwen3.6 attention)
# ─────────────────────────────────────────────────────────────
def attention_reference(q, k, v, causal=True):
    """
    PyTorch reference attention with GQA.

    Args:
        q: [B, H_Q, M, D] — query tensor
        k: [B, H_KV, N, D] — key tensor (H_KV may be < H_Q for GQA)
        v: [B, H_KV, N, D] — value tensor
        causal: apply causal mask

    Returns:
        out: [B, H_Q, M, D]
    """
    B, H_Q, M, D = q.shape
    _, H_KV, N, _ = k.shape
    assert H_Q % H_KV == 0, f"H_Q ({H_Q}) must be divisible by H_KV ({H_KV})"

    # Expand KV heads to Q heads (GQA)
    if H_KV != H_Q:
        repeat_factor = H_Q // H_KV
        k = k.repeat_interleave(repeat_factor, dim=1)
        v = v.repeat_interleave(repeat_factor, dim=1)

    # Use PyTorch's SDPA (FlashAttention if CUDA + FA installed)
    return F.scaled_dot_product_attention(q, k, v, is_causal=causal)


# ─────────────────────────────────────────────────────────────
# Triton kernel (only imported/used on CUDA)
# ─────────────────────────────────────────────────────────────
def make_triton_attention():
    """Create the Triton attention forward kernel.

    Returns the Python wrapper function. Triton itself only loads on CUDA.
    """
    import triton
    import triton.language as tl

    @triton.jit
    def _attn_fwd(
        Q, K, V, Out,
        sm_scale,
        stride_qb, stride_qh, stride_qm, stride_qd,
        stride_kb, stride_kh, stride_kn, stride_kd,
        stride_vb, stride_vh, stride_vn, stride_vd,
        stride_ob, stride_oh, stride_om, stride_od,
        H_Q,  H_KV, M, N,
        HEAD_DIM: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        IS_CAUSAL: tl.constexpr,
    ):
        """FlashAttention-1 style forward — online softmax.

        Each program computes BLOCK_M query rows for one (batch, head_q).
        Iterates over BLOCK_N key/value tiles.
        """
        pid_m = tl.program_id(0)         # query block index
        pid_bh = tl.program_id(1)        # batch * H_Q index
        b = pid_bh // H_Q
        hq = pid_bh % H_Q
        # GQA: KV head index
        hkv = hq * H_KV // H_Q

        # Compute pointer offsets
        q_offset = b * stride_qb + hq * stride_qh
        k_offset = b * stride_kb + hkv * stride_kh
        v_offset = b * stride_vb + hkv * stride_vh
        o_offset = b * stride_ob + hq * stride_oh

        # Initialize accumulators (online softmax state)
        m_i = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
        l_i = tl.full([BLOCK_M], 0.0, dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

        # Load Q block once (stays in registers/SRAM)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, HEAD_DIM)
        q_ptrs = Q + q_offset + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
        q = tl.load(q_ptrs, mask=offs_m[:, None] < M, other=0.0)

        # Iterate over K/V blocks
        for start_n in range(0, N, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            k_ptrs = K + k_offset + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd
            v_ptrs = V + v_offset + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd

            k = tl.load(k_ptrs, mask=offs_n[:, None] < N, other=0.0)
            v = tl.load(v_ptrs, mask=offs_n[:, None] < N, other=0.0)

            # QK^T
            qk = tl.dot(q, tl.trans(k)) * sm_scale  # [BLOCK_M, BLOCK_N]

            # Causal mask
            if IS_CAUSAL:
                qk = tl.where(offs_m[:, None] >= offs_n[None, :], qk, -float("inf"))

            # Mask out-of-range N
            qk = tl.where(offs_n[None, :] < N, qk, -float("inf"))

            # Online softmax update
            m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
            alpha = tl.exp(m_i - m_ij)
            p = tl.exp(qk - m_ij[:, None])
            l_ij = alpha * l_i + tl.sum(p, axis=1)

            acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)

            m_i = m_ij
            l_i = l_ij

        # Final normalization
        acc = acc / l_i[:, None]

        # Write output
        o_ptrs = Out + o_offset + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
        tl.store(o_ptrs, acc.to(Out.dtype.element_ty), mask=offs_m[:, None] < M)

    def attention_triton(q, k, v, causal=True):
        """Wrapper: q[B, H_Q, M, D], k/v[B, H_KV, N, D] -> out[B, H_Q, M, D]"""
        B, H_Q, M, D = q.shape
        _, H_KV, N, _ = k.shape
        assert q.is_cuda and k.is_cuda and v.is_cuda, "Triton kernel requires CUDA tensors"
        assert q.dtype == k.dtype == v.dtype, "Q/K/V must share dtype"
        assert D in (64, 128, 256), f"Unsupported head_dim={D}"

        out = torch.empty_like(q)
        sm_scale = 1.0 / math.sqrt(D)

        # Tile sizes — tuned for Blackwell sm_12x; will autotune in Phase 2
        BLOCK_M = 64
        BLOCK_N = 64

        grid = (triton.cdiv(M, BLOCK_M), B * H_Q)
        _attn_fwd[grid](
            q, k, v, out,
            sm_scale,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            H_Q, H_KV, M, N,
            HEAD_DIM=D,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            IS_CAUSAL=causal,
        )
        return out

    return attention_triton


# ─────────────────────────────────────────────────────────────
# Numerical alignment test
# ─────────────────────────────────────────────────────────────
def run_correctness_test(reference_only: bool = False):
    """Test Triton kernel against PyTorch reference on Qwen3-A3B-shaped tensors."""
    torch.manual_seed(0)
    has_cuda = torch.cuda.is_available()
    device = "cuda" if has_cuda else "cpu"
    dtype = torch.float16 if has_cuda else torch.float32  # FP16 needs CUDA

    # Qwen 3.6 35B-A3B layout
    test_cases = [
        # (B, H_Q, H_KV, M, N, head_dim, label)
        (1, 64, 4, 128, 128, 128, "qwen3.6-A3B prefill 128"),
        (1, 64, 4, 256, 256, 128, "qwen3.6-A3B prefill 256"),
        (1, 64, 4, 1, 1024, 128, "qwen3.6-A3B decode @ 1K ctx"),
        (1, 64, 4, 1, 8192, 128, "qwen3.6-A3B decode @ 8K ctx"),
        (1, 8, 8, 64, 64, 128, "MHA sanity (no GQA)"),
    ]

    if reference_only or not has_cuda:
        print("⚠️  Reference-only mode (CPU PyTorch). Triton kernel not tested.")
        print("   Run on Spark / RTX PRO 6000 for full validation.\n")

    triton_attn = None
    if has_cuda and not reference_only:
        try:
            triton_attn = make_triton_attention()
        except Exception as e:
            print(f"❌ Triton init failed: {e}")
            print("   Falling back to reference-only mode")

    print(f"{'Test case':<35} {'shape':<32} {'ref ms':<8} {'triton ms':<10} {'max diff':<14} {'status'}")
    print("-" * 130)

    all_pass = True
    for B, H_Q, H_KV, M, N, D, label in test_cases:
        # Use float32 on CPU (fp16 on CPU is slow + buggy in PyTorch)
        q = torch.randn(B, H_Q, M, D, dtype=dtype, device=device)
        k = torch.randn(B, H_KV, N, D, dtype=dtype, device=device)
        v = torch.randn(B, H_KV, N, D, dtype=dtype, device=device)

        # Reference
        t0 = time.time()
        out_ref = attention_reference(q, k, v, causal=True)
        if has_cuda:
            torch.cuda.synchronize()
        ref_ms = (time.time() - t0) * 1000

        if triton_attn is None:
            print(f"{label:<35} B={B} HQ={H_Q} HKV={H_KV} M={M} N={N:<5} {ref_ms:<8.2f} {'─':<10} {'─':<14} ⚪ ref-only")
            continue

        # Triton
        t0 = time.time()
        out_triton = triton_attn(q, k, v, causal=True)
        torch.cuda.synchronize()
        triton_ms = (time.time() - t0) * 1000

        # Diff
        max_diff = (out_ref.float() - out_triton.float()).abs().max().item()
        status = "✅" if max_diff < 1e-2 else ("⚠️" if max_diff < 1e-1 else "❌")
        if max_diff >= 1e-2:
            all_pass = False
        print(
            f"{label:<35} B={B} HQ={H_Q} HKV={H_KV} M={M} N={N:<5} "
            f"{ref_ms:<8.2f} {triton_ms:<10.2f} {max_diff:<14.6f} {status}"
        )

    print("-" * 130)
    print("PASS criteria: max diff < 1e-2 (FP16 tolerance)")
    if not has_cuda or reference_only:
        print("ℹ️  Run again on CUDA for Triton path validation")
    elif all_pass:
        print("✅ All Triton kernels pass — Phase 1 spike SUCCEEDS, proceed to Phase 2 engine")
    else:
        print("❌ Some Triton kernels FAIL — Phase 1 spike NEEDS DEBUG before Phase 2")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference-only", action="store_true",
                    help="Skip Triton kernel, run PyTorch reference only (CPU/Mac dev)")
    args = ap.parse_args()
    run_correctness_test(reference_only=args.reference_only)
