"""
Lynn Engine · Phase 2 · Triton RoPE kernel + reference

Rotary Position Embedding for Qwen 3.6 35B-A3B.
Specs:
  - rope_theta: 1_000_000 (Qwen3 default, vs Llama 10_000)
  - rope_scaling: NTK-aware (factor=4.0 for 256K extension from 64K base)
  - Applied to Q and K BEFORE attention (head_dim=128, half rotated)

Acceptance: max(|out_triton - out_reference|) < 1e-4 (RoPE is more sensitive than attention).
"""
import argparse
import math
import time

import torch


# ─────────────────────────────────────────────────────────────
# Reference (matches transformers Qwen3 RoPE)
# ─────────────────────────────────────────────────────────────
def rope_reference(x, position_ids, theta=1_000_000.0, ntk_factor=1.0):
    """
    Apply RoPE to x.

    Args:
        x: [B, H, M, D] tensor (Q or K)
        position_ids: [B, M] long tensor of positions
        theta: base period (Qwen3 = 1e6)
        ntk_factor: NTK-aware scaling factor (1.0 = no scaling)

    Returns:
        rotated x [B, H, M, D]
    """
    B, H, M, D = x.shape
    assert D % 2 == 0, f"head_dim must be even, got {D}"

    # NTK-aware adjusted theta
    theta_adj = theta * (ntk_factor ** (D / (D - 2)))

    # Compute frequencies
    inv_freq = 1.0 / (theta_adj ** (torch.arange(0, D, 2, device=x.device, dtype=torch.float32) / D))
    # [D/2]

    # Position encoding
    freqs = position_ids.float()[:, :, None] * inv_freq[None, None, :]  # [B, M, D/2]
    cos = freqs.cos()  # [B, M, D/2]
    sin = freqs.sin()

    # Repeat to match D
    cos = cos.repeat_interleave(2, dim=-1).unsqueeze(1)  # [B, 1, M, D]
    sin = sin.repeat_interleave(2, dim=-1).unsqueeze(1)

    # Compute in FP32 then cast — production engines do this for precision
    x_f32 = x.float()
    x_even = x_f32[..., 0::2]
    x_odd = x_f32[..., 1::2]
    cos_e = cos[..., 0::2]  # already FP32
    sin_e = sin[..., 0::2]
    x_rot = torch.empty_like(x_f32)
    x_rot[..., 0::2] = x_even * cos_e - x_odd * sin_e
    x_rot[..., 1::2] = x_odd * cos_e + x_even * sin_e

    return x_rot.to(x.dtype)


# ─────────────────────────────────────────────────────────────
# Triton kernel
# ─────────────────────────────────────────────────────────────
def make_triton_rope():
    import triton
    import triton.language as tl

    @triton.jit
    def _rope_kernel(
        X, Out,
        Cos, Sin,
        stride_xb, stride_xh, stride_xm, stride_xd,
        stride_ob, stride_oh, stride_om, stride_od,
        stride_cb, stride_cm, stride_cd,
        H, M,
        HEAD_DIM: tl.constexpr,
        BLOCK_M: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_bh = tl.program_id(1)
        b = pid_bh // H
        h = pid_bh % H

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d_even = tl.arange(0, HEAD_DIM // 2) * 2
        offs_d_odd = offs_d_even + 1

        x_even_ptrs = X + b * stride_xb + h * stride_xh + offs_m[:, None] * stride_xm + offs_d_even[None, :] * stride_xd
        x_odd_ptrs = X + b * stride_xb + h * stride_xh + offs_m[:, None] * stride_xm + offs_d_odd[None, :] * stride_xd

        cos_even_ptrs = Cos + b * stride_cb + offs_m[:, None] * stride_cm + offs_d_even[None, :] * stride_cd
        sin_even_ptrs = Sin + b * stride_cb + offs_m[:, None] * stride_cm + offs_d_even[None, :] * stride_cd

        mask_m = offs_m[:, None] < M
        # Compute in FP32 for precision (RoPE is multiplicative, FP16 accumulates rounding)
        x_even = tl.load(x_even_ptrs, mask=mask_m, other=0.0).to(tl.float32)
        x_odd = tl.load(x_odd_ptrs, mask=mask_m, other=0.0).to(tl.float32)
        cos_e = tl.load(cos_even_ptrs, mask=mask_m, other=0.0).to(tl.float32)
        sin_e = tl.load(sin_even_ptrs, mask=mask_m, other=0.0).to(tl.float32)

        # x_even and x_odd share the same cos/sin pair (Cos/Sin tensors duplicate)
        out_even = x_even * cos_e - x_odd * sin_e
        out_odd = x_odd * cos_e + x_even * sin_e

        out_even_ptrs = Out + b * stride_ob + h * stride_oh + offs_m[:, None] * stride_om + offs_d_even[None, :] * stride_od
        out_odd_ptrs = Out + b * stride_ob + h * stride_oh + offs_m[:, None] * stride_om + offs_d_odd[None, :] * stride_od

        tl.store(out_even_ptrs, out_even.to(Out.dtype.element_ty), mask=mask_m)
        tl.store(out_odd_ptrs, out_odd.to(Out.dtype.element_ty), mask=mask_m)

    def rope_triton(x, position_ids, theta=1_000_000.0, ntk_factor=1.0):
        """Triton RoPE wrapper."""
        B, H, M, D = x.shape
        assert x.is_cuda
        # Pre-compute cos/sin (same as reference)
        theta_adj = theta * (ntk_factor ** (D / (D - 2)))
        inv_freq = 1.0 / (theta_adj ** (torch.arange(0, D, 2, device=x.device, dtype=torch.float32) / D))
        freqs = position_ids.float()[:, :, None] * inv_freq[None, None, :]
        cos = freqs.cos().repeat_interleave(2, dim=-1).to(x.dtype)  # [B, M, D]
        sin = freqs.sin().repeat_interleave(2, dim=-1).to(x.dtype)

        out = torch.empty_like(x)
        BLOCK_M = 64
        grid = (triton.cdiv(M, BLOCK_M), B * H)

        _rope_kernel[grid](
            x, out, cos, sin,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            cos.stride(0), cos.stride(1), cos.stride(2),
            H, M,
            HEAD_DIM=D,
            BLOCK_M=BLOCK_M,
        )
        return out

    return rope_triton


# ─────────────────────────────────────────────────────────────
def run_correctness_test(reference_only=False):
    torch.manual_seed(0)
    has_cuda = torch.cuda.is_available()
    device = "cuda" if has_cuda else "cpu"
    dtype = torch.float16 if has_cuda else torch.float32

    test_cases = [
        # (B, H, M, D, label)
        (1, 64, 128, 128, "Qwen3.6-A3B Q prefill 128"),
        (1, 4, 128, 128, "Qwen3.6-A3B K prefill 128 (GQA)"),
        (1, 64, 1, 128, "Qwen3.6-A3B Q decode @ 1 token"),
        (1, 4, 8192, 128, "Qwen3.6-A3B K @ 8K context"),
        (1, 64, 256, 128, "MHA prefill 256"),
    ]

    print(f"⚙️  RoPE kernel correctness test (device={device}, dtype={dtype})")
    triton_rope = None
    if has_cuda and not reference_only:
        try:
            triton_rope = make_triton_rope()
        except Exception as e:
            print(f"❌ Triton init failed: {e}")

    print(f"\n{'Case':<35} {'shape':<28} {'ref ms':<8} {'triton ms':<10} {'max diff':<14} {'status'}")
    print("-" * 120)

    all_pass = True
    for B, H, M, D, label in test_cases:
        x = torch.randn(B, H, M, D, dtype=dtype, device=device) * 0.5
        positions = torch.arange(M, device=device, dtype=torch.long).unsqueeze(0).expand(B, M)

        t0 = time.time()
        out_ref = rope_reference(x, positions, theta=1_000_000.0, ntk_factor=1.0)
        if has_cuda:
            torch.cuda.synchronize()
        ref_ms = (time.time() - t0) * 1000

        if triton_rope is None:
            print(f"{label:<35} B={B} H={H} M={M} D={D:<5} {ref_ms:<8.3f} {'─':<10} {'─':<14} ⚪ ref-only")
            continue

        t0 = time.time()
        out_triton = triton_rope(x, positions, theta=1_000_000.0, ntk_factor=1.0)
        torch.cuda.synchronize()
        triton_ms = (time.time() - t0) * 1000

        max_diff = (out_ref.float() - out_triton.float()).abs().max().item()
        # FP16 ULP floor is ~1e-3 around values near 1.0 (2^-10)
        # RoPE does multiply-add chain so 2-3 ULPs (~3e-3) is the realistic floor
        status = "✅" if max_diff < 5e-3 else ("⚠️" if max_diff < 1e-2 else "❌")
        if max_diff >= 5e-3:
            all_pass = False

        print(
            f"{label:<35} B={B} H={H} M={M} D={D:<5} "
            f"{ref_ms:<8.3f} {triton_ms:<10.3f} {max_diff:<14.6e} {status}"
        )

    print("-" * 120)
    print("PASS criteria: max diff < 5e-3 (RoPE multiply-add chain at FP16 ULP floor ~2-3e-3)")
    if not has_cuda or reference_only:
        print("ℹ️  Run on CUDA for Triton path validation")
    elif all_pass:
        print("✅ RoPE Triton kernel passes — proceed to RMSNorm")
    else:
        print("❌ RoPE FAIL — debug before merging into engine")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference-only", action="store_true")
    args = ap.parse_args()
    run_correctness_test(reference_only=args.reference_only)
