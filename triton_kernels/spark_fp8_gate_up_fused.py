"""Spark sm_121 FP8 fused gate/up + SwiGLU Triton kernel.

This is Phase 2 step 2 of the Spark TPS plan
(reference_spark_fp8_w4a8_design_strategy_20260519). The kernel takes
a BF16 activation plus pre-repacked FP8 E4M3 gate + up weights with
per-row scales, runs both projections through the FP8 MMA unit
(GB10 sm_121, 162 TFLOPS peak vs 99 BF16), applies per-row weight
scale + per-token activation scale, fuses SwiGLU (``silu(gate) * up``),
and emits the BF16 intermediate ready for ``down_proj``.

The fused boundary collapses what would be three kernel launches +
two activation-cast launches into one launch, which is the heart of
why naive ``torch._scaled_mm`` per-projection landed at only 14 TPS
in the May-19 PoC.

V0 scope:
  * Decode-time M ∈ {1, small batch} hot path (no autotune yet).
  * Weight layout: row-major [N, K] in FP8 E4M3 (matches existing
    Lynn-native storage shape; reuses ``spark_pack_w4a8_fp8`` output).
  * Per-row weight scale [N] (F32), per-token activation scale
    derived inside the kernel from a precomputed [M] F32 tensor.
  * Caller is responsible for computing the activation scale ahead
    of time (a one-shot reduction, cheap).
  * Output: BF16 [M, intermediate].

V1 scope (current — autotune sweep result applied):
  * Default block config = ``(BLOCK_M=16, BLOCK_K=128, BLOCK_N=32)``,
    the universal winner from the 2160-config Spark sm_121 sweep
    (best/near-best for 60%+ of shapes; see
    ``reports/mtp/QWEN36_FP8_AUTOTUNE_SWEEP_RESULT_20260520.md``).
  * Shape-aware override via ``select_block_config(M, K, N)`` helper
    for high-traffic specialised shapes (M=1 N=6144, M=16 K=6144,
    etc.). Callers can opt in via ``auto_block=True``.
  * N ≤ 256 fast-path advisory: caller should fall back to BF16
    ``torch._scaled_mm`` — FP8 cannot beat BF16 at this output size
    (memory-bandwidth bound, ~0.86× best speedup).

V2 scope (next):
  * Concatenated gate+up weight ([2N, K]) for one B-matrix load.
  * Native-owned intermediate buffer (no Python/Torch round-trip
    before down_proj — see P190 finding #2).
  * Optional col-major weight layout for cuBLASLt parity.
"""
from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:  # pragma: no cover
    triton = None
    tl = None
    HAS_TRITON = False


def _require_triton() -> None:
    if not HAS_TRITON:
        raise RuntimeError("Triton is required for FP8 fused gate/up kernel")


if HAS_TRITON:

    @triton.jit
    def _fp8_gate_up_silu_kernel(
        # Pointers
        x_ptr,                     # [M, K] BF16 activation
        x_scale_ptr,               # [M] F32 per-token activation scale (max_abs / 448)
        w_gate_ptr,                # [N, K] FP8 E4M3 gate weight (row-major)
        w_up_ptr,                  # [N, K] FP8 E4M3 up weight (row-major)
        w_gate_scale_ptr,          # [N] F32 per-row gate weight scale
        w_up_scale_ptr,            # [N] F32 per-row up weight scale
        out_ptr,                   # [M, N] BF16 output = silu(gate*s_g*s_x) * (up*s_u*s_x)
        # Sizes
        M, K, N,
        # Strides
        stride_xm, stride_xk,
        stride_wm, stride_wk,
        stride_om, stride_on,
        # Block sizes
        BLOCK_M: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        mask_m = offs_m < M
        mask_n = offs_n < N

        acc_gate = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        acc_up = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        # K-loop: accumulate FP8 × FP8 → F32 MMA.
        for k_block in range(0, K, BLOCK_K):
            k_offs = k_block + offs_k
            mask_k = k_offs < K

            # Load BF16 activation block [BLOCK_M, BLOCK_K]
            x_block = tl.load(
                x_ptr + offs_m[:, None] * stride_xm + k_offs[None, :] * stride_xk,
                mask=mask_m[:, None] & mask_k[None, :],
                other=0.0,
            )
            # Load per-token activation scale [BLOCK_M]
            x_scale = tl.load(x_scale_ptr + offs_m, mask=mask_m, other=1.0)
            # Quantize activation to FP8: x_fp8 = x_bf16 / x_scale; clipped to FP8 range.
            x_q = x_block.to(tl.float32) / x_scale[:, None]
            x_fp8 = x_q.to(tl.float8e4nv)

            # Load FP8 weight blocks [BLOCK_N, BLOCK_K]
            w_gate_block = tl.load(
                w_gate_ptr + offs_n[:, None] * stride_wm + k_offs[None, :] * stride_wk,
                mask=mask_n[:, None] & mask_k[None, :],
                other=tl.zeros((1,), dtype=tl.float8e4nv),
            )
            w_up_block = tl.load(
                w_up_ptr + offs_n[:, None] * stride_wm + k_offs[None, :] * stride_wk,
                mask=mask_n[:, None] & mask_k[None, :],
                other=tl.zeros((1,), dtype=tl.float8e4nv),
            )

            # FP8 × FP8 MMA. tl.dot expects [M, K] × [K, N], so transpose weight on read.
            acc_gate += tl.dot(x_fp8, w_gate_block.trans(), out_dtype=tl.float32)
            acc_up += tl.dot(x_fp8, w_up_block.trans(), out_dtype=tl.float32)

        # Apply per-row weight scale + per-token activation scale.
        w_gate_scale = tl.load(w_gate_scale_ptr + offs_n, mask=mask_n, other=1.0)
        w_up_scale = tl.load(w_up_scale_ptr + offs_n, mask=mask_n, other=1.0)
        x_scale = tl.load(x_scale_ptr + offs_m, mask=mask_m, other=1.0)

        scale_combined_g = x_scale[:, None] * w_gate_scale[None, :]
        scale_combined_u = x_scale[:, None] * w_up_scale[None, :]
        gate = acc_gate * scale_combined_g
        up = acc_up * scale_combined_u

        # SwiGLU: silu(gate) * up
        inter = (gate * tl.sigmoid(gate)) * up

        # Store BF16
        tl.store(
            out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
            inter.to(tl.bfloat16),
            mask=mask_m[:, None] & mask_n[None, :],
        )


def select_block_config(m: int, k: int, n: int) -> tuple[int, int, int]:
    """Shape-aware best block config from the Spark sm_121 autotune sweep.

    Returns ``(BLOCK_M, BLOCK_K, BLOCK_N)`` selected from the per-shape
    top-1 entries of ``reports/mtp/QWEN36_FP8_AUTOTUNE_SWEEP_RESULT_20260520.md``.

    For shapes not in the override table this falls back to the universal
    winner ``(16, 128, 32)``. Callers should still avoid this kernel when
    ``n <= 256`` (memory-bound; use BF16 ``_scaled_mm`` instead).
    """
    # High-impact specialised overrides for the Lynn 35B-A3B hot shapes.
    if m == 1:
        if k == 2048 and n == 6144:
            return (16, 64, 32)        # 6.00× — best in sweep
        if k == 4096 and n == 2048:
            return (16, 128, 64)       # 5.76×
        if k == 6144 and n == 2048:
            return (16, 128, 32)       # 4.25×
        if k == 6144 and n == 6144:
            return (64, 128, 32)       # 1.79×
    elif 4 <= m <= 8:
        if k == 2048 and n == 6144:
            return (16, 64, 32)        # 5.67× / 5.38×
        if k == 6144 and n == 6144:
            return (64, 128, 32) if m == 4 else (16, 128, 32)
    elif m >= 16:
        if k == 6144 and n == 6144:
            return (64, 128, 32)       # 2.04×
        if k == 2048 and n == 6144:
            return (32, 64, 64)        # 3.65×
    return (16, 128, 32)               # universal winner


def fp8_gate_up_silu_fused(
    x_bf16: torch.Tensor,             # [M, K]
    w_gate_fp8: torch.Tensor,         # [N, K] FP8 E4M3
    w_up_fp8: torch.Tensor,           # [N, K] FP8 E4M3
    w_gate_scale: torch.Tensor,       # [N] F32
    w_up_scale: torch.Tensor,         # [N] F32
    *,
    block_m: int = 16,
    block_k: int = 128,
    block_n: int = 32,
    auto_block: bool = False,
) -> torch.Tensor:
    """Run fused FP8 gate/up + SwiGLU on Spark sm_121.

    Returns BF16 intermediate [M, N] = silu(gate * scales) * (up * scales)
    ready for the down_proj step.

    The default block config ``(16, 128, 32)`` is the universal winner
    from the 2160-config sweep (see ``select_block_config`` for
    shape-aware overrides). Pass ``auto_block=True`` to dispatch on the
    runtime shape.
    """
    _require_triton()
    if not x_bf16.is_cuda:
        raise ValueError("activation must be CUDA tensor")
    if x_bf16.dtype != torch.bfloat16:
        raise ValueError(f"activation must be BF16, got {x_bf16.dtype}")
    if w_gate_fp8.dtype != torch.float8_e4m3fn or w_up_fp8.dtype != torch.float8_e4m3fn:
        raise ValueError("weights must be float8_e4m3fn")
    if w_gate_fp8.shape != w_up_fp8.shape:
        raise ValueError("gate and up weights must have same shape")
    if w_gate_scale.dtype != torch.float32 or w_up_scale.dtype != torch.float32:
        raise ValueError("scales must be float32")

    M, K = x_bf16.shape
    N, K_w = w_gate_fp8.shape
    if K != K_w:
        raise ValueError(f"K mismatch: act K={K}, weight K={K_w}")

    if auto_block:
        block_m, block_k, block_n = select_block_config(M, K, N)

    # Compute per-token activation scale = max_abs / 448 (FP8 E4M3 max).
    x_scale = (x_bf16.abs().amax(dim=-1).clamp_min(1.0e-12) / 448.0).to(torch.float32)

    out = torch.empty((M, N), dtype=torch.bfloat16, device=x_bf16.device)

    grid = (triton.cdiv(M, block_m), triton.cdiv(N, block_n))
    _fp8_gate_up_silu_kernel[grid](
        x_bf16,
        x_scale,
        w_gate_fp8,
        w_up_fp8,
        w_gate_scale,
        w_up_scale,
        out,
        M, K, N,
        x_bf16.stride(0), x_bf16.stride(1),
        w_gate_fp8.stride(0), w_gate_fp8.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=block_m,
        BLOCK_K=block_k,
        BLOCK_N=block_n,
    )
    return out


def fp8_gate_up_silu_reference(
    x_bf16: torch.Tensor,
    w_gate_bf16: torch.Tensor,     # [N, K] BF16 reference weight
    w_up_bf16: torch.Tensor,       # [N, K] BF16 reference weight
) -> torch.Tensor:
    """Reference BF16 path for correctness verification.

    Returns silu(x @ w_gate^T) * (x @ w_up^T) in BF16.
    """
    gate = torch.nn.functional.linear(x_bf16, w_gate_bf16)
    up = torch.nn.functional.linear(x_bf16, w_up_bf16)
    inter = torch.nn.functional.silu(gate) * up
    return inter.to(torch.bfloat16)
