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

V2.1 (this commit — FP8 native intermediate buffer):
  * ``fp8_gate_up_silu_fused_fp8out(...)`` returns
    ``(inter_bf16, inter_fp8, inter_scale)`` so callers feed
    ``torch._scaled_mm`` directly without a Python-side rescale.
  * Matmul kernel writes BF16 + ``atomic_max`` per-row absolute max
    into a F32 buffer in one launch; a fused cast kernel then divides
    by ``max/448`` and casts to FP8 E4M3. The four-op Python rescale
    block (``abs().amax().clamp_min(1e-12)/448 + divide + cast``) goes
    away (P190 finding #2 — ~10% FFN time on Spark sm_121).
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

    @triton.jit
    def _fp8_gate_up_silu_fp8out_kernel(
        # Pointers
        x_ptr,                     # [M, K] BF16 activation
        x_scale_ptr,               # [M] F32 per-token activation scale (max_abs / 448)
        w_gate_ptr,                # [N, K] FP8 E4M3 gate weight (row-major)
        w_up_ptr,                  # [N, K] FP8 E4M3 up weight (row-major)
        w_gate_scale_ptr,          # [N] F32 per-row gate weight scale
        w_up_scale_ptr,            # [N] F32 per-row up weight scale
        out_bf16_ptr,              # [M, N] BF16 intermediate (also used as FP8 cast source)
        inter_rowmax_ptr,          # [M] F32 per-row abs-max of intermediate (atomic_max target)
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
        """Matmul-plus-atomic-max variant of ``_fp8_gate_up_silu_kernel``.

        Identical to the BF16-out kernel except the epilogue also atomically
        updates a per-row max-abs buffer so the FP8 cast kernel that runs
        next can build ``inter_scale = max / 448`` without a Python-side
        reduction. The BF16 intermediate is still written so callers that
        need it (e.g. residual add, debugging) can reuse it.
        """
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        mask_m = offs_m < M
        mask_n = offs_n < N

        acc_gate = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        acc_up = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k_block in range(0, K, BLOCK_K):
            k_offs = k_block + offs_k
            mask_k = k_offs < K

            x_block = tl.load(
                x_ptr + offs_m[:, None] * stride_xm + k_offs[None, :] * stride_xk,
                mask=mask_m[:, None] & mask_k[None, :],
                other=0.0,
            )
            x_scale = tl.load(x_scale_ptr + offs_m, mask=mask_m, other=1.0)
            x_q = x_block.to(tl.float32) / x_scale[:, None]
            x_fp8 = x_q.to(tl.float8e4nv)

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

            acc_gate += tl.dot(x_fp8, w_gate_block.trans(), out_dtype=tl.float32)
            acc_up += tl.dot(x_fp8, w_up_block.trans(), out_dtype=tl.float32)

        w_gate_scale = tl.load(w_gate_scale_ptr + offs_n, mask=mask_n, other=1.0)
        w_up_scale = tl.load(w_up_scale_ptr + offs_n, mask=mask_n, other=1.0)
        x_scale = tl.load(x_scale_ptr + offs_m, mask=mask_m, other=1.0)

        gate = acc_gate * (x_scale[:, None] * w_gate_scale[None, :])
        up = acc_up * (x_scale[:, None] * w_up_scale[None, :])
        inter = (gate * tl.sigmoid(gate)) * up

        # Store BF16 intermediate (still needed as the FP8 cast source).
        tl.store(
            out_bf16_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
            inter.to(tl.bfloat16),
            mask=mask_m[:, None] & mask_n[None, :],
        )

        # Per-row abs-max reduction within this block, then atomic_max into the
        # row-shared buffer. This replaces the four-op Python rescale
        # (``abs().amax().clamp_min(1e-12)/448 + divide + cast``).
        block_abs = tl.abs(inter)
        block_abs = tl.where(mask_n[None, :], block_abs, 0.0)
        block_rowmax = tl.max(block_abs, axis=1)  # [BLOCK_M]
        tl.atomic_max(inter_rowmax_ptr + offs_m, block_rowmax, mask=mask_m)

    @triton.jit
    def _fp8_cast_per_row_kernel(
        inter_bf16_ptr,          # [M, N] BF16 input
        inter_rowmax_ptr,        # [M] F32 per-row abs-max (from atomic_max stage)
        inter_fp8_ptr,           # [M, N] FP8 E4M3 output
        inter_scale_ptr,         # [M, 1] F32 per-row scale = max / 448, clamp_min 1e-12
        M, N,
        stride_im, stride_in,
        stride_fm, stride_fn,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        """Cast BF16 intermediate → FP8 E4M3 using a precomputed per-row max.

        Runs after ``_fp8_gate_up_silu_fp8out_kernel`` has populated
        ``inter_rowmax_ptr`` via ``atomic_max``. Writes both the FP8 buffer
        and the F32 ``inter_scale`` tensor so callers can pass it straight
        to ``torch._scaled_mm`` as ``scale_a``.
        """
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

        mask_m = offs_m < M
        mask_n = offs_n < N

        rowmax = tl.load(inter_rowmax_ptr + offs_m, mask=mask_m, other=1.0)
        # scale = max / 448, clamp_min 1e-12 (matches the Python rescale block).
        scale = tl.maximum(rowmax / 448.0, 1.0e-12)

        # Only program (pid_n=0) writes the per-row scale to avoid redundant stores.
        if pid_n == 0:
            tl.store(inter_scale_ptr + offs_m, scale, mask=mask_m)

        bf16 = tl.load(
            inter_bf16_ptr + offs_m[:, None] * stride_im + offs_n[None, :] * stride_in,
            mask=mask_m[:, None] & mask_n[None, :],
            other=0.0,
        )
        q = bf16.to(tl.float32) / scale[:, None]
        fp8 = q.to(tl.float8e4nv)
        tl.store(
            inter_fp8_ptr + offs_m[:, None] * stride_fm + offs_n[None, :] * stride_fn,
            fp8,
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


def fp8_gate_up_silu_fused_fp8out(
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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused FP8 gate/up + SwiGLU with native FP8 intermediate output.

    Returns ``(inter_bf16, inter_fp8, inter_scale)``:

    * ``inter_bf16`` — [M, N] BF16 intermediate (same as the BF16-out
      kernel above; kept around so debug / residual paths still work).
    * ``inter_fp8`` — [M, N] FP8 E4M3 intermediate ready as ``A`` for
      ``torch._scaled_mm``.
    * ``inter_scale`` — [M, 1] F32 per-row scale = ``max_abs / 448``
      clamped to 1e-12; pass straight as ``scale_a`` to ``_scaled_mm``.

    The four-line Python rescale block in
    ``engine/full_forward.py::_moe_forward``,
    ``engine/full_forward.py::_dense_ffn_forward`` and
    ``engine/moe_optimized.py::moe_forward_decode_fp8`` is replaced by a
    single call here — kernel #1 does matmul + SwiGLU + ``atomic_max``
    per row in one launch; kernel #2 does the FP8 cast using the
    finalised per-row max. Net: 4 PyTorch op launches (``abs``,
    ``amax``, ``divide``, ``to(fp8)``) collapse to 1 fused cast kernel.
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

    x_scale = (x_bf16.abs().amax(dim=-1).clamp_min(1.0e-12) / 448.0).to(torch.float32)

    inter_bf16 = torch.empty((M, N), dtype=torch.bfloat16, device=x_bf16.device)
    inter_fp8 = torch.empty((M, N), dtype=torch.float8_e4m3fn, device=x_bf16.device)
    # Per-row abs-max buffer; atomic_max needs an initialised value (0 — any
    # real positive abs value is >= 0 so the first atomic_max wins).
    inter_rowmax = torch.zeros((M,), dtype=torch.float32, device=x_bf16.device)
    # _scaled_mm wants scale_a shape [M, 1] for row-wise scaling, so allocate
    # in that shape directly (the cast kernel writes a flat [M] view).
    inter_scale = torch.empty((M, 1), dtype=torch.float32, device=x_bf16.device)

    grid_matmul = (triton.cdiv(M, block_m), triton.cdiv(N, block_n))
    _fp8_gate_up_silu_fp8out_kernel[grid_matmul](
        x_bf16,
        x_scale,
        w_gate_fp8,
        w_up_fp8,
        w_gate_scale,
        w_up_scale,
        inter_bf16,
        inter_rowmax,
        M, K, N,
        x_bf16.stride(0), x_bf16.stride(1),
        w_gate_fp8.stride(0), w_gate_fp8.stride(1),
        inter_bf16.stride(0), inter_bf16.stride(1),
        BLOCK_M=block_m,
        BLOCK_K=block_k,
        BLOCK_N=block_n,
    )

    # Cast kernel reuses the matmul block sizes; BLOCK_N tuned a bit larger
    # for the bandwidth-bound cast pass (no MMA pressure).
    cast_block_m = block_m
    cast_block_n = max(block_n, 128)
    grid_cast = (triton.cdiv(M, cast_block_m), triton.cdiv(N, cast_block_n))
    _fp8_cast_per_row_kernel[grid_cast](
        inter_bf16,
        inter_rowmax,
        inter_fp8,
        inter_scale,
        M, N,
        inter_bf16.stride(0), inter_bf16.stride(1),
        inter_fp8.stride(0), inter_fp8.stride(1),
        BLOCK_M=cast_block_m,
        BLOCK_N=cast_block_n,
    )

    return inter_bf16, inter_fp8, inter_scale


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
