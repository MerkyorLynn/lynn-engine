"""
Triton bf16 GEMV kernel: y = A @ x
  A : [N, K] bf16
  x : [K]    bf16
  y : [N]    float32
"""

import triton
import triton.language as tl
import torch


# ---------------------------------------------------------------------------
# Triton kernel
# ---------------------------------------------------------------------------
@triton.jit
def gemv_kernel(
    A_ptr,          # *const bf16  (N, K)  row-major
    x_ptr,          # *const bf16  (K,)
    y_ptr,          # *fp32       (N,)
    N,              # int32  — number of rows
    K,              # int32  — inner dimension
    stride_am,      # int32  — stride between rows of A  (in elements)
    stride_ak,      # int32  — stride between cols of A  (in elements)
    stride_x,       # int32  — stride of x              (in elements)
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Each program (block) processes BLOCK_N consecutive rows of A.
    K is tiled in chunks of BLOCK_K; the fp32 accumulator is updated
    per tile and written back at the end.
    """
    pid_n = tl.program_id(axis=0)                     # which row-block

    # --- row offsets for this block (mask N tail) --------------------------
    row_start = pid_n * BLOCK_N
    row_end   = tl.minimum(row_start + BLOCK_N, N)   # exclusive
    rows      = row_end - row_start                   # may be < BLOCK_N

    # --- fp32 accumulator: one entry per row in this block -----------------
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    # --- tile over K -------------------------------------------------------
    for k0 in range(0, K, BLOCK_K):
        k_end = tl.minimum(k0 + BLOCK_K, K)
        k_len = k_end - k0

        # Load A tile [rows, k_len]  — cast bf16→fp32 on load
        offs_a = row_start + tl.arange(0, BLOCK_N)[:, None]   # [BLOCK_N,1]
        offs_k = k0        + tl.arange(0, BLOCK_K)[None, :]   # [1,BLOCK_K]
        a_mask = (offs_a < N)[:, None] & (offs_k < K)          # [BLOCK_N,BLOCK_K]
        a_ptrs = A_ptr + offs_a * stride_am + offs_k * stride_ak
        a_tile = tl.load(a_ptrs, mask=a_mask, other=0.0).to(tl.float32)

        # Load x tile [k_len]  — cast bf16→fp32 on load
        x_offs = k0 + tl.arange(0, BLOCK_K)
        x_mask = x_offs < K
        x_ptrs = x_ptr + x_offs * stride_x
        x_tile = tl.load(x_ptrs, mask=x_mask, other=0.0).to(tl.float32)

        # Accumulate: a_tile [BLOCK_N, k_len] * x_tile [k_len]
        # → sum over K axis (axis=1) → [BLOCK_N]
        acc += tl.sum(a_tile * x_tile[:, None].T, axis=1)

    # --- write results (mask rows that exceed N) ----------------------------
    row_offs = row_start + tl.arange(0, BLOCK_N)
    row_mask = row_offs < N
    tl.store(y_ptr + row_offs, acc, mask=row_mask)


# ---------------------------------------------------------------------------
# Reference / test harness
# ---------------------------------------------------------------------------
def main():
    if not torch.cuda.is_available():
        print("NO_CUDA")
        return

    # --- problem sizes ------------------------------------------------------
    N, K = 512, 2048
    BLOCK_N, BLOCK_K = 64, 128

    # --- random bf16 inputs on GPU ------------------------------------------
    A = torch.randn(N, K, dtype=torch.bfloat16, device="cuda")
    x = torch.randn(K,    dtype=torch.bfloat16, device="cuda")

    # --- launch kernel ------------------------------------------------------
    y_triton = torch.empty(N, dtype=torch.float32, device="cuda")
    grid = (triton.cdiv(N, BLOCK_N),)
    gemv_kernel[grid](
        A, x, y_triton,
        N, K,
        A.stride(0), A.stride(1),
        x.stride(0),
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )

    # --- torch reference ----------------------------------------------------
    y_ref = A.float() @ x.float()

    # --- compare ------------------------------------------------------------
    ok = torch.allclose(y_triton, y_ref, atol=1e-1, rtol=1e-2)
    if ok:
        max_err = (y_triton - y_ref).abs().max().item()
        print(f"PASS  max_abs_err={max_err:.6f}")
    else:
        max_err = (y_triton - y_ref).abs().max().item()
        print(f"FAIL  max_abs_err={max_err:.6f}")


if __name__ == "__main__":
    main()
