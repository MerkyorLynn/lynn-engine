Implementation complete and byte-compiles cleanly. Here's the concise report.

## Files changed

| File | Change |
|------|--------|
| **`triton_kernels/shared_expert_fused.py`** *(new, 290 lines)* | Two Triton kernels + `shared_expert_decode_fused_triton(...)` wrapper |
| **`engine/moe_packed_nvfp4.py`** | Import; new `_try_fused_shared_expert_output()` helper; wired into the **fixed/production** path (`_moe_forward_decode_packed_nvfp4_fixed_triton`, ~889‑892) and the **general** path (`moe_forward_decode_packed_nvfp4`, ~1181‑1191) |
| **`engine/resident_runner.py`** | New counter; `_prepare_shared_expert_gate_up_fused()` now also runs under the fuse flag and attaches a BF16 `_inter_scratch`; load-time call gated on the flag |

## Design — 2 launches, not 1

The down projection reduces over the *full* intermediate vector, so it needs every `inter[i]` before it can start, and Triton has no cross-program barrier. A true single kernel would have to **recompute** `gate_up` per output block. So I keep the BF16 intermediate store and read it back (no recompute) — the exact trade-off the full-attn fusion made by leaving o_proj a separate GEMM.

1. **`_shared_gate_up_silu_kernel`** — parallel GEMV over the intermediate dim (grid = `ceil(I/BLOCK_I)`): computes `gate_up = h @ gate_up.T` **and** SwiGLU `silu(gate)*up`, stores BF16 `inter[1,I]`.
2. **`_shared_down_gate_add_kernel`** — parallel GEMV over the hidden dim (grid = `ceil(H/BLOCK_D)`): `inter @ down.T`, optional `sigmoid(h @ gate.T)` scale, and residual add into `moe_out` **in place**.

I used the repo's own `tl.sum`-reduction idiom (as in `full_attn_fused.py` / `shared_expert_gate.py`) rather than the codex draft's `tl.dot`, which pads M=1→16 and wastes 15/16 of the MMA lanes. Both kernels parallelize across SMs, so weight memory traffic stays optimal — this is a pure launch-overhead win, not a compute regression.

## Launch reduction

Per MoE layer, the eager BF16 tail is **~8 launches** (gate_up GEMM · silu · mul · down GEMM · gate GEMM · sigmoid · mul · add), or 9 when gate/up aren't pre-fused. The fused path is **exactly 2**.

**~8 → 2 per layer × ~40 MoE layers ≈ 320 → 80 launches/token (~240 fewer/token).** In the campaign's GEMM-centric "160–200" framing, the shared expert collapses to 2 launches/layer.

## Token-exactness

Token-coherent (cos ~ 1), **not** bit-exact. Anchors:
- **BF16 in/out**; fp32 accumulation throughout (matches cuBLAS bf16-input/fp32-accumulate — bf16×bf16 products are exact in fp32, only the reduction *order* differs).
- **Intermediate stored BF16 and read back — no recompute** (the cross-kernel anchor = `(F.silu(gate_s)*up_s)`).
- BF16 rounding at each materialization boundary mirrors the eager dtype staging: `gate_s/up_s` (bf16 chunks) → `F.silu(gate_s)` (bf16) → `inter` (bf16) → down output "shared" (bf16) → `+moe_out` (bf16).
- Scalar gate reduced + `sigmoid` in **fp32**, matching the already-accepted `apply_shared_expert_gate_triton` convention.

**Risks:** (1) Triton vs cuBLAS reduction order — dominant but sub-ULP per term, cos ~ 1; (2) block sizes must be powers of two — enforced by `_check_pow2` in the wrapper; (3) any shape/dtype/device/packed-path mismatch makes `_try_fused_shared_expert_output` return `None` → safe fall-through to the eager path. The packed scalar-bridge (`LYNN_PACKED_SHARED_EXPERT`) path is explicitly left untouched. Validate with fixture comparison vs the eager lines before promotion.

## Env flag

**`LYNN_SHARED_EXPERT_FUSED=1`** — master gate (default `0` → behavior fully unchanged). When set, the resident runner force-attaches `mlp.shared_expert._gate_up_proj.weight` and a `mlp.shared_expert._inter_scratch` (allocation-free / graph-capturable). Tunables: `LYNN_SHARED_EXPERT_FUSED_BLOCK_HIDDEN` (128), `_BLOCK_INTER` (32), `_BLOCK_OUT` (32), `_NUM_WARPS` (4).

Did not run (per instruction). `python -m py_compile` passes on all three files; the new kernel module is import-safe via its `HAS_TRITON` guard (torch isn't installed on this macOS dev box — the engine runs on the DGX Spark/R6000 GPU host).
