# Native MoE Slot Strict BF16 — Final Report

Date: 2026-05-18
Branch: `claude/native-moe-slot-strict-fused-20260518`

## Verdict: RESEARCH_ARTIFACT

Numerics exact (slot-order), but latency far over target.

## Key Finding

| Comparison | max_abs | Exact? |
|------------|---------|--------|
| Native strict vs Slot PyTorch (F.linear loop) | **0.0** | **YES 18/18** |
| Native strict vs Stored (unique/index_add) | 3.9e-3 | NO |
| Slot PyTorch vs Stored (unique/index_add) | 3.9e-3 | NO |

**The 3.9e-3 drift is NOT a kernel bug.** It's the FP non-associativity floor between
slot-order accumulation (`out += ffn_k * rw_k`) and unique-expert accumulation
(`index_add_` over active experts). Both are mathematically correct; they differ
because floating-point addition is not associative.

## Root Cause of Previous Drift

The previous `native_slot_output_owned_bf16` had `out.add_(down_out, alpha=route_w)`
which applies float32 alpha to a BF16 tensor — different from Python's
`out += ffn * rw[k].to(bf16)` which first truncates route_w to BF16 then
multiplies element-wise. Fixing to `out += down_out * bf16(route_w)` yields
**exact match**.

## Latency

| Kernel | Avg ms | vs Triton |
|--------|--------|-----------|
| Custom scalar (0.052ms) | 0.052 | -12% faster |
| cuBLAS strict (this) | 0.467 | +7.9x slower |
| Triton active baseline | 0.059 | — |

cuBLAS strict is slow because it makes **16 sequential torch::mm calls** (8 slots × 2 GEMMs),
each incurring kernel launch overhead (~25μs). The actual GEMM compute is ~2μs per call
but launch dominates.

## Implication for Production

The fast path (0.052ms) is numerically equivalent to F.linear slot-order **except** for
the BF16 intermediate truncation in Stage 1→2. To achieve both speed AND exactness:

**Option A (recommended)**: Accept slot-order semantics as the new reference. Regenerate
`routed_output` in p135 using slot-order F.linear (not unique/index_add). Then the
fast 0.052ms kernel can be tested against a slot-order ground truth and only the
inter BF16 truncation remains.

**Option B**: Fuse gate_up + down into a single kernel that keeps FP32 intermediate
in shared memory (never BF16 round-trip). This eliminates the last source of drift
vs slot-order PyTorch, but requires significant CUDA engineering.

## Files

- `csrc/lynn_native/moe_slot_strict_bf16.cu` — cuBLAS-matched strict kernel
- `benchmarks/candidates/native_slot_strict_bf16.py` — candidate runner
- This report

## Acceptance vs Requirements

| Requirement | Status |
|-------------|--------|
| py_compile pass | ✅ |
| native-vs-slot PyTorch: 18/18 max_abs=0 | ✅ **EXACT** |
| native-vs-unique/index_add: max_abs<=1e-3 | ❌ 15/18 pass, 3 RED (L32/36/39) |
| native-vs-unique/index_add: cosine>=0.999999 | ❌ (0.99998, FP floor) |
| Sprint max_abs<=4.88e-4 | ❌ (FP non-associativity ceiling) |
| Latency <= 0.059ms | ❌ cuBLAS strict = 0.467ms |
| Not promote to resident | ✅ |
