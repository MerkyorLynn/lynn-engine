# Lynn Engine P46 · Fused Atomic Active-MoE Probe

Date: 2026-05-16

## Summary

P46 tests the first single-kernel active-MoE CUDA experiment behind the P45 ABI.
The kernel computes each `(slot, intermediate)` gate/up scalar, then immediately
contributes to all down-projection output rows with fp32 atomics.

This avoids the materialized `[top_k, 512]` intermediate tensor, but the atomics
dominate.  The path is a negative result.

## Evidence

Report:

```text
reports/p16_155/p46_fused_atomic_active_moe_probe.json
```

Mean across layers 2, 8, 14, 20, 28, and 36:

| Path | Mean latency |
|---|---:|
| Current Triton active MoE | 0.059206 ms |
| Fused atomic scalar kernel | 0.176838 ms |

```text
fused_atomic_vs_triton_speedup = 0.3348x
min cosine vs Triton           = 0.9999973
max rel_l2                     = 0.00269
```

## Decision

Do not promote the fused-atomic kernel.  It proves the one-kernel shape is
buildable, but atomics are too expensive and introduce small accumulation drift.

## What We Learned

P43-P46 now rule out the easy routes:

- shared BF16 micro-optimization is too small;
- merged-top-k Triton scheduling is slower;
- cross-expert `_scaled_mm` composition is slower and drifts;
- fused scalar atomics are slower and drift.

The next implementation cannot be "one more wrapper".  It needs a true
non-atomic grouped/block-diagonal active expert kernel.

## P47 Direction

The next useful kernel shape should avoid atomics by owning one output tile per
program and reducing over `top_k * intermediate` locally, or by using a CuTe /
CUTLASS grouped GEMM form that expresses the active experts as block-diagonal
matrices.

Minimal P47 goals:

1. Layer-level kernel beats the P45 scalar contract.
2. Layer-level output cosine stays >= 0.999999 against the current Triton path.
3. No promotion until multi-layer and full-generate gates pass.
