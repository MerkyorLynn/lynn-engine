# P79: NVCC FP4 MMA Target Matrix

Date: 2026-05-16

## Summary

P78 showed that the default torch CUDA extension target (`sm_120`) rejects CuTe
E2M1 FP4 MMA at `ptxas` time.

P79 isolates whether this is a global toolchain limitation or a target-feature
selection issue.

## Result

Report:

```text
reports/p16_155/p79_nvcc_fp4_mma_target_matrix.json
```

The installed CUDA stack advertises:

```text
nvcc --list-gpu-arch: ... compute_100 compute_101 compute_120
nvcc --list-gpu-code: ... sm_100 sm_101 sm_120
```

But direct compile probes reveal more than the public list:

| Target | Result | Notes |
|---|---:|---|
| `sm_120` | FAIL | `Feature '.kind::f8f6f4' not supported on .target 'sm_120'` |
| `compute_120` | PASS | PTX-only compile succeeds |
| `sm_120a` | PASS | Feature target accepts the CuTe E2M1 MMA source |
| `compute_120a` | PASS | Feature PTX target accepts it |
| `sm_120f` / `compute_120f` | FAIL | Unsupported architecture string |
| `sm_121*` / `compute_121*` | FAIL | Unsupported architecture string on this CUDA 12.8 stack |
| `sm_100a` / `sm_101a` | PASS | Also accept the source, but not runnable for R6000 sm_120 |

## Interpretation

This changes the P78 conclusion from "FP4 MMA is unavailable" to:

> FP4 MMA is not available under the default `sm_120` target, but the current
> compiler can emit the instruction for the architecture-specific `sm_120a`
> target.

That is an important distinction. The direct CUTLASS/CuTe path is not dead, but
it cannot be reached through PyTorch's default `sm_120` extension flags.

## Next Gate

P80 must answer the runtime question:

1. Can a torch extension force `sm_120a` without PyTorch also injecting a
   failing `sm_120` compile?
2. If it compiles, can the resulting code be loaded and executed on the R6000?
3. If it runs, grouped per-16 active expert FFN can proceed on the direct CuTe
   MMA route.
4. If it cannot run, the remaining practical routes are vendor library
   integration or a newer official toolchain/runtime stack.

## Decision

Proceed to P80 `sm_120a` runtime smoke before writing the grouped expert
kernel.

Do not continue tuning scalar CUDA wrappers as the main path; P75 already closed
that route as correctness scaffolding only.

