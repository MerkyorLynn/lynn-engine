# P78: sm_120 E2M1 MMA Smoke Blocked By Current Toolchain

Date: 2026-05-16

## Summary

P76 proved that the R6000 host can compile a Lynn torch CUDA extension against
the CUTLASS/CuTe headers bundled through `deep_gemm`.

P77 proved that Lynn native E2M1 nibble storage is exactly compatible with
CUTLASS `float_e2m1_t::bitcast`.

P78 was the next smallest gate: execute one CuTe Blackwell FP4 MMA instruction
from inside a Lynn torch CUDA extension.

That gate failed at `ptxas`.

## Result

Report:

```text
reports/p16_155/p78_sm120_e2m1_mma_smoke.json
```

Host/runtime:

```text
torch: 2.10.0+cu128
cuda capability: [12, 0]
include path: /root/miniconda3/lib/python3.12/site-packages/deep_gemm/include
```

The smoke used:

```cpp
cute::SM120_16x8x32_TN<
    cute::float_e2m1_t,
    cute::float_e2m1_t,
    float>::fma(...);
```

The compile reached `ptxas`, then failed with:

```text
Instruction 'mma with with FP6/FP4 floating point type' not supported on .target 'sm_120'
Feature '.kind::f8f6f4' not supported on .target 'sm_120'
```

## Interpretation

This is not a header discovery failure and not a Lynn packed-weight encoding
failure.

What is green:

- CUTLASS/CuTe headers are present.
- CuTe exposes SM120 FP4 MMA wrappers.
- Lynn E2M1 nibble values are storage-compatible with CUTLASS E2M1.
- The torch extension build path can invoke `nvcc`/`ptxas`.

What is red:

- The current CUDA 12.8 `sm_120` target on this R6000 rejects the actual
  `.kind::f8f6f4` MMA instruction.

That means the direct CuTe FP4 MMA route is not ready to become the P79 grouped
per-16 active expert kernel on this exact toolchain.

## Decision

Do not start a large grouped active-expert kernel assuming direct
`SM120_16x8x32_TN<float_e2m1_t, float_e2m1_t, ...>` will compile on the current
R6000 stack.

First isolate the target/toolchain contract:

1. Probe which `sm_12x` / feature-suffixed targets this `nvcc` supports.
2. Check whether a newer CUDA or vendor library exposes an accepted target for
   Blackwell FP4 MMA.
3. Keep the Lynn-native exact scalar/Triton route as the correctness scaffold.
4. Treat vendor-compatible NVFP4 as a separate artifact format, not a silent
   replacement for Lynn-native per-16 quantization.

## Why This Matters

The 155 TPS route still points at active routed experts, but P78 narrows the
implementation choices:

- A pure scalar CUDA wrapper is correctness-safe but too small a speedup.
- Direct CuTe FP4 MMA is blocked by the current `sm_120` target.
- The next practical step is a target matrix / vendor-kernel probe, not another
  blind kernel rewrite.

