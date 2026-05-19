# P189 R6000 FP4xFP8 Boundary Capability

**Date:** 2026-05-19  
**Model line:** Qwen3.5-9B Lynn-native NVFP4  
**Purpose:** decide whether the R6000 W4A8 packed boundary can be built on
`torch._scaled_mm`, or whether it must be a Lynn-owned CuTe/CUDA kernel.

## Result

`torch._scaled_mm` is not a usable ABI for the target mixed shape:

| Attempt | Result |
|---|---|
| FP4 activation x FP4 weight | OK, but this is W4A4, not the target |
| FP8 activation x FP8 weight | OK control |
| FP8 activation x FP4 weight, checkpoint layout | Fails with logical-K mismatch |
| FP8 activation x FP4 weight, expanded logical-K layout | Fails with invalid scaling configuration |

The installed CUTLASS/CuTe headers do expose the relevant R6000 instruction:

```text
mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e4m3.e2m1.f32
```

So the next path is a custom SM120a CuTe/inline-asm dense boundary, not more
PyTorch `_scaled_mm` wrapping.

## Decision

`TORCH_MIXED_FP4XFP8_UNAVAILABLE_CUTE_REQUIRED`

## Implication

For R6000:

- Do not promote the existing FP4-activation `native_fast_2d` path as W4A8.
- W4A8 quality remains useful, but the speed path needs A=E4M3 activation and
  B=E2M1 packed weight through a Lynn-owned mixed kernel.
- Claude/Kimi/Qwen helper streams are split into kernel PoC, offline repack,
  and admission gate respectively.

Artifact:

- `reports/qwen35_9b/p189_qwen35_9b_fp4x_fp8_scaled_mm_capability_20260519_142012_p189.json`
