# Qwen3.6 35B W4A16 Native MoE P156 No-Fast-Math Gate/Up Probe

Date: 2026-05-19

## Purpose

P155 showed that native packed gate/up drifts at the raw FP32 accumulator stage.
P156 tests whether CUDA `--use_fast_math` or compiler algebra changes are the
cause by rebuilding the Lynn native extension with:

```bash
LYNN_NATIVE_CUDA_NO_FAST_MATH=1
```

The default native build remains unchanged; this is an opt-in diagnostic build.

## Result

Verdict: **no-fast-math does not change the outcome**

Artifact:
`reports/qwen36_35b/p156_native_packed_gateup_no_fast_math_20260519_042240.json`

| Check | P155 Default | P156 No Fast Math |
|---|---:|---:|
| native raw gate exact | 0/18 | 0/18 |
| native raw up exact | 0/18 | 0/18 |
| native inter exact | 6/18 | 6/18 |
| Triton-order raw gate exact | 0/18 | 0/18 |
| Triton-order raw up exact | 0/18 | 0/18 |
| Triton-order inter exact | 10/18 | 10/18 |
| raw max_abs | 9.536743e-7 | 9.536743e-7 |
| inter max_abs | 2.44140625e-4 | 2.44140625e-4 |

## Interpretation

The gate/up exactness blocker is not caused by CUDA fast-math flags. The
remaining likely cause is the exact reduction tree / operation ordering used by
Triton `tl.sum` over each 256-column block. Next work should build a slow
correctness-first native gate/up reference that explicitly mimics Triton
block-level accumulation, then optimize only after P155/P147 are exact.
