# Stage 6 R5-C NVF4 + UE4M3 CUTLASS Contract

Date: 2026-06-04

Verdict: **contract/tooling gate only; no grouped-MoE FP4-MMA kernel, speed
claim, or default runtime promotion is banked by this document.**

## Why R5-C Exists

R5-A proved that per-16 grouping can be preserved, but the current Lynn
E4M3-like scales were not zero-copy compatible with the tested block-scaled
layout. R5-B then closed the simple e8m0 repack route: best rel_l2 was about
0.166, so that repack is not accurate enough.

R5-C therefore changes direction: use CUTLASS/CuTe's native Blackwell
`mxf4nvf4` path with E2M1 values and UE4M3 block scales, instead of forcing
the Lynn scale tensor through e8m0.

## R5-C0 ABI Census Gate

The first R5-C gate is source/ABI census only. A PASS requires the local
CUTLASS checkout on the RTX PRO 6000 lane to expose all of:

| Requirement | Evidence token |
|---|---|
| sm120 native UE4M3 macro | `CUTE_ARCH_MXF4NVF4_4X_UE4M3_MMA_ENABLED` |
| E2M1 value format | `MXF4Format::E2M1` |
| UE4M3 scale format | `ScaleFormat::UE4M3` / `float_ue4m3_t` |
| sm120 specialization | `SM120_16x8x64_TN_VS<float_e2m1_t, float_e2m1_t, float, float_ue4m3_t` |
| sm120 asm route | `mxf4nvf4.block_scale.scale_vec::4X...ue4m3` |
| public POC anchors | CUTLASS NVFP4 grouped and MoE examples/tests are present |

The only allowed banked flag from R5-C0 is:

```text
PASS_R5C_NVF4_UE4M3_CUTLASS_ABI
banked_cutlass_abi=true
banked_grouped_moe_fp4_mma_poc=false
banked_kernel_speed=false
banked_default_promotion=false
```

## Rung Ladder

| Rung | Goal | Promotion boundary |
|---|---|---|
| R5-C0 | CUTLASS/CuTe native NVF4 + UE4M3 ABI census | ABI only; no compile/kernel/speed claim |
| R5-C1 | Minimal numeric GEMM smoke, M=16/64, N/K aligned to Lynn MoE tiles | Compare native result to dequant->bf16 reference; no grouped-MoE speed claim |
| R5-C2 | Selected expert gate/up native GEMM smoke | Preserve expert IDs/top-k and scale semantics |
| R5-C3 | Grouped active-MoE prefill POC | Only then may speed be measured against W4A16/P2-N/P3 paths |

## Commands

On the R6000 lane:

```bash
scripts/r6000_stage6_r5c_cutlass_ue4m3_census.sh
```

Local GPU-free check:

```bash
python3 scripts/test_stage6_r5c_cutlass_ue4m3_census_tools.py
```

## Explicit Non-Claims

- R5-C0 does not prove a Lynn grouped-MoE kernel exists.
- R5-C0 does not prove the path is faster than W4A16 or llama.cpp.
- R5-C0 does not change runtime defaults.
- R5-C0 exists only to prevent the next implementation from starting from the
  already-closed simple e8m0 repack route.
