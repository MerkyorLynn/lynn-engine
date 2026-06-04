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
| R5-C1 | Minimal numeric GEMM smoke with CUTLASS 79d native NVF4 + UE4M3 grouped GEMM | CUTLASS host-reference numeric smoke only; no Lynn grouped-MoE speed claim |
| R5-C2A | MoE shape/source census | Decide whether 79d/92 can be used directly or a new minimal harness is required |
| R5-C2 | Selected expert gate/up native GEMM smoke | Preserve expert IDs/top-k and scale semantics |
| R5-C2B | Slot-preserving selected-output bridge | Preserve `(token, top_k slot, expert)` through grouping and scatter-back |
| R5-C3 | Grouped active-MoE prefill POC | Only then may speed be measured against W4A16/P2-N/P3 paths |

## R5-C1 Minimal Numeric GEMM Smoke Gate

R5-C1 must run a real CUTLASS kernel, not only scan source. The allowed anchor
is a minimal numeric GEMM smoke built from:

```text
examples/79_blackwell_geforce_gemm/79d_blackwell_geforce_nvfp4_grouped_gemm.cu
```

A PASS requires:

```text
PASS_R5C1_CUTLASS_NVF4_UE4M3_NUMERIC_SMOKE
banked_numeric_smoke=true
banked_grouped_moe_fp4_mma_poc=false
banked_kernel_speed=false
banked_default_promotion=false
```

The run must show both Cooperative and Pingpong schedules, host-side reference
verification, at least two `Disposition: Passed` lines, and no device-gate
no-op. Runtime/TFLOPS may be recorded for traceability, but R5-C1 does **not**
bank speed.

## R5-C2A MoE Shape Census Gate

R5-C2A is the source-census bridge between R5-C1's generic grouped GEMM and
the actual selected-expert gate/up smoke. A PASS requires evidence that:

```text
PASS_R5C2_MOE_SHAPE_CENSUS_NEW_HARNESS_REQUIRED
banked_moe_shape_census=true
requires_new_minimal_harness=true
banked_selected_expert_gate_up_smoke=false
banked_grouped_moe_fp4_mma_poc=false
banked_kernel_speed=false
banked_default_promotion=false
```

Expected source split:

- CUTLASS 79d has the SM120 native NVF4 + UE4M3 grouped-GEMM route, but lacks
  `MoEProblemShape` and `tokens_per_expert`.
- CUTLASS 92 has `MoEProblemShape` and `tokens_per_expert` semantics, but uses
  Sm100 schedules.
- Therefore R5-C2 implementation must combine 92-style MoE shape semantics with
  79d-style SM120 NVF4 + UE4M3 execution. R5-C2A does **not** bank selected
  expert gate/up numeric smoke.

## R5-C2 Selected-Expert Gate/Up Numeric Smoke Gate

R5-C2 is the first selected-expert bridge. It maps a deterministic top-k route
to `tokens_per_expert`, then encodes each expert as one CUTLASS 79d grouped-GEMM
problem:

```text
M = tokens_per_expert[e]
N = gate/up output width
K = hidden input width
```

A PASS requires:

```text
PASS_R5C2_SELECTED_EXPERT_GATEUP_NUMERIC_SMOKE
banked_selected_expert_gate_up_smoke=true
banked_grouped_moe_fp4_mma_poc=false
banked_kernel_speed=false
banked_default_promotion=false
```

The gate must prove route/count consistency (`tokens * top_k ==
sum(tokens_per_expert)`, unique experts per token, benchmark groups matching
experts), 32-element alignment so 79d does not silently round the smoke shape,
both Cooperative and Pingpong schedules passing host-reference verification,
and no no-op device gate.

R5-C2 still does **not** bank a full Lynn MoE kernel: it does not implement
slot-preserving gather/scatter, down projection, router logits, or end-to-end
decode. It only proves that the selected-expert gate/up shape can be represented
on the SM120 native NVF4 + UE4M3 CUTLASS path with numeric verification.

The next gate is R5-C2B, not R5-C3: R5-C2B must preserve `token_idx`,
`top_k_slot`, `expert_id`, per-expert prefix offsets, and inverse scatter order,
then compare a slot-preserving `[T, top_k, inter]` host reference.

## Commands

On the R6000 lane:

```bash
scripts/r6000_stage6_r5c_cutlass_ue4m3_census.sh
scripts/r6000_stage6_r5c1_cutlass_numeric_smoke.sh
scripts/r6000_stage6_r5c2_moe_shape_census.sh
scripts/r6000_stage6_r5c2_selected_expert_gateup_smoke.sh
```

Local GPU-free check:

```bash
python3 scripts/test_stage6_r5c_cutlass_ue4m3_census_tools.py
python3 scripts/test_stage6_r5c1_cutlass_numeric_smoke_tools.py
python3 scripts/test_stage6_r5c2_moe_shape_census_tools.py
python3 scripts/test_stage6_r5c2_selected_expert_gateup_smoke_tools.py
```

## Explicit Non-Claims

- R5-C0 does not prove a Lynn grouped-MoE kernel exists.
- R5-C0 does not prove the path is faster than W4A16 or llama.cpp.
- R5-C0 does not change runtime defaults.
- R5-C0 exists only to prevent the next implementation from starting from the
  already-closed simple e8m0 repack route.
- R5-C1 proves only a CUTLASS native NVF4 + UE4M3 numeric smoke with host
  reference verification.
- R5-C1 does not prove a Lynn selected-expert gate/up kernel exists.
- R5-C1 does not bank grouped-MoE FP4-MMA speed or runtime defaults.
- R5-C2A proves only the source-shape reason for a new minimal harness.
- R5-C2A does not prove selected expert gate/up numeric smoke, grouped-MoE
  FP4-MMA speed, or runtime defaults.
- R5-C2 proves only selected-expert gate/up numeric smoke via per-group
  `tokens_per_expert` shapes and CUTLASS host-reference verification.
- R5-C2 does not prove Lynn slot-preserving gather/scatter, down projection,
  full grouped-MoE FP4-MMA speed, or runtime defaults.
