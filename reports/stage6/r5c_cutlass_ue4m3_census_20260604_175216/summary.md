# Stage 6 R5-C CUTLASS UE4M3 Census Summary

| Field | Value |
|---|---|
| Result | `/root/autodl-tmp/src/lynn-engine-r5c-github-20260604_175209/reports/stage6/r5c_cutlass_ue4m3_census_20260604_175216/result.json` |
| Decision | `PASS_R5C_NVF4_UE4M3_CUTLASS_ABI` |
| CUTLASS dir | `/root/autodl-tmp/src/cutlass` |
| CUTLASS git | `2599f2975b06a67d5ee25e4a7292afeda1475c9b` (`main`) |
| CUTLASS ABI banked | `True` |
| Grouped-MoE FP4-MMA POC banked | `False` |
| Kernel speed banked | `False` |
| Default promotion banked | `False` |

## Required Tokens

| Gate | Pass |
|---|---:|
| `sm120_ue4m3_macro_seen` | `True` |
| `scale_format_ue4m3_seen` | `True` |
| `scale_type_ue4m3_seen` | `True` |
| `mxf4_e2m1_format_seen` | `True` |
| `sm120_e2m1_ue4m3_specialization_seen` | `True` |
| `sm120_mxf4nvf4_ue4m3_asm_seen` | `True` |
| `expected_examples_seen` | `True` |
| `sm120_tests_seen` | `True` |

## Evidence Snippets

| Source | Line | Text |
|---|---:|---|
| `config` | 165 | `#    define CUTE_ARCH_MXF4NVF4_4X_UE4M3_MMA_ENABLED` |
| `scale_desc` | 205 | `case MXF4Format::E2M1:   return "E2M1";` |
| `scale_desc` | 212 | `if constexpr (is_same_v<T, float_e2m1_t>) { return MXF4Format::E2M1;  } else` |
| `scale_desc` | 223 | `case ScaleFormat::UE4M3:   return "UE4M3";` |
| `scale_desc` | 231 | `if constexpr (is_same_v<T, float_ue4m3_t>) { return ScaleFormat::UE4M3;  } else` |
| `scale_desc` | 400 | `if constexpr (is_same_v<T, float_e2m1_t>) { return MXF4Format::E2M1;  } else` |
| `sm120_mma` | 3137 | `"mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue8m0 "` |
| `sm120_mma` | 3187 | `struct SM120_16x8x64_TN_VS<float_e2m1_t, float_e2m1_t, float, float_ue4m3_t, VS>` |
| `sm120_mma` | 3216 | `"mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "` |

## Boundary

- This summary may bank only `banked_cutlass_abi=true`.
- `banked_grouped_moe_fp4_mma_poc`, `banked_kernel_speed`, and `banked_default_promotion` must remain false.
- A PASS means the next step is a minimal numeric GEMM smoke using CUTLASS/CuTe native NVF4 + UE4M3, not a grouped-MoE speed claim.

