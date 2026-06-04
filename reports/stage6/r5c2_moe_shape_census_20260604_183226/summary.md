# Stage 6 R5-C2 MoE Shape Census Summary

| Field | Value |
|---|---|
| Result | `/root/autodl-tmp/src/lynn-engine-r5c2-codex/reports/stage6/r5c2_moe_shape_census_20260604_183226/result.json` |
| Decision | `PASS_R5C2_MOE_SHAPE_CENSUS_NEW_HARNESS_REQUIRED` |
| CUTLASS dir | `/root/autodl-tmp/src/cutlass` |
| CUTLASS git | `2599f2975b06a67d5ee25e4a7292afeda1475c9b` (`main`) |
| MoE shape census banked | `True` |
| Requires new minimal harness | `True` |
| Selected expert gate/up smoke banked | `False` |
| Grouped-MoE FP4-MMA POC banked | `False` |
| Kernel speed banked | `False` |
| Default promotion banked | `False` |

## Key Source Split

| Source | Evidence |
|---|---|
| CUTLASS 79d | SM120 native NVF4+UE4M3 generic grouped GEMM; lacks `MoEProblemShape` and `tokens_per_expert`. |
| CUTLASS 92 | Has `MoEProblemShape` + `tokens_per_expert` and NVF4+UE4M3, but uses Sm100 schedules. |

## Boundary

- This artifact banks only `banked_moe_shape_census=true`.
- It does not bank selected-expert gate/up numeric smoke, grouped-MoE speed, or default promotion.
- R5-C2 implementation must combine 92-style MoE shape semantics with 79d-style SM120 execution.

