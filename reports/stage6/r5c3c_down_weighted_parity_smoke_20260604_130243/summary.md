# Stage 6 R5-C3C Down + Weighted Top-K Parity Smoke Summary

| Field | Value |
|---|---|
| Result | `reports/stage6/r5c3c_down_weighted_parity_smoke_20260604_130243/result.json` |
| Decision | `PASS_R5C3C_DOWN_WEIGHTED_PARITY_SMOKE` |
| Input R5-C3B decision | `PASS_R5C3B_GATEUP_VALUE_MATERIALIZATION_SMOKE` |
| Selected tokens/top_k/experts | `128 / 2 / 4` |
| SwiGLU hidden / down out dim | `64 / 48` |
| Down projection numeric parity banked | `True` |
| Weighted top-k numeric parity banked | `True` |
| Grouped-MoE FP4-MMA POC banked | `False` |
| Kernel speed banked | `False` |
| Default promotion banked | `False` |

## Schedule Parity Gates

| Schedule | Records | SwiGLU max abs | Down max abs | Weighted max abs | Weighted hash match | Fault injections |
|---|---:|---:|---:|---:|---:|---:|
| `cooperative` | `256` | `0.0` | `0.0` | `0.0` | `True` | `True` |
| `pingpong` | `256` | `0.0` | `0.0` | `0.0` | `True` | `True` |

## Boundary

- This R5-C3C artifact banks only `banked_down_projection_numeric_parity=true` and `banked_weighted_topk_numeric_parity=true`.
- It consumes real R5-C3B CUTLASS gate/up D/ref values, then runs host SwiGLU, deterministic down projection, and route-weighted top-k reduction.
- It does not bank full active-MoE FP4-MMA speed, decode TPS, server/RC behavior, or runtime default promotion.

