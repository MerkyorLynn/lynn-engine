# Stage 6 R5-C3B Gate/Up Value Materialization Smoke Summary

| Field | Value |
|---|---|
| Result | `/root/autodl-tmp/src/lynn-engine-r5c2c-codex/reports/stage6/r5c3b_gateup_value_materialization_smoke_20260604_204920/result.json` |
| Decision | `PASS_R5C3B_GATEUP_VALUE_MATERIALIZATION_SMOKE` |
| Selected tokens/top_k/experts | `128 / 2 / 4` |
| Tokens per expert | `[32, 64, 64, 96]` |
| Gate/up output width N | `128` |
| Temporary D-row value patch applied/restored | `True` / `True` |
| Gate/up value materialization banked | `True` |
| Host SwiGLU checksum smoke banked | `True` |
| Down projection numeric parity banked | `False` |
| Grouped-MoE FP4-MMA POC banked | `False` |
| Kernel speed banked | `False` |
| Default promotion banked | `False` |
| Avg runtime ms (trace only) | `[0.025824, 0.020608]` |

## Schedule Value Gates

| Schedule | Records | Row counts | Value digest match | Scatter max abs | SwiGLU checksum | Fault injections |
|---|---:|---|---:|---:|---:|---:|
| `cooperative` | `256` | `[32, 64, 64, 96]` | `True` | `0.0` | `-2572.67956982228` | `True` |
| `pingpong` | `256` | `[32, 64, 64, 96]` | `True` | `0.0` | `-2572.67956982228` | `True` |

## Boundary

- This R5-C3B artifact banks only `banked_gateup_value_materialization=true` and `banked_host_swiglu_checksum_smoke=true`.
- It emits full real CUTLASS D/ref row values and scatters them into `[T, top_k, N_gateup]` selected slots.
- It does not bank down projection, weighted top-k reduction, full grouped-MoE speed, decode TPS, server/RC behavior, or runtime default promotion.

