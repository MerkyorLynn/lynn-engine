# Stage 6 R5-C3C — Down Projection + Weighted Top-K Parity Contract

Date: 2026-06-04

Verdict: **numeric composition gate only; no full active-MoE FP4-MMA speed,
decode TPS, server behavior, RC quality, or default promotion is banked.**

## Why This Gate Exists

R5-C3B proved that real R6000/CUTLASS gate/up D/ref values can be scattered into
Lynn selected slots and passed through host SwiGLU. Full active-MoE still needs
two more numeric pieces before any speed claim is meaningful:

1. down projection after SwiGLU;
2. route-weighted top-k reduction across selected experts.

R5-C3C consumes the R5-C3B materialized value artifact and verifies those two
pieces for CUTLASS D values against the CUTLASS host reference values.

## Scope

R5-C3C is a host composition parity smoke over real R5-C3B gate/up rows. The
down weights and route weights are deterministic smoke weights, not model
weights. A valid PASS must:

1. consume a banked `PASS_R5C3B_GATEUP_VALUE_MATERIALIZATION_SMOKE` artifact;
2. reconstruct Lynn route order and the `[T, top_k, N_gateup]` selected slots;
3. compute host SwiGLU for both D and reference selected values;
4. apply deterministic expert-specific down projection;
5. apply deterministic route weights for top-k reduction;
6. prove D/ref parity after SwiGLU, down projection, and weighted top-k.

## Required PASS Fields

```text
PASS_R5C3C_DOWN_WEIGHTED_PARITY_SMOKE
banked_down_projection_numeric_parity=true
banked_weighted_topk_numeric_parity=true
banked_grouped_moe_fp4_mma_poc=false
banked_kernel_speed=false
banked_default_promotion=false
```

The result must also include:

| Field | Requirement |
|---|---|
| `swiglu_d_ref_match` | true for every captured schedule |
| `down_projection_d_ref_match` | true for every captured schedule |
| `weighted_topk_d_ref_match` | true for every captured schedule |
| `weighted_topk_hash_match` | true for every captured schedule |
| `fault_injections_detected` | selected-value perturbation and route-weight swap are detected |

## Explicit Non-Claims

- R5-C3C does not bank full active-MoE FP4-MMA speed.
- R5-C3C does not bank a CUDA down-projection kernel.
- R5-C3C does not bank model-weight parity.
- R5-C3C does not bank decode TPS.
- R5-C3C does not bank server/RC behavior.
- R5-C3C does not change runtime defaults.

## Next Gate

After R5-C3C, the next evidence gate may measure a full active-MoE prefill
candidate against W4A16/P2-N/P3 paths. That speed gate must use real kernel
timing and must not reuse R5-C3C host composition parity as a speed claim.

## Local Static Gate

```bash
python3 scripts/test_stage6_r5c3c_down_weighted_parity_contract_static.py
```

This static gate prevents R5-C3C parity evidence from being promoted as full
active-MoE speed or default-runtime evidence.
