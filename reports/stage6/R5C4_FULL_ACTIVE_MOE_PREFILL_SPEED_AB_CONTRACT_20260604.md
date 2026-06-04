# Stage 6 R5-C4 — Full Active-MoE Prefill Speed A/B Contract

Date: 2026-06-04

Verdict: **contract only; no full active-MoE speed, decode TPS, server behavior,
RC quality, default runtime, or README speed claim is banked by this document.**

## Why This Gate Exists

R5-C3A recorded a positive gate/up-only FP4-MMA timing signal on R6000. R5-C3B
then materialized real CUTLASS D/ref gate-up values, and R5-C3C proved host
composition parity through SwiGLU, down projection, and weighted top-k
reduction.

That still does not prove a useful kernel. R5-C4 is the first gate allowed to
measure a **full active-MoE prefill** candidate as a speed result. It must cover
the whole active expert path:

```text
route/order -> grouped gate/up FP4-MMA -> SwiGLU -> down projection -> top-k weighted sum
```

## Scope

R5-C4 is an R6000 prefill/batch gate. It is not a Spark decode-speed gate and
must not be described as decode TPS.

The candidate must report at least two lanes:

| Lane | Purpose |
|---|---|
| Smoke lane | small selected-expert shape derived from R5-C3C, for fast fault isolation |
| Production-shape lane | Lynn active-MoE dimensions `H=2048`, `I=512`, `top_k=8`, with realistic token/expert counts |

The production-shape lane may be one layer and synthetic routed tokens, but it
must use the same layout/scale interpretation established by R5-A/R5-C and must
not time hidden BF16 shadow materialization as part of the candidate path.

The benchmark boundary is:

```text
hidden[T,H] + expert_ids[T,top_k] + routing_weights[T,top_k]
  + packed gate_up/down -> active_moe_out[T,H]
```

Router computation, shared expert, attention, full transformer prefill, decode,
server, and RC quality are out of scope.

## Required PASS Fields

```text
PASS_R5C4_FULL_ACTIVE_MOE_PREFILL_SPEED_AB
input_r5c3c_decision=PASS_R5C3C_DOWN_WEIGHTED_PARITY_SMOKE
same_scope_ab=true
real_model_weights=true
real_router_outputs=true
candidate_no_active_bf16_shadow=true
candidate_no_reload=true
candidate_no_bf16_weight_materialization=true
candidate_full_active_moe_boundary_timed=true
timing_includes_gateup_swiglu_down_weighted_scatter=true
numeric_vs_w4a16_or_p3_reference=true
candidate_median_speedup_vs_best_reference_ge_1p05=true
banked_full_active_moe_prefill_speed=true
banked_grouped_moe_fp4_mma_poc=true
banked_kernel_speed=true
banked_decode_tps=false
banked_server_rc=false
banked_default_promotion=false
banked_full_transformer_prefill=false
```

The artifact must include:

| Field | Requirement |
|---|---|
| `input_r5c3c_passed` | true; R5-C4 must build on banked R5-C3C parity |
| `same_scope_ab` | true; candidate and baselines use the same layer(s), hidden trace, routes, token counts, and model weights |
| `real_model_weights` / `real_router_outputs` | true for the production-shape lane |
| `candidate_no_bf16_shadow_in_timed_region` | true |
| `candidate_no_reload` | true |
| `candidate_no_bf16_weight_materialization` | true |
| `candidate_covers_gateup_swiglu_down_weighted` | true |
| `timing_includes_gateup_swiglu_down_weighted_scatter` | true |
| `numeric_max_abs` / `numeric_rel_l2` / `numeric_cosine` | reported against BF16 active and current packed/P3 reference where available |
| `route_order_preserved` | true for `(token_idx, top_k_slot, expert_id)` |
| `repack_cost_reported` | true; zero-copy vs explicit repack must be separated |
| `baseline_w4a16_ms` | measured with warmup/repeats |
| `baseline_packed_p3_ms` | measured or explicitly marked unavailable with reason |
| `candidate_ms` | measured with warmup/repeats |
| `speedup_vs_w4a16` | must be `>= 1.10` to bank speed |
| `speedup_vs_packed_p3` | must be reported; if `< 1.00`, result is diagnostic only |
| `median_speedup_vs_best_reference` | must be `>= 1.05` with no declared-shape regression |
| `fault_injections_detected` | route swap, expert swap, and value perturbation are rejected |

## Go / No-Go

- If numeric parity fails, the result is `FAILED_ARTIFACT`; do not widen prompts
  or report speed.
- If numeric parity passes but `speedup_vs_w4a16 < 1.10` or
  `median_speedup_vs_best_reference < 1.05`, the result is
  `DIAGNOSTIC_BANKED_SPEED_CLOSED`, not a speed win.
- If `speedup_vs_packed_p3 < 1.00`, report it as a partial R6000 FP4-MMA signal
  only; do not claim a Lynn prefill improvement.
- If the candidate hides BF16 shadow materialization, full-weight dequant, or
  scale repack inside the timed region without reporting it, the result is
  invalid.

## Explicit Non-Claims

- R5-C4 does not bank Spark decode TPS.
- R5-C4 does not bank server/RC behavior.
- R5-C4 does not bank default promotion.
- R5-C4 does not bank full transformer prefill.
- R5-C4 does not prove long-context quality.
- R5-C4 does not replace the separate MTP/compiled-loop speed tracks.

## Local Static Gate

```bash
python3 scripts/test_stage6_r5c4_candidate_from_metrics.py
python3 scripts/test_stage6_r5c4_full_active_moe_speed_contract_static.py
```

This static gate prevents gate/up-only timing traces or host composition parity
artifacts from being promoted as full active-MoE prefill speed evidence.

R6000 harnesses may either emit the canonical `CANDIDATE_JSON` directly or emit
raw same-scope metrics as `CANDIDATE_METRICS_JSON`; the wrapper normalizes the
latter through `scripts/stage6_r5c4_candidate_from_metrics.py` before running
the validator. `scripts/stage6_r5c4_candidate_metrics_template.json` is the
expected raw metrics shape for new kernel harnesses.
