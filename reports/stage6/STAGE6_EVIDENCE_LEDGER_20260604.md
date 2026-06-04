# Stage 6 Evidence Ledger

Date: 2026-06-04

Verdict: **evidence ledger only; this document does not bank new GPU results.**

This ledger separates banked positive evidence, intentional negative closures,
and gates that are wired but still waiting for Spark PASS/FAIL artifacts.

## Current Spark Status

| Field | Value |
|---|---|
| Status | `SPARK_P4C_GO_NOGO_DIAGNOSTIC_CAPTURED` |
| Note | P4C tile2 basic server smoke banked; rc-mini agreement rejected; shadow-cycle first-divergence kept arithmetic top-1 for 8 steps but showed hidden/logit drift, so P4C remains opt-in diagnostic only. |

## Promotion Boundaries

| Boundary | Value |
|---|---|
| P3 default promotion | `false` |
| P4 fused kernel banked | `false` |
| P4 default promotion | `false` |

## Gate Ledger

| Gate | Status | Evidence | Artifact | Next step |
|---|---|---|---|---|
| `stage6_60g_decode_release` | **BANKED** | report contains explicit pass evidence | `reports/stage6/DECODE_ONLY_SHADOW_FREE_SERVING_RECIPE.md` | Use as serving capability; high-throughput multi-request still needs packed prefill. |
| `p0_1_no_reload_prefill` | **BANKED** | report contains explicit pass evidence | `reports/stage6/P01_NO_RELOAD_SMOKE_20260603.md` | Replace slow stream_bf16 proof path with real grouped kernels. |
| `p0_2_resident_inventory` | **BANKED** | report contains explicit pass evidence | `reports/stage6/P02_RESIDENT_INVENTORY_20260603.md` | Prioritize projection/embed/lm-head residents after MoE shadow removal. |
| `p1_single_dense_projection` | **BANKED** | report contains explicit pass evidence | `reports/stage6/P1_DENSE_PROJECTION_POC_20260604.md` | Do not infer M>1 serving win from this M=1 gate. |
| `p1a_batched_dense_rejected` | **CLOSED_NEGATIVE** | report closes the scalar M>1 dense bridge as slower than BF16 | `reports/stage6/P1A_TILED_PROJECTION_SWEEP_20260604.md` | Keep as correctness probe; real M>1 dense win needs FP4-MMA/CUTLASS. |
| `p2_grouped_moe_census` | **BANKED** | report contains explicit pass evidence | `reports/stage6/P2_GROUPED_MOE_PREFILL_CENSUS_20260604.md` | Use routed grouped MoE as the real packed-prefill target. |
| `p2ka_gated_delta_loop_rejected` | **CLOSED_NEGATIVE** `passes.all=false` | latest result.json failed as an intentional negative gate | `reports/stage6/p2ka_gated_delta_native_loop_long_20260604_023700` | Use block linear-attn instead of token-loop decode recurrence. |
| `p2kb_block_linear_kernel` | **BANKED** `passes.all=true` | latest result.json has passes.all=true | `reports/stage6/p2kb_gated_delta_block_kernel_long_20260604_024710` | Keep composing into selected-layer prefill gates. |
| `p2l_linear_attn_block_integration` | **BANKED** `passes.all=true` | latest result.json has passes.all=true | `reports/stage6/p2l_linear_attn_block_integration_long_20260604_025348` | Combine with packed MoE in selected-layer gates. |
| `p2m_selected_layer_block_linear` | **BANKED** `passes.all=true` | latest result.json has passes.all=true | `reports/stage6/p2m_selected_layer_block_linear_20260604_030005` | Expand layer coverage before server RC. |
| `p2n_wider_layer_block_linear` | **BANKED** `passes.all=true` | latest result.json has passes.all=true | `reports/stage6/p2n_wider_layer_block_linear_20260604_030926` | Next real proof is P2-O/P3 server-shaped RC. |
| `p2o_packed_prefill_rc` | **READY_WAITING_SPARK** | tooling/runbook exists, but no Spark PASS artifact is banked | `reports/stage6/P2O_PACKED_PREFILL_RC_GATE_RUNBOOK_20260604.md, scripts/run_spark_stage6_p2o_rc_smoke.sh` | Run on Spark when SSH is reachable. |
| `p3b_selected_prefill` | **READY_WAITING_SPARK** | tooling/runbook exists, but no Spark PASS artifact is banked | `reports/stage6/P3B_SELECTED_PREFILL_GATE_RUNBOOK_20260604.md, scripts/run_spark_stage6_p3b_selected_prefill_gate.sh` | Run after P2-O basic/rc-mini plus P3-A predecessor evidence. |
| `p3c_resident_prompt` | **READY_WAITING_SPARK** | tooling/runbook exists, but no Spark PASS artifact is banked | `reports/stage6/P3C_RESIDENT_PROMPT_GATE_RUNBOOK_20260604.md, scripts/run_spark_stage6_p3c_resident_prompt_gate.sh` | Run after P3-B PASS. |
| `p3d_server_rc` | **READY_WAITING_SPARK** | tooling/runbook exists, but no Spark PASS artifact is banked | `reports/stage6/P3D_SERVER_RC_PROMOTION_GATE_RUNBOOK_20260604.md, scripts/run_spark_stage6_p3d_server_rc_gate.sh` | Run after P3-C PASS; banks opt-in server smoke only. |
| `p3e_rc_quality_battery` | **READY_WAITING_SPARK** | tooling/runbook exists, but no Spark PASS artifact is banked | `reports/stage6/P3E_RC_QUALITY_BATTERY_RUNBOOK_20260604.md, scripts/run_spark_stage6_p3e_rc_quality_battery.sh` | Run after P3-D PASS; full leaderboard/default promotion remains closed. |
| `p4_native_abi_preflight` | **READY_WAITING_SPARK** | tooling/runbook exists, but no Spark PASS artifact is banked | `reports/stage6/P4_NATIVE_FUSED_MOE_ABI_CONTRACT_20260604.md, scripts/run_spark_stage6_p4_native_abi_preflight.sh` | Run on Spark; may bank native ABI preflight only, not fused kernel. |
| `p4_runtime_bridge_preflight` | **READY_WAITING_SPARK** | tooling/runbook exists, but no Spark PASS artifact is banked | `reports/stage6/P4_NATIVE_FUSED_MOE_ABI_CONTRACT_20260604.md, scripts/run_spark_stage6_p4_runtime_bridge_preflight.sh` | Run on Spark; must prove native layer selection and no fallback. |
| `p4b_single_kernel_preflight` | **BANKED** `PASS_SINGLE_KERNEL_FAILLOUD_CONTRACT` | latest Spark result.json has passes.all=true | `reports/stage6/p4b_single_kernel_preflight_20260604_083106` | Run on Spark; may bank fail-loud single-kernel contract preflight only, not fused-kernel speed. |
| `p4b_runtime_bridge_preflight` | **BANKED** `PASS_P4B_RUNTIME_BRIDGE_FAILLOUD` | latest Spark result.json has passes.all=true | `reports/stage6/p4b_runtime_bridge_preflight_20260604_083152` | Run on Spark; proves resident-runner routing reaches P4B fail-loud symbol after active BF16 shadows are removed. |
| `p4b_single_cta_numeric_preflight` | **BANKED** `PASS_P4B_SINGLE_CTA_NUMERIC_REFERENCE` | latest Spark result.json has passes.all=true | `reports/stage6/p4b_single_cta_numeric_preflight_20260604_084451` | Use as correctness reference only; next gate needs byte-count profiler plus speed/RC evidence. |
| `p4b_single_cta_microbench` | **BANKED** `PASS_P4B_SINGLE_CTA_MICROBENCH_RECORDED` | latest Spark result.json has passes.all=true | `reports/stage6/p4b_single_cta_microbench_20260604_085842` | Treat as an intentional speed anti-proof for single-CTA; next implementation must be multi-CTA/CUTLASS-style. |
| `p4b_multi_cta_numeric_preflight` | **BANKED** `PASS_P4B_SINGLE_CTA_NUMERIC_REFERENCE` | latest Spark result.json has passes.all=true | `reports/stage6/p4b_multi_cta_numeric_preflight_20260604_091106` | If numeric passes, use microbench to accept or reject this recompute strategy. |
| `p4b_multi_cta_recompute_microbench` | **CLOSED_NEGATIVE** `PASS_P4B_SINGLE_CTA_MICROBENCH_RECORDED` | numeric exact but slower than P4A two-stage; speedup=0.005778x | `reports/stage6/p4b_multi_cta_microbench_20260604_091150` | Reject per-output-tile active recompute; next candidate must preserve active reuse. |
| `p4b_single_kernel_contract` | **REFERENCE_IMPL_BANKED_SPEED_CLOSED** | opt-in single-CTA output-returning numeric reference is banked; speed/default promotion remain closed | `reports/stage6/P4B_NATIVE_FUSED_SINGLE_KERNEL_CONTRACT_20260604.md, scripts/test_stage6_p4b_single_kernel_static.py, scripts/run_spark_stage6_p4b_single_cta_numeric_preflight.sh` | Scale beyond T=1/top_k=8, add byte-count profiler and speed gate before any promotion. |
| `p4c_active_reuse_runtime_bridge` | **BANKED** `PASS_P4C_ACTIVE_REUSE_RUNTIME_BRIDGE` | latest Spark result.json has passes.all=true | `reports/stage6/p4c_runtime_bridge_preflight_20260604_113325` | Run on Spark; banks P4C active-reuse route/numeric evidence only, not fused speed or default promotion. |
| `p4c_active_reuse_microbench` | **BANKED** `PASS_P4C_ACTIVE_REUSE_SPEED_BASELINE_RECORDED` | latest Spark result.json has passes.all=true | `reports/stage6/p4c_active_reuse_microbench_20260604_104254` | Run on Spark; banks the current P4C symbol speed baseline before replacing it with a real active-reuse speed candidate. |
| `p4c_component_profile` | **BANKED** `PASS_P4C_COMPONENT_PROFILE_RECORDED` | latest Spark result.json has passes.all=true | `reports/stage6/p4c_component_profile_20260604_105640` | Run on Spark; use the larger component to choose the first active-reuse speed candidate. |
| `p4c_gateup_shape_sweep` | **BANKED** `PASS_P4C_GATEUP_SHAPE_SWEEP_RECORDED` | latest Spark result.json has passes.all=true | `reports/stage6/p4c_gateup_shape_sweep_20260604_111603` | Run on Spark; bank only launch-shape diagnostic evidence before writing a real gate/up CUDA/CUTLASS candidate. |
| `p4c_gateup_shape_candidate` | **BANKED** `PASS_P4C_GATEUP_SHAPE_CANDIDATE_RECORDED` | latest Spark result.json has passes.all=true | `reports/stage6/p4c_gateup_shape_candidate_20260604_112635` | Run on Spark; compare current tile_inter=8 vs candidate tile_inter=2 on the full P4C active-reuse symbol. |
| `p4c_tile2_server_smoke` | **BANKED** `PASS_P4C_TILE2_SERVER_SMOKE` | latest Spark result.json has passes.all=true | `reports/stage6/p4c_tile2_server_smoke_20260604_122443` | Run on Spark; bank opt-in server evidence for tile_inter=2 P4C decode after prefill releases shadows. |
| `p4c_tile2_rcmini_agreement` | **CLOSED_NEGATIVE** `FAIL_P4C_TILE2_SERVER_SMOKE` | latest result.json failed as an intentional negative gate | `reports/stage6/p4c_tile2_rcmini_agreement_20260604_124019` | Run first-divergence diagnostics before widening P4C tile2 server prompts or considering promotion. |
| `p4c_tile2_shadow_cycle_first_divergence` | **DIAGNOSTIC_BANKED** `pass=true; first_top1_divergence=null` | server-like shadow cycle kept top-1 stable on the arithmetic prompt, but first hidden drift appears at step 0/layer 13 | `reports/stage6/p4c_tile2_shadow_cycle_first_divergence_20260604_130839` | Use as a go/no-go diagnostic only; do not promote P4C tile2 without wider RC exactness and e2e speed. |
| `p4c_active_reuse_decision` | **DECISION_BANKED** | single-CTA and multi-CTA recompute anti-proofs are captured; next candidate must preserve active reuse | `reports/stage6/P4C_ACTIVE_REUSE_KERNEL_DECISION_20260604.md, scripts/test_stage6_p4c_active_reuse_decision_static.py` | Implement P4C active-reuse two-phase/CUTLASS-style candidate; do not report it as P4B out-only fused speed. |

## Counts

| Status | Count |
|---|---:|
| `BANKED` | 20 |
| `CLOSED_NEGATIVE` | 4 |
| `DECISION_BANKED` | 1 |
| `DIAGNOSTIC_BANKED` | 1 |
| `READY_WAITING_SPARK` | 7 |
| `REFERENCE_IMPL_BANKED_SPEED_CLOSED` | 1 |

## Local Check

```bash
python3 scripts/write_stage6_evidence_ledger.py \
  --markdown-out reports/stage6/STAGE6_EVIDENCE_LEDGER_20260604.md \
  --json-out reports/stage6/stage6_evidence_ledger_20260604.json
```
