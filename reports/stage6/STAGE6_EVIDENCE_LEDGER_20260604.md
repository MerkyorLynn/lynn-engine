# Stage 6 Evidence Ledger

Date: 2026-06-04

Verdict: **evidence ledger only; this document does not bank new GPU results.**

This ledger separates banked positive evidence, intentional negative closures,
and gates that are wired but still waiting for Spark PASS/FAIL artifacts.

## Current Spark Status

| Field | Value |
|---|---|
| Status | `BLOCKED_BY_SPARK_SSH` |
| Note | FRP ssh-dgx closed at 2026-06-04 04:45:23 CST; jump ports 2222/2224 are not listening; N5 can reach Spark 192.168.100.26:22 at TCP/ping level but SSH banner exchange times out. See reports/stage6/SPARK_CONNECTIVITY_DIAGNOSTIC_20260604.md. |

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
| `p4b_single_kernel_contract` | **CONTRACT_READY_UNIMPLEMENTED** | fail-loud ABI/static gate exists, but no fused implementation or GPU result is banked | `reports/stage6/P4B_NATIVE_FUSED_SINGLE_KERNEL_CONTRACT_20260604.md, scripts/test_stage6_p4b_single_kernel_static.py` | Replace fail-loud contract with a real fused CUDA/CUTLASS kernel, then run byte-count/numeric/speed/RC gates. |

## Counts

| Status | Count |
|---|---:|
| `BANKED` | 9 |
| `CLOSED_NEGATIVE` | 2 |
| `CONTRACT_READY_UNIMPLEMENTED` | 1 |
| `READY_WAITING_SPARK` | 7 |

## Local Check

```bash
python3 scripts/write_stage6_evidence_ledger.py \
  --markdown-out reports/stage6/STAGE6_EVIDENCE_LEDGER_20260604.md \
  --json-out reports/stage6/stage6_evidence_ledger_20260604.json
```
