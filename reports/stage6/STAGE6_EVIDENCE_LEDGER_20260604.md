# Stage 6 Evidence Ledger

Date: 2026-06-04

Verdict: **evidence ledger only; this document does not bank new GPU results.**

This ledger separates banked positive evidence, intentional negative closures,
and gates that are wired but still waiting for lane-specific PASS/FAIL artifacts.

## Current Stage 6 Status

| Field | Value |
|---|---|
| Status | `R5C3C_DOWN_WEIGHTED_PARITY_BANKED` |
| Note | R5-C3C down + weighted top-k parity smoke is banked: real R5-C3B CUTLASS gate/up D/ref values were composed through host SwiGLU, deterministic down projection, and route-weighted top-k reduction with D/ref parity. This does not bank full active-MoE FP4-MMA speed, decode TPS, server behavior, RC quality, or default promotion. Next gate is full active-MoE prefill speed A/B against W4A16/P2-N/P3 paths. Spark decode ROI probe remains BORDERLINE_REMEASURE_OR_NSIGHT. |

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
| `p2o_basic_packed_prefill_rc` | **BANKED** `passes.all=true` | latest Spark result.json has passes.all=true | `reports/stage6/p2o_basic_packed_prefill_rc_smoke_20260604_132946` | Basic smoke is banked only as active-MoE no-reload correctness/memory evidence; do not treat the slow-mode prefill as a speed win. |
| `p2o_rcmini_nonlong_packed_prefill_rc` | **BANKED** `passes.all=true` | latest Spark result.json has passes.all=true | `reports/stage6/p2o_rc-mini_idx0-4_packed_prefill_rc_smoke_20260604_141640` | Banks only the rc-mini 0-4 shard; long-context remains a separate slow-mode scale gate. |
| `p2o_rcmini_packed_prefill_rc` | **TIMEOUT_NOT_CLEAN** `terminated_by_codex_timeout_after_17m_no_result` | latest Spark run was terminated after slow-mode opt-in produced no result.json | `reports/stage6/p2o_rc-mini_packed_prefill_rc_smoke_20260604_134826` | Rerun only after slow-mode prefill is accelerated or the rc-mini prompt set is split/chunked; the 2048 run was max_seq_len-invalid and the 8192 run timed out. |
| `p3a_grouped_moe_contract_probe` | **BANKED** `PASS` | latest Spark result.json has passes.all=true | `reports/stage6/p3a_layer0_grouped_moe_contract_probe_20260604_143854` | Banks only the P3-A active-MoE contract probe; speed/default/fused-kernel promotion remain closed. |
| `p3a_batched_down_candidate` | **DIAGNOSTIC_BANKED** `PASS_BUT_SPEED_GATE_CLOSED` | numeric/shadow gates pass, but average speed is 0.760x vs BF16 active | `reports/stage6/p3a_batched_down_layer0_grouped_moe_contract_probe_20260604_152955` | Do not promote; continue with route/materialization or true batched gate-up/down fusion. |
| `p3b_selected_prefill` | **BANKED** `PASS` | latest Spark result.json has passes.all=true | `reports/stage6/p3b_layers0-3_selected_prefill_gate_20260604_144842` | Banks selected-layer composition only; P3-C server behavior and full rc-mini long-context remain separate gates. |
| `p3c_resident_prompt` | **BANKED** `PASS` | latest Spark result.json has passes.all=true | `reports/stage6/p3c_basic_resident_prompt_gate_20260604_145940` | Banks resident-prompt basic smoke only; P3-D server behavior and RC quality remain separate gates. |
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
| `decode_gpu_idle_probe` | **DIAGNOSTIC_BANKED** `BORDERLINE_REMEASURE_OR_NSIGHT` | host-gap fraction 0.247, CUDA launches/token 1969.0 | `reports/stage6/decode_gpu_idle_probe_20260604_154648` | Use this ROI signal to choose MTP-light/compiled-loop prototype scope; do not treat it as speed promotion. |
| `r6000_fp4_mma_census` | **BANKED** `PASS_R6000_FP4_MMA_BRINGUP` | R6000 FP4-MMA census passed on NVIDIA RTX PRO 6000 Blackwell Server Edition, capability [12, 0] | `reports/stage6/r6000_fp4_mma_census_20260604_164457` | Start Lynn NVFP4 grouped-MoE FP4-MMA POC from CUTLASS/CuTe plus public Marlin/Machete census. |
| `r6000_grouped_moe_fp4_mma_poc_contract` | **CONTRACT_READY_UNIMPLEMENTED** | fail-loud ABI/static gate exists, but no fused implementation or GPU result is banked | `reports/stage6/R6000_GROUPED_MOE_FP4_MMA_POC_CONTRACT_20260604.md, scripts/test_stage6_r6000_grouped_moe_poc_contract_static.py` | Implement R5-A layout bridge first; do not start with a full fused MoE kernel. |
| `r5a_layout_bridge` | **DIAGNOSTIC_BANKED** `PASS_R5A_LAYOUT_BRIDGE_E8M0_REPACK_REQUIRED` | R5-A layout bridge banked; no kernel speed/default/grouped-MoE POC promotion; current Lynn E4M3-like scales require e8m0 repack/custom scale handling | `reports/stage6/r5a_layout_bridge_20260604_172706` | Use the bridge verdict to choose R5-B: e8m0 repack/custom scale path first, then CUTLASS/CuTe grouped-MoE POC. |
| `r5b_e8m0_repack` | **CLOSED_NEGATIVE** `FAIL_R5B_E8M0_REPACK_NUMERIC` | best rel_l2=0.165278, cosine=0.986248; e8m0 repack is not accurate enough | `reports/stage6/r5b_e8m0_repack_20260604_173435` | Do not pursue simple e8m0 repack; next route is custom scale/NVFP4-native CUTLASS/CuTe handling. |
| `r5c_cutlass_ue4m3_census` | **BANKED** `PASS_R5C_NVF4_UE4M3_CUTLASS_ABI` | CUTLASS/CuTe exposes sm120 mxf4nvf4 block16 E2M1 + UE4M3 scale path; no kernel speed/default/grouped-MoE POC promotion | `reports/stage6/r5c_cutlass_ue4m3_census_20260604_175216` | Proceed to R5-C1 minimal numeric GEMM smoke; do not jump directly to grouped-MoE speed claims. |
| `r5c1_cutlass_numeric_smoke` | **BANKED** `PASS_R5C1_CUTLASS_NVF4_UE4M3_NUMERIC_SMOKE` | CUTLASS 79d native NVF4+UE4M3 grouped GEMM ran Cooperative and Pingpong schedules with host-side verification and >=2 Disposition: Passed lines; no Lynn grouped-MoE kernel/speed/default promotion; recorded avg_runtime_ms=[0.021184, 0.01808] | `reports/stage6/r5c1_cutlass_numeric_smoke_20260604_181947` | Proceed to R5-C2 selected expert gate/up numeric smoke; do not jump directly to grouped-MoE speed claims. |
| `r5c2_moe_shape_census` | **BANKED** `PASS_R5C2_MOE_SHAPE_CENSUS_NEW_HARNESS_REQUIRED` | CUTLASS 79d has SM120 NVF4+UE4M3 generic grouped GEMM but lacks MoEProblemShape/tokens_per_expert; CUTLASS 92 has MoEProblemShape/tokens_per_expert but uses Sm100 schedules; new minimal harness is required | `reports/stage6/r5c2_moe_shape_census_20260604_183226` | Build R5-C2 selected expert gate/up numeric smoke by combining 92-style MoE shape semantics with 79d-style SM120 execution. |
| `r5c2_selected_expert_gateup_smoke` | **BANKED** `PASS_R5C2_SELECTED_EXPERT_GATEUP_NUMERIC_SMOKE` | CUTLASS 79d SM120 native NVF4+UE4M3 grouped GEMM ran a deterministic selected-expert gate/up shape with host-side reference verification; tokens_per_expert mapped to per-group M shapes; tokens_per_expert=[32, 64, 64, 96] | `reports/stage6/r5c2_selected_expert_gateup_smoke_20260604_192904` | Proceed to R5-C2B slot-preserving selected-output bridge; do not claim speed/default promotion from R5-C2. |
| `r5c2b_slot_bridge_contract` | **STATIC_CONTRACT_BANKED** `PASS_R5C2B_SLOT_PRESERVING_SELECTED_OUTPUT_CONTRACT` | route/order/scatter contract is banked as a GPU-free static guard for preserving (token_idx, top_k_slot, expert_id) through expert grouping and inverse scatter | `reports/stage6/R5C2B_SLOT_PRESERVING_SELECTED_OUTPUT_CONTRACT_20260604.md, scripts/test_stage6_r5c2b_slot_bridge_contract_static.py` | Implement the R5-C2B CUTLASS selected-output harness; do not claim speed/default from the static contract. |
| `r5c2c_real_d_row_slot_scatter_smoke` | **BANKED** `PASS_R5C2C_REAL_D_ROW_SLOT_SCATTER_SMOKE` | real CUTLASS D/ref row digests were captured after host-reference verification and scattered through the R5-C2B inverse-order contract; tokens/top_k/N=128/2/128; tokens_per_expert=[32, 64, 64, 96]; schedules=['cooperative', 'pingpong'] | `reports/stage6/r5c2c_real_d_row_slot_scatter_smoke_20260604_201440` | Proceed to R5-C3 grouped active-MoE prefill POC or first implement an in-epilogue selected-output scatter; do not claim speed/default from R5-C2C. |
| `r5c3b_gateup_value_materialization_smoke` | **BANKED** `PASS_R5C3B_GATEUP_VALUE_MATERIALIZATION_SMOKE` | full real CUTLASS D/ref row values were captured, exact value-bit digests matched, and values scattered through the R5-C2B inverse-order contract; tokens/top_k/N=128/2/128; schedules=['cooperative', 'pingpong']; scatter_max_abs=[0.0] | `reports/stage6/r5c3b_gateup_value_materialization_smoke_20260604_204920` | Proceed to R5-C3C down projection + weighted top-k numeric parity; do not claim full MoE speed/default from R5-C3B. |
| `r5c3c_down_weighted_parity_smoke` | **BANKED** `PASS_R5C3C_DOWN_WEIGHTED_PARITY_SMOKE` | real R5-C3B CUTLASS D/ref gate-up values were composed through host SwiGLU, deterministic down projection, and weighted top-k reduction with D/ref parity; tokens/top_k/hidden/out=128/2/64/48; schedules=['cooperative', 'pingpong']; weighted_max_abs=[0.0] | `reports/stage6/r5c3c_down_weighted_parity_smoke_20260604_130243` | Proceed to full active-MoE prefill speed A/B; do not claim speed/default from R5-C3C. |

## Counts

| Status | Count |
|---|---:|
| `BANKED` | 33 |
| `CLOSED_NEGATIVE` | 5 |
| `CONTRACT_READY_UNIMPLEMENTED` | 1 |
| `DECISION_BANKED` | 1 |
| `DIAGNOSTIC_BANKED` | 4 |
| `READY_WAITING_SPARK` | 4 |
| `REFERENCE_IMPL_BANKED_SPEED_CLOSED` | 1 |
| `STATIC_CONTRACT_BANKED` | 1 |
| `TIMEOUT_NOT_CLEAN` | 1 |

## Local Check

```bash
python3 scripts/write_stage6_evidence_ledger.py \
  --markdown-out reports/stage6/STAGE6_EVIDENCE_LEDGER_20260604.md \
  --json-out reports/stage6/stage6_evidence_ledger_20260604.json
```
