# Lynn engine — MTP Phase 1 + FP8 Phase 2 session handoff (final) · 2026-05-20

## TL;DR

* **MTP Phase 1 closed**. M21–M26 chain ruled out lm_head, KV restore, commit outputs, captured tensor state, and CUDA allocator as the K=2 batched 3/6 exact bug source — residue is process-side and not lever-able this session. Sequential `spec_k1` 6/6 at 26.4 TPS ships as the only correctness-clean K-step path. All 8 diagnostic switches stay opt-in (default OFF).
* **FP8 Phase 2 build complete** (5 parallel CLI dispatch + my integration). Kernel benchmarks on Spark sm_121:
  * dense fused gate/up + SwiGLU: **3–4× over BF16** at M=1 K=2048 N=6144 (peak 6.0× with autotuned blocks)
  * MoE expert variant: **1.82–2.10× over BF16** at MoE shapes (intermediate=1408)
  * autotune sweep: best universal config `(BLOCK_M=16, BLOCK_K=128, BLOCK_N=32)`, N=256 always memory-bound (fallback BF16 recommended)
* **End-to-end TPS measurement blocked on ONE thing**: claude-internal's repack V1 deferred 3D MoE expert weight repack (gate_up_proj / down_proj at `[E, ...]` layouts). Without these, the full Lynn-native FP8 model dir cannot be produced.

## Main branch state

HEAD: `836da56` (origin/main).

Commit chain (latest first):
```
836da56 Merge origin/claude/fp8-repack-v1-20260520   ← claude-internal repack V1 full-dir
2222a7a fp8: phase 2 step 3.1 — MoE forward FP8 path (codebuddy MoE kernel)
eae8a79 Merge origin/qwen/fp8-autotune-sweep-20260520
e82db33 Merge origin/codebuddy/fp8-moe-expert-kernel-20260520
12188f4 Merge origin/codebuddy/fp8-tps-smoke-harness-20260520
e0e82d5 Merge origin/qwen/fp8-quality-regression-20260520
dac9e7d fp8: phase 2 step 3 — engine integration scaffold (loader + dense FFN)
8e43809 docs: dispatch briefs for FP8 parallel CLIs
7401a32 fp8: spark sm_121 FP8 fused gate/up + SwiGLU Triton kernel v0
2431595 reports: session handoff — MTP Phase 1 closed, FP8 Phase 2 opened
ca77375 fp8: spark_pack_w4a8_fp8 v0 — FP64 cosine verification fix
e370cbc fp8: phase 2 step 1 — offline NVFP4 → FP8 E4M3 repack tool v0
3d697ad reports: MTP Phase 1 closing
[... M21–M26 commits ...]
```

## What landed (file-level)

### Kernels (Spark sm_121 FP8 MMA)
* `triton_kernels/spark_fp8_gate_up_fused.py` — dense fused gate/up + SwiGLU
* `triton_kernels/spark_fp8_moe_expert_fused.py` — MoE expert variant (per-expert routing, smaller intermediate)

### Engine integration
* `engine/loader.py` — `_is_lynn_variable_w4a8_fp8` + `_load_qwen36_layer_lynn_variable_w4a8_fp8` (manifest schema `lynn-variable-w4a8-fp8-v1`)
* `engine/full_forward.py::_dense_ffn_forward` — FP8 branch keyed on `mlp.gate_proj.weight_fp8` presence → fused gate/up kernel + `torch._scaled_mm` for down
* `engine/full_forward.py::_moe_forward` — FP8 branch keyed on `mlp.experts.gate_up_proj.weight_fp8` → per-expert MoE kernel + shared expert FP8 path

### Tools
* `scripts/spark_pack_w4a8_fp8.py` — offline NVFP4 → FP8 repack
  * `self-test`: synthetic NVFP4 verification (PASS cos > 0.999 across 4 shapes × 2 granularities)
  * `one`: single-weight CLI
  * `full-dir`: full model dir manifest-driven repack (**3D MoE weights deferred** to manifest's `deferred_tensors` map; see V2 work below)
* `scripts/spark_fp8_kernel_verify.py` — dense kernel verify + bench
* `scripts/spark_fp8_moe_expert_verify.py` — MoE kernel verify + bench
* `scripts/spark_fp8_kernel_autotune_sweep.py` — block-size sweep harness
* `scripts/spark_fp8_e2e_tps_smoke.py` — W4A16 vs W4A8 end-to-end TPS smoke
* `scripts/spark_w4a8_vs_w4a16_quality_regression.py` — MMLU-100 + GPQA-50 eval

### Reports
* `reports/mtp/QWEN36_FP8_AUTOTUNE_SWEEP_RESULT_20260520.md` (qwen)
* `reports/mtp/QWEN36_FP8_QUALITY_REGRESSION_PLAN_20260520.md` (qwen)
* `reports/mtp/QWEN36_MTP_PHASE2_SESSION_HANDOFF_20260520.md` (early Phase 2 plan)
* `reports/mtp/_DISPATCH_FP8_PARALLEL_CLI_BRIEFS_20260520.md` (original CLI dispatch)
* `reports/mtp/spark_fp8_autotune_sweep_TS.json` (full sweep data, 2160 configs)
* `reports/mtp/QWEN36_MTP_PHASE1_CLOSING_REPORT_20260520.md` (MTP Phase 1 close)

### M21–M26 MTP probe trail (closed)
* `mtp_m21_residual_exact_20260520_114144.json`
* `mtp_m22_reject_rollback_20260520_123627.json`
* `mtp_m24_commit_repair_20260520_130415.json`
* `mtp_m26_no_expandable_20260520_134725.json`
* MD trails in `reports/mtp/QWEN36_MTP_M{21,22,24,25,26}_*_RESULT_*.md`

## Spark TPS state (per memory)

| Path | TPS | Source |
|---|---:|---|
| Lynn baseline graph (W4A16 NVFP4, current ship) | **38.96** | `project_spark_single_stream_tps_baseline_20260518` |
| Lynn spec_k1 sequential 6/6 (eager) | 26.4 | M22 / M24 / M26 smoke |
| Lynn spec_k1_batched 3/6 (eager, NOT promoted) | 24.5 | M22 / M24 / M26 |
| SGLang BF16 single-stream | 30.14 | memory |
| **SGLang FP8 + MTP single-stream (target)** | **60–70** | `reference_dgx_spark_llm_candidates_0501.md:94` |

Phase 2 kernel speedups (Spark sm_121 GB10, torch 2.11 + triton 3.6):

| Shape (M, K, N) | Kernel | Speedup vs BF16 |
|---|---|---:|
| 1, 2048, 6144 | dense fused | 4.51× |
| 1, 6144, 2048 | dense fused | 3.40× |
| 8, 2048, 6144 | dense fused | 3.46× |
| 1, 2048, 6144 | dense + autotune (16,64,32) | 6.0× |
| MoE expert N=1408 | MoE variant | 1.82–2.10× |
| MoE expert N=768 | MoE variant | 1.25–1.89× |
| N=256 (any) | — | memory-bound, BF16 fallback recommended |

## Critical V2 work (next-session priority)

### 1. Repack V1 → V2: handle 3D MoE expert weights

`scripts/spark_pack_w4a8_fp8.py` `full-dir` currently writes a `deferred_tensors` map for 3D shapes (`mlp.experts.gate_up_proj`, `mlp.experts.down_proj`). Need:

* Read flattened storage shape `[prod(original_shape[:-1]), K/2]` → unpack to FP4 → dequant → reshape to logical 3D `[E, N, K]` → cast to FP8 per-expert + per-row.
* Output FP8 weight tensor `[E, N, K]` + scale `[E, N]` (or scalar per expert).
* Update manifest's `quantized_tensors` map with these (not `deferred_tensors`).

The MoE kernel `triton_kernels/spark_fp8_moe_expert_fused.py` already expects this layout (`expert_id` selects slice from `[E, N, K]`).

### 2. End-to-end TPS smoke

Once repack V2 produces a full FP8 model dir:

```bash
python scripts/spark_pack_w4a8_fp8.py full-dir \
  --input /home/merkyor/models/Qwen3.6-35B-A3B-lynn-native-w4a16-nvfp4-from-r6000 \
  --output /home/merkyor/models/Qwen3.6-35B-A3B-lynn-native-w4a8-fp8 \
  --scale-granularity per_row

python scripts/spark_fp8_e2e_tps_smoke.py \
  --w4a16-model-dir /home/merkyor/models/Qwen3.6-35B-A3B-lynn-native-w4a16-nvfp4-from-r6000 \
  --w4a8-model-dir  /home/merkyor/models/Qwen3.6-35B-A3B-lynn-native-w4a8-fp8 \
  --out reports/mtp/spark_fp8_e2e_$(date +%s).json
```

Engine auto-selects FP8 path via loader manifest schema. No env knob needed (opt-out via `LYNN_DISABLE_W4A8_FP8_PATH=1`).

### 3. Apply autotune-recommended block sizes

Update default `BLOCK_M=16, BLOCK_K=64, BLOCK_N=128` in `spark_fp8_gate_up_fused.py` to autotune-recommended `(16, 128, 32)`. Memory `QWEN36_FP8_AUTOTUNE_SWEEP_RESULT_20260520.md` has per-shape best configs for further specialization.

### 4. Quality regression

Run `scripts/spark_w4a8_vs_w4a16_quality_regression.py` on both dirs. Acceptance gates (per qwen plan MD):
* MMLU Δ ≤ 1pp (HARD)
* GPQA Δ ≤ 2pp (HARD)

Memory baseline: W4A16 NVFP4 MMLU 84.40 / GPQA 49.49.

## Estimated TPS lift forecast

If V2 MoE repack works + kernel autotune applied + integration is correct end-to-end:

* dense FFN: 3-4× → assume ~3.5× actual after Python/launch overhead
* MoE expert: 2× → assume ~1.8× actual
* MoE layers dominate Lynn 35B-A3B forward (30/40 layers MoE + 10 full-attn)
* Per-layer FFN cost was ~50% of layer time on Spark (rough estimate from M22 step timings)
* End-to-end speedup forecast: **~1.4–1.8× baseline 38.96 = ~55–70 TPS**

That LANDS in the SGLang FP8+MTP 60–70 target range if everything plays. Worst case ~55 TPS still beats SGLang BF16 (30) by ~80%.

## Branches still on origin (uncleaned)

```
origin/claude/mtp-k2-strict-diag-20260520   ← my branch (where I integrated)
origin/claude/fp8-repack-v1-20260520        ← claude-internal repack
origin/codebuddy/fp8-tps-smoke-harness-20260520
origin/codebuddy/fp8-moe-expert-kernel-20260520
origin/qwen/fp8-quality-regression-20260520
origin/qwen/fp8-autotune-sweep-20260520
```

All 6 already merged into main. Safe to delete once next session confirms end-to-end works.

## Cross-reference

* MTP Phase 1 closing: `reports/mtp/QWEN36_MTP_PHASE1_CLOSING_REPORT_20260520.md`
* Phase 2 strategy memory: `reference_spark_fp8_w4a8_design_strategy_20260519`
* T=1 kernel contract memory: `project_lynn_engine_t1_only_kernel_contract_20260519`
* Spark TPS baseline memory: `project_spark_single_stream_tps_baseline_20260518`
* SGLang target memory: `reference_dgx_spark_llm_candidates_0501`
