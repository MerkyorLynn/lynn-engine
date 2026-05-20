# FP8 Phase 2 V2 — 5-CLI parallel dispatch · 2026-05-20 (afternoon)

Goal: break SGLang FP8+MTP 60-70 TPS target on Spark sm_121 with the
Lynn engine FP8 path. V1 build complete (`358bc69`); V2 closes the
end-to-end gate and adds three parallel kernel/eval optimisations.

Main HEAD: `358bc69` (origin/main). All briefs branch from there.
This Claude (orchestrator) session has already landed Task #2 on a
follow-up commit — see "Already done" section.

---

## Task allocation matrix

| # | Task | CLI | Branch | Priority | Blocks |
|---|---|---|---|---|---|
| 1 | V2 MoE 3D repack — fill `deferred_tensors` | claude-internal | `claude/fp8-repack-v2-20260520` | **P0 BLOCKING** | e2e + Tasks 7 |
| 2 | Apply autotune block default + shape helper | claude-main (done) | merged to main | P1 | — |
| 3 | P190 verdict gate port → smoke harness | codebuddy/smoke-harness | `codebuddy/fp8-smoke-harness-v2-20260520` | P1 | Task 7 |
| 4 | MoE kernel autotune sweep + V2 default | qwen/autotune | `qwen/fp8-moe-autotune-sweep-20260520` | P2 | — |
| 5 | Intermediate buffer audit (Python round-trip) | codebuddy/moe-kernel | `codebuddy/fp8-moe-kernel-v2-20260520` | P2 | — |
| 6 | Quality regression V2 — stratified + drift | qwen/quality | `qwen/fp8-quality-regression-v2-20260520` | P2 | — |
| 7 | Wave 2 orchestration (e2e smoke + verdict) | claude-main | main | P0 | Final |

All P1/P2 tasks can run in parallel. Wave 2 (Task 7) starts as soon as
Task 1 lands.

---

## 🟦 CLAUDE-INTERNAL brief — Task 1 · V2 MoE 3D repack

**Goal**: extend `scripts/spark_pack_w4a8_fp8.py full-dir` to repack the
3D MoE expert weights currently sitting in the manifest's
`deferred_tensors` map. Without this, the FP8 model dir is incomplete
and the engine falls back to BF16 for 75% of layers (MoE dominates).

**Branch**: `claude/fp8-repack-v2-20260520` (from `358bc69`).

**Read first**:
1. `scripts/spark_pack_w4a8_fp8.py` — V1 `full-dir` writes
   `deferred_tensors` for any tensor whose `original_shape` is 3D.
2. `engine/loader.py::_load_qwen36_layer_lynn_variable_w4a8_fp8` —
   shows what key layout V2 must emit:
   `mlp.experts.gate_up_proj.weight_fp8` [E, 2*intermediate, K] +
   `.weight_fp8_scale` [E, 2*intermediate]; `mlp.experts.down_proj.*`
   similarly at [E, K, intermediate].
3. `triton_kernels/spark_fp8_moe_expert_fused.py` — confirms the
   kernel reads the 3D `[E, N, K]` layout (`expert_id` selects slice).
4. Memory `project_lynn_engine_fp8_phase2_progress_20260520`.

**Storage layout reminder**:
* Input NVFP4 layout: `packed` is `[prod(original_shape[:-1]), K/2]`
  flattened (Lynn variable-NVFP4 standard) — so for MoE
  `gate_up_proj` with `original_shape = [E, 2*intermediate, K]`,
  packed shape is `[E * 2*intermediate, K/2]`.
* Scale layout: `[E * 2*intermediate]` (also flattened) or
  `[E, 2*intermediate]` depending on prior repack tooling. Read
  the manifest entry to confirm.
* Global scale: scalar per expert OR scalar overall — check
  manifest schema. Apply consistently.

**⚠️ HARD output contract** — engine FP8 path keys on these exact
shapes (`engine/full_forward.py:237-245`):

| Logical key | weight_fp8 shape | weight_fp8_scale shape | notes |
|---|---|---|---|
| `mlp.experts.gate_up_proj` | `[E, 2*intermediate, K]` | `[E, 2*intermediate]` | gate scales = `[:, :intermediate]`, up scales = `[:, intermediate:]` |
| `mlp.experts.down_proj`    | `[E, K, intermediate]`   | `[E, K]`               | per-row over output dim |

DO NOT output flat `[E*N]` scale — the engine slices the 2D scale
into gate/up halves and indexes by expert id. A flat scale silently
mismatches with no validation today (loader patched in this V2
commit to assert, but cleanest is for repack to emit 2D directly).

**Concrete deliverable**:
1. In `repack_full_dir` (or equivalent), detect 3D `original_shape`
   and dispatch to a new helper `repack_nvfp4_to_fp8_3d`:
   * Reshape flattened `packed` to `[E, N, K/2]`.
   * Unpack FP4 nibbles → reshape to `[E, N, K]`.
   * Dequantize to BF16 using per-expert per-row scale.
   * Re-quantize to FP8 E4M3 with **per-expert per-row** scale
     (matches `select_block_config(M=1, K, N)` expectations).
   * **Output FP8 weight `[E, N, K]` + scale 2D `[E, N]` (F32).**
     Per the hard contract above — not flat.
2. Move 3D entries from `deferred_tensors` to `quantized_tensors`
   in the output manifest.
3. Verification: per-expert cosine vs BF16 dequant of original
   NVFP4. Threshold = `--verify-cos-threshold` (default 0.999).
   Log per-expert pass/fail; summary at end.
4. Output a JSON summary including 3D tensor count + cos
   min/mean/max + total fp8 bytes saved vs BF16.

**Acceptance**:
- `full-dir` runs end-to-end on
  `/home/merkyor/models/Qwen3.6-35B-A3B-lynn-native-w4a16-nvfp4-from-r6000`
- `deferred_tensors` map is empty in output manifest.
- All 3D tensors pass cos > 0.999 verification.
- Output dir is ~12-13GB (FP8 is half of FP16 byte count for the
  quantized portion).
- Engine loader (`_load_qwen36_layer_lynn_variable_w4a8_fp8`) loads
  the output dir without raising.

**Branch + commit**:
- Branch from main `358bc69`: `git checkout -b claude/fp8-repack-v2-20260520`
- Single commit: `fp8: phase 2 step 1.6 — V2 MoE 3D expert weight repack`
- Push to `origin claude/fp8-repack-v2-20260520`.
- DO NOT open PR — orchestrator session will fetch-merge.

**Out of scope**:
- Don't touch the kernel or engine integration.
- Don't run the engine end-to-end here (Task 7 will).
- Don't tune scale granularity beyond per-row — V3 work if quality
  regression flags it.

---

## 🟩 CODEBUDDY brief — Task 3 · P190 verdict gate → smoke harness V2

**Goal**: harden `scripts/spark_fp8_e2e_tps_smoke.py` with the P190
verdict-gate schema so the TPS measurement cannot silently promote a
correctness-broken FP8 path.

**Branch**: `codebuddy/fp8-smoke-harness-v2-20260520` (from `358bc69`).

**Read first**:
1. `benchmarks/p190_qwen35_9b_true_fp8_resident_gate.py` — the
   reference verdict implementation. Key points:
   * Schema: `lynn-qwen35-9b-true-fp8-resident-gate-v1`
   * Two-mode run via `_run_mode` (reference + candidate)
   * Comparison via `_compare_modes` returning `exact_count`,
     `all_exact`, `total`.
   * Verdict 3-tier:
     * `all_exact` → `TRUE_FP8_RESIDENT_EXACT`
     * `exact_count == 0` → `TRUE_FP8_RESIDENT_RED`
     * else → `TRUE_FP8_RESIDENT_AMBER_DRIFT`
2. `lynn-engine/reports/qwen35_9b/P190_FP4XFP8_RESIDENT_FINDINGS_20260519.md`
   — context why exact gate matters: full 32-layer FP8 was 1.09× TPS
   but exact 0/6 RED. Fixture pass ≠ resident exact.
3. Current `scripts/spark_fp8_e2e_tps_smoke.py` — needs to evolve from
   "compare TPS" into "compare exact + TPS, RED-gate on exact==0".

**Concrete deliverable**:
1. Extend `spark_fp8_e2e_tps_smoke.py` to emit a verdict block:
   ```json
   {
     "schema": "lynn-spark-fp8-e2e-smoke-v2",
     "verdict": "FP8_E2E_EXACT" | "FP8_E2E_AMBER_DRIFT" | "FP8_E2E_RED",
     "exact_count": ...,
     "total": ...,
     "tps_lift": candidate_decode_tps / reference_decode_tps,
     "first_token_drift": [...] // per-prompt first-8-token alignment
   }
   ```
2. Add a `--gate-mode {strict, loose}` arg:
   * `strict` (default): non-zero exit if verdict is `RED` regardless
     of TPS. AMBER + TPS_lift > 1.4× → exit 0 with warning.
   * `loose`: always exit 0, print verdict only (for nightly bench).
3. Add per-prompt structural drift detector:
   * Compare first 8 decoded tokens.
   * Flag if any of `{<|im_start|>, <|im_end|>, <BOS>, role markers}`
     diverges between reference and candidate at any of those positions.
   * Report `structural_drift_prompts: [int]`.
4. Reuse `_compare_modes` / `_summarize_mode` patterns from P190; keep
   the JSON schema compatible with existing analysis tooling.

**Acceptance**:
- Harness runs on the existing W4A16 dir alone (FP8 leg fails clearly
  if FP8 dir is missing; doesn't crash).
- Verdict + structural drift fields populate correctly when run with
  two identical configs (sanity check: should be EXACT 6/6).
- JSON schema documented in script docstring.

**Branch + commit**:
- Branch from main `358bc69`: `git checkout -b codebuddy/fp8-smoke-harness-v2-20260520`
- Single commit: `fp8: phase 2 step 4 V2 — P190 verdict gate + structural drift`
- Push to `origin codebuddy/fp8-smoke-harness-v2-20260520`.

**Out of scope**:
- Don't actually run the FP8 dir today (waits for Task 1).
- Don't touch kernel / loader.

---

## 🟨 QWEN brief — Task 4 · MoE kernel autotune sweep

**Goal**: sweep `triton_kernels/spark_fp8_moe_expert_fused.py` block
configs and pick a V2 default, similar to the dense-kernel sweep that
landed `(16, 128, 32)`. Current MoE kernel is 1.82-2.10× at N=1408 /
1.25-1.89× at N=768 — we believe autotune can lift this another 20-50%.

**Branch**: `qwen/fp8-moe-autotune-sweep-20260520` (from `358bc69`).

**Read first**:
1. `scripts/spark_fp8_kernel_autotune_sweep.py` — the dense sweep
   harness. Reuse its structure: warmup 5 / run 50 / per-combo
   correctness via cosine (threshold 0.99).
2. `triton_kernels/spark_fp8_moe_expert_fused.py` — kernel signature
   + current default block config.
3. `reports/mtp/QWEN36_FP8_AUTOTUNE_SWEEP_RESULT_20260520.md` — dense
   sweep result format / report MD template to mirror.

**Sweep parameters**:
| Parameter | Values |
|---|---|
| M (tokens routed per expert) | 1, 4, 8 |
| K (hidden) | 2048, 6144 |
| N (intermediate) | 768, 1408, 2048 |
| BLOCK_M | 16, 32, 64 |
| BLOCK_K | 32, 64, 128 |
| BLOCK_N | 32, 64, 128 |
| expert_count fixed | 128 (Qwen3.6-35B-A3B canonical) |

Total combos: 18 shapes × 27 block configs = **486 combos**. ~30
minutes wall on Spark assuming dense sweep ran in similar time.

**Concrete deliverable**:
1. `scripts/spark_fp8_moe_kernel_autotune_sweep.py` — new sweep harness.
2. Update default `BLOCK_M/K/N` in
   `triton_kernels/spark_fp8_moe_expert_fused.py` to the winning config.
3. Add `select_moe_block_config(M, K, N, E)` helper analogous to
   `select_block_config` in `spark_fp8_gate_up_fused.py`.
4. Report MD: `reports/mtp/QWEN36_FP8_MOE_AUTOTUNE_SWEEP_RESULT_20260520.md`.

**Acceptance**:
- All configs pass cosine ≥ 0.99 (reject any that drift).
- Winning config beats current default by ≥ 5% on at least 60% of shapes.
- Report MD mirrors dense-kernel result format.

**Branch + commit**:
- Branch from main `358bc69`: `git checkout -b qwen/fp8-moe-autotune-sweep-20260520`
- Single commit: `fp8: phase 2 step 4 V2 — MoE kernel autotune sweep`
- Push.

**Out of scope**:
- Don't touch the dense kernel.
- Don't change kernel correctness path / quantization scheme.

---

## 🟧 CODEBUDDY brief — Task 5 · Intermediate buffer audit

**Goal**: P190 finding #2 said native-owned intermediate buffer beats
Python/Torch round-trip between fused gate/up and down_proj. Audit
both `_dense_ffn_forward` and `_moe_forward` FP8 branches in
`engine/full_forward.py` to see if the intermediate BF16 tensor is
making an unnecessary Python round-trip.

**Branch**: `codebuddy/fp8-moe-kernel-v2-20260520` (from `358bc69`).

**Read first**:
1. `engine/full_forward.py:265-300` — `_moe_forward` FP8 branch.
   After kernel call, `shared_inter` flows through:
   * `.abs().amax(dim=-1, keepdim=True)` — Python tensor op
   * `.to(torch.float32) / inter_scale_s).to(torch.float8_e4m3fn)` —
     more Python tensor ops
   * Then `torch._scaled_mm(...)` for down.
   That's 3 Python/CUDA round-trips between two CUDA kernels.
2. `engine/full_forward.py:420-440` — `_dense_ffn_forward` FP8 branch.
   Same pattern.
3. `triton_kernels/spark_fp8_gate_up_fused.py` — kernel output is BF16.

**Concrete deliverable**:
1. Profile (Spark) the current FP8 dense + MoE FFN path: measure
   * Total time per FFN call
   * Time inside Triton kernel
   * Time spent in inter-kernel Python ops
   Output: `reports/mtp/QWEN36_FP8_FFN_PROFILE_20260520.md` with
   breakdown.
2. If round-trip overhead is > 10% of FFN time, fuse the
   activation-rescale + FP8 cast into the gate/up kernel epilogue
   (kernel writes both BF16 intermediate + FP8 intermediate + scale).
   New kernel signature: emit FP8 intermediate + FP8 scale as a
   second output buffer.
3. Update `_dense_ffn_forward` + `_moe_forward` to use FP8 output
   directly without the rescale step.

**Acceptance**:
- Profile report exists with concrete µs/op breakdown.
- If fix is needed: cosine ≥ 0.999 vs current path (per-token via
  spark_fp8_kernel_verify.py with the new dual-output kernel).
- Net FFN time decreases by at least 10% on Spark (M=1).

**Branch + commit**:
- Branch from main `358bc69`: `git checkout -b codebuddy/fp8-moe-kernel-v2-20260520`
- Single commit: `fp8: phase 2 step 5 — FFN intermediate buffer audit + fix`
- Push.

**Out of scope**:
- Don't touch the MoE kernel for the V2 autotune sweep (that's Task 4).
- Don't change the dense kernel's block config (done in Task 2).

---

## 🟪 QWEN brief — Task 6 · Quality regression V2

**Goal**: extend `scripts/spark_w4a8_vs_w4a16_quality_regression.py`
with diagnostics that pinpoint where quality drift starts if the FP8
path regresses.

**Branch**: `qwen/fp8-quality-regression-v2-20260520` (from `358bc69`).

**Read first**:
1. Current `scripts/spark_w4a8_vs_w4a16_quality_regression.py`.
2. `reports/mtp/QWEN36_FP8_QUALITY_REGRESSION_PLAN_20260520.md` —
   accepted gates.

**Concrete deliverable**:
1. Add `--diagnose` flag that, on top of the standard MMLU/GPQA run,
   also emits:
   * Per-subject MMLU accuracy delta (e.g., `accuracy_delta_by_subject`).
   * First-N-token logit divergence on a fixed 20-prompt probe set
     (saves both models' top-5 logits at positions 0-7 for offline
     diff). Output: `reports/mtp/QWEN36_FP8_LOGIT_DRIFT_*.json`.
   * GPQA-Diamond `parse_fail_rate` for each model independently
     (some quality issues manifest as broken formatting not wrong
     answers).
2. Acceptance gates unchanged: MMLU Δ ≤ 1pp / GPQA-excl-pf Δ ≤ 2pp.

**Acceptance**:
- Script runs cleanly on existing W4A16 dir alone (FP8 leg falls
  back to skip if dir missing).
- Diagnose mode produces the three new outputs.
- Subject-level deltas correctly populate for all 57 MMLU subjects.

**Branch + commit**:
- Branch from main `358bc69`: `git checkout -b qwen/fp8-quality-regression-v2-20260520`
- Single commit: `fp8: phase 2 step 5 V2 — quality regression diagnostics`
- Push.

**Out of scope**:
- Don't add new accuracy gates beyond MMLU/GPQA.
- Don't touch eval prompt sets (use whatever is already canonical).

---

## Wave 2 plan (Task 7 — orchestrator)

After Task 1 + at least 2 of Tasks 3-6 land:

1. Fetch + merge all CLI branches into main in order:
   `claude/fp8-repack-v2` → `codebuddy/fp8-smoke-harness-v2` →
   `qwen/fp8-moe-autotune-sweep` → `codebuddy/fp8-moe-kernel-v2` →
   `qwen/fp8-quality-regression-v2`.
2. Run repack V2 on Spark:
   ```
   python scripts/spark_pack_w4a8_fp8.py full-dir \
     --input  /home/merkyor/models/Qwen3.6-35B-A3B-lynn-native-w4a16-nvfp4-from-r6000 \
     --output /home/merkyor/models/Qwen3.6-35B-A3B-lynn-native-w4a8-fp8 \
     --scale-granularity per_row \
     --verify-cos-threshold 0.999
   ```
3. Run e2e TPS smoke with strict verdict gate:
   ```
   python scripts/spark_fp8_e2e_tps_smoke.py \
     --w4a16-model-dir /home/merkyor/models/Qwen3.6-35B-A3B-lynn-native-w4a16-nvfp4-from-r6000 \
     --w4a8-model-dir  /home/merkyor/models/Qwen3.6-35B-A3B-lynn-native-w4a8-fp8 \
     --gate-mode strict \
     --out reports/mtp/spark_fp8_e2e_$(date +%s).json
   ```
4. Run quality regression V2:
   ```
   python scripts/spark_w4a8_vs_w4a16_quality_regression.py \
     --model-a /home/merkyor/models/Qwen3.6-35B-A3B-lynn-native-w4a16-nvfp4-from-r6000 \
     --model-b /home/merkyor/models/Qwen3.6-35B-A3B-lynn-native-w4a8-fp8 \
     --diagnose \
     --out reports/mtp/spark_fp8_quality_$(date +%s).json
   ```

**Decision tree** based on results:

| TPS | Verdict | Quality | Action |
|---|---|---|---|
| ≥ 60 | EXACT | OK | ✅ Lock & release |
| 55-60 | EXACT | OK | ✅ Ship + MTP K=1 overlay as next-iteration multiplier |
| < 55 | EXACT | OK | Investigate: intermediate buffer / launch overhead / MoE kernel autotune V2 |
| Any | AMBER_DRIFT | OK | Trace prompt subset where drift occurred; per-shape kernel investigation |
| Any | RED | — | Halt promote; root-cause exact mismatch (probably scale granularity or epsilon) |
| Any | EXACT | FAIL | Switch scale granularity per_row → per_block_32 → per_block_64 on the offending layer types |

**MTP K=1 overlay (fallback path if FP8 alone < 60)**:
* `spec_k1` sequential is already 6/6 exact @ 26.4 TPS on baseline W4A16.
* On W4A8 FP8 base, effective TPS = base_TPS × (1 + accept_rate).
* If base = 50 TPS + 30% accept rate → 65 TPS.
* Wire path: `LYNN_MTP_SPECULATIVE=1 LYNN_MTP_K=1` env knob, no
  kernel changes needed (sidecar already trained, see
  `project_mtp_held_w4a8_first_20260516` memory).

---

## Already done (this orchestrator session)

* **Task 2 — Autotune block default + shape-aware helper (commit
  pending push)**:
  * `triton_kernels/spark_fp8_gate_up_fused.py`: default block config
    `(16, 64, 128)` → `(16, 128, 32)` (universal winner per sweep).
  * Added `select_block_config(M, K, N)` helper with per-shape
    overrides for Lynn 35B-A3B hot shapes.
  * Added `auto_block: bool = False` flag to `fp8_gate_up_silu_fused`
    that opts into shape-aware dispatch.
  * `engine/full_forward.py`: both FP8 caller sites now pass
    `auto_block=True` (shared expert + dense FFN).

Forecast lift: M=1 K=2048 N=6144 default 3× → auto_block 6.00× = 2×
local. Net e2e impact depends on layer shape mix.

---

## Cross-reference

* Phase 2 V1 handoff: `reports/mtp/QWEN36_FP8_PHASE2_SESSION_HANDOFF_FINAL_20260520.md`
* V1 dispatch brief: `reports/mtp/_DISPATCH_FP8_PARALLEL_CLI_BRIEFS_20260520.md`
* P190 reference: `lynn-engine/reports/qwen35_9b/P190_FP4XFP8_RESIDENT_FINDINGS_20260519.md`
* Strategy memory: `reference_spark_fp8_w4a8_design_strategy_20260519`
* T=1 kernel contract: `project_lynn_engine_t1_only_kernel_contract_20260519`
