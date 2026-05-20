# FP8 Phase 2 — 3-CLI parallel dispatch briefs · 2026-05-20

Copy each brief into the corresponding CLI as the initial prompt.
All three work against `MerkyorLynn/lynn-engine` repo, `main` branch
HEAD = `7401a32` (just pushed). Each works on a SEPARATE feature
branch to avoid file collisions.

I (this Claude session) continue on `claude/mtp-k2-strict-diag-20260520`
with kernel hardening (autotune, MoE expert variant) +
`engine/resident_runner.py` integration design.

---

## 🟦 CLAUDE CODE brief — Phase 2 step 1.5: full Lynn-native model dir → FP8 repack V1

**Goal**: extend the v0 offline NVFP4 → FP8 repack tool
(`scripts/spark_pack_w4a8_fp8.py`) into a manifest-driven full Lynn-
native model dir repacker that emits a parallel FP8 model dir ready
for the resident_runner FP8 path.

**Branch**: `claude/fp8-repack-v1-20260520`

**Read first** (in this order):
1. `scripts/spark_pack_w4a8_fp8.py` (commits `e370cbc` + `ca77375`):
   the v0 single-weight function `repack_nvfp4_to_fp8` and the
   `RepackResult` dataclass. The math is correct (self-test PASS
   cos > 0.999 across 4 shapes × 2 granularities).
2. `engine/loader.py:119-237` (`_dequantize_lynn_variable_nvfp4` +
   `_load_qwen36_layer_lynn_variable_nvfp4`): the manifest-driven
   loader that reads `lynn_quant_manifest.json` and maps each
   `quantized_tensors` entry to `(packed_key, scale_key, global_scale_key,
   original_shape)`.
3. `reports/mtp/QWEN36_MTP_PHASE2_SESSION_HANDOFF_20260520.md`
   (Phase 2 state) and `reference_spark_fp8_w4a8_design_strategy_20260519`
   memory for the strategy framing.
4. A real Lynn-native model dir's manifest on Spark:
   `/home/merkyor/models/Qwen3.6-35B-A3B-lynn-native-w4a16-nvfp4-from-r6000/lynn_quant_manifest.json`

**Concrete deliverable**:
1. Extend `scripts/spark_pack_w4a8_fp8.py` with a new CLI subcommand
   `full-dir`:
   ```bash
   python scripts/spark_pack_w4a8_fp8.py full-dir \
       --input  /home/merkyor/models/Qwen3.6-35B-A3B-lynn-native-w4a16-nvfp4-from-r6000 \
       --output /home/merkyor/models/Qwen3.6-35B-A3B-lynn-native-w4a8-fp8 \
       --scale-granularity per_row \
       --verify-cos-threshold 0.999
   ```
2. Behaviour:
   - Reads `lynn_quant_manifest.json` + `model.safetensors.index.json`.
   - For each `quantized_tensors` entry: load `(packed, scale, global_scale)`,
     call existing `repack_nvfp4_to_fp8`, get `RepackResult`.
   - Write **new** safetensors files under `--output` with keys:
     - `<base>.weight_fp8` (was `<base>.weight.packed`)
     - `<base>.weight_fp8_scale` (was `<base>.weight.scale` + `.global_scale`)
     - **Preserve `original_shape`** if 3D (MoE experts are flattened
       in storage but logically 3D — the FP8 output should reshape
       back to original).
   - Copy all `kept_tensors` (non-quantized, e.g. layernorm weights,
     embed_tokens, lm_head if not quantized) as-is.
   - Write new `lynn_quant_manifest.json` with schema `lynn-variable-w4a8-fp8-v1`
     mapping logical keys → `(weight_fp8_key, weight_fp8_scale_key,
     original_shape, fp8_scale_granularity)`.
   - Write new `model.safetensors.index.json` reflecting new key map.
3. Streaming verification: per-tensor cosine vs BF16 dequant of the
   ORIGINAL NVFP4. Threshold = `--verify-cos-threshold`. Per-tensor
   pass/fail. Summary report at end.
4. Output a JSON summary: tensor count, total fp8 bytes, total scale
   bytes, per-tensor cosine min/mean/max, total wall time.

**Acceptance**:
- Tool runs end-to-end on the 35B Lynn-native dir (~23 GB input,
  ~12-13 GB FP8 output expected since FP8 is half of FP16).
- All quantized tensors pass cos > 0.999 verification.
- Output dir is self-describing via the new manifest (V1-compatible
  loader can read it without seeing the old NVFP4 dir).
- Doesn't modify the input dir.
- Skip MoE expert weights with `original_shape` 3D where FP8 layout
  is uncertain — log and continue; produce explicit "deferred" list
  in the summary.

**Branch + commit format**:
- Branch from `main` HEAD `7401a32`: `git checkout -b claude/fp8-repack-v1-20260520`
- Single commit titled: `fp8: phase 2 step 1.5 — full Lynn-native dir → FP8 repack V1`
- Push to `origin claude/fp8-repack-v1-20260520`, don't open PR yet.

**Out of scope** (don't touch):
- The Triton kernel (`triton_kernels/spark_fp8_gate_up_fused.py`)
- `engine/resident_runner.py` integration
- Anything under `engine/mtp_*` (MTP Phase 1 is closed)

---

## 🟩 CODEBUDDY brief — Phase 2 step 4: end-to-end FP8 vs W4A16 TPS smoke harness

**Goal**: build the smoke harness that will measure the actual end-to-
end TPS lift once the Claude Code agent's repack V1 produces a FP8
model dir and the kernel/integration is wired in.

**Branch**: `codebuddy/fp8-tps-smoke-harness-20260520`

**Read first**:
1. `scripts/spark_mtp_speculative_smoke.py` — the canonical Lynn smoke
   runner (M19/M20/M22/M24 all built on this). Note BASE_ENV, config
   list shape, summary JSON schema, gates dict.
2. `reports/mtp/QWEN36_MTP_PHASE2_SESSION_HANDOFF_20260520.md`
   — the current TPS state table (Lynn 38.96 baseline, SGLang FP8+MTP
   60-70 target).
3. `reference_spark_fp8_w4a8_design_strategy_20260519` memory.

**Concrete deliverable**:
`scripts/spark_fp8_e2e_tps_smoke.py` — a new smoke runner that runs
TWO model configs side by side on the same canonical 6-prompt set:

```
configs:
  - label: "w4a16_baseline_graph"
    model_dir: /home/merkyor/models/Qwen3.6-35B-A3B-lynn-native-w4a16-nvfp4-from-r6000
    env:
      LYNN_LINEAR_BLOCK_GRAPH=1
      LYNN_LINEAR_BLOCK_GRAPH_REUSE=1
      LYNN_MTP_SPECULATIVE=0
      (plus existing Spark Config D env)
  - label: "w4a8_fp8_baseline_graph"
    model_dir: /home/merkyor/models/Qwen3.6-35B-A3B-lynn-native-w4a8-fp8  # produced by repack V1
    env:
      LYNN_W4A8_FP8_PATH=1     # new env knob the integration step will gate on
      LYNN_LINEAR_BLOCK_GRAPH=1
      LYNN_LINEAR_BLOCK_GRAPH_REUSE=1
      LYNN_MTP_SPECULATIVE=0
      (plus existing Spark Config D env minus LYNN_PACKED_DECODE if not applicable)
  - label: "w4a8_fp8_baseline_eager"
    model_dir: <same FP8 dir>
    env:
      LYNN_W4A8_FP8_PATH=1
      LYNN_LINEAR_BLOCK_GRAPH=0
      LYNN_LINEAR_BLOCK_GRAPH_REUSE=0
      LYNN_MTP_SPECULATIVE=0
```

For each config:
- Run on canonical 6 prompts at `--max-new 64` and `--max-new 128`.
- Capture: mean decode TPS, exact-match (vs the w4a16 baseline run),
  prefix-match length, per-prompt timing.
- Output JSON like spark_mtp_speculative_smoke does, plus a derived
  "TPS lift" table.

**Acceptance**:
- Harness is wired up and runs cleanly on the existing w4a16 model
  (FP8 config will fail until the repack + kernel integration land —
  harness should handle that with a clear error rather than crash).
- JSON schema documented in the file docstring.
- Reports go to `reports/mtp/spark_fp8_e2e_*_$(date).json` and a
  small companion `.md`.

**Branch + commit format**:
- Branch from `main` HEAD `7401a32`: `git checkout -b codebuddy/fp8-tps-smoke-harness-20260520`
- Single commit: `fp8: phase 2 step 4 — FP8 vs W4A16 end-to-end TPS smoke harness`
- Push to `origin codebuddy/fp8-tps-smoke-harness-20260520`.

**Out of scope**:
- Don't write the kernel.
- Don't modify resident_runner.
- Don't try to actually run the FP8 config tonight — just make the
  harness ready for when the FP8 path lands.

---

## 🟧 QWEN brief — Phase 2 step 5: W4A8 vs W4A16 quality regression eval scaffold

**Goal**: scaffold a quality regression eval that compares Lynn-native
W4A16 NVFP4 (current ship) against the upcoming W4A8 FP8 path on a
quick MMLU-100 + GPQA-50 subset. The actual W4A8 dir doesn't exist
yet; the scaffold should be runnable against the W4A16 dir alone today
and add the W4A8 leg once it's available.

**Branch**: `qwen/fp8-quality-regression-20260520`

**Read first**:
1. `reports/mtp/QWEN36_MTP_PHASE2_SESSION_HANDOFF_20260520.md`
   (Phase 2 state).
2. Existing eval scripts under `scripts/spark_*_mmlu*.py` or
   `benchmarks/*_mmlu*` (model card recipes for MMLU eval —
   pick whichever Lynn-engine already has).
3. `reference_qwen36_35b_release_numbers_20260519` memory (baseline
   MMLU 84.40 / GPQA 49.49 for current W4A16 NVFP4).
4. Memory `reference_spark_fp8_w4a8_design_strategy_20260519` —
   note section "量化损失数" says PoC showed W4A8 fake-quant FFN-only
   MMLU 75.80 / GPQA 43.94 vs W4A16 76.00 / 42.93 = essentially flat.
   The real W4A8 native FP8 should land in the same range.

**Concrete deliverable**:
`scripts/spark_w4a8_vs_w4a16_quality_regression.py` — a script that:
- Takes two model dirs as positional args
  (`--model-a` and `--model-b`, default labels `w4a16` and `w4a8`).
- Runs both through a fixed MMLU-100 + GPQA-50 subset (reuse whatever
  prompt loader the existing benchmarks use).
- Generates with deterministic decode (greedy, max_new = 32-256
  depending on the eval).
- Emits per-question + summary JSON: per-subject MMLU accuracy,
  GPQA-Diamond accuracy excluding parse_fail, mean prefix
  agreement of A vs B sequences.
- Verdict: A vs B accuracy delta and whether W4A8 stays within
  1pp of W4A16 (= acceptable quality regression).

Plus `reports/mtp/QWEN36_FP8_QUALITY_REGRESSION_PLAN_20260520.md`
documenting:
- Eval recipe
- Acceptance gates (W4A8 MMLU within 1pp of W4A16, GPQA within 2pp)
- How to run once W4A8 dir is ready

**Acceptance**:
- Script runs successfully against the EXISTING W4A16 dir alone
  (`--model-a /home/merkyor/models/Qwen3.6-35B-A3B-lynn-native-w4a16-nvfp4-from-r6000`)
  and outputs the W4A16 baseline numbers in the JSON. The W4A8 leg
  can be left as a "not yet" placeholder.
- MD plan is self-contained.

**Branch + commit format**:
- Branch from `main` HEAD `7401a32`: `git checkout -b qwen/fp8-quality-regression-20260520`
- Single commit: `fp8: phase 2 step 5 — W4A8 vs W4A16 quality regression eval scaffold`
- Push to `origin qwen/fp8-quality-regression-20260520`.

**Out of scope**:
- Don't run the W4A8 model today (doesn't exist yet).
- Don't touch any other reports / files outside the two listed.

---

## Coordination & merge plan

After all three CLIs push their branches:
1. I (this Claude session) review each branch, do conflict-free merges
   into `main` in order: repack V1 → smoke harness → quality scaffold.
2. Integration commit on `claude/mtp-k2-strict-diag-20260520` ties
   everything together with the kernel + resident_runner FP8 path
   detection.
3. End-to-end smoke runs against the new FP8 model dir.

If any of the three CLIs hits a blocker (model file missing, schema
mismatch, etc.), they should write the blocker to a TODO comment in
their commit and push anyway — better to surface than silently fail.
