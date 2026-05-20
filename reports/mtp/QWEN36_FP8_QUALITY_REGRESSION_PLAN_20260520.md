# Qwen3.6-35B FP8 Quality Regression Plan

**Date**: 2026-05-20
**Phase**: FP8 Phase 2, Step 5
**Author**: Qwen CLI (automated scaffold)

---

## 1. Objective

Verify that the upcoming Lynn-native W4A8 FP8 model artifact (produced by
Phase 2 step 1.5 repack V1) does not regress quality beyond acceptable
thresholds compared to the current W4A16 NVFP4 production model on quick
MMLU and GPQA subsets.

## 2. Background

### 2.1 Current production baseline (W4A16 NVFP4)

| Benchmark | Score | Source |
|-----------|-------|--------|
| MMLU (full) | 84.40 | Release numbers 2026-05-19 |
| GPQA-Diamond (full) | 49.49 | Release numbers 2026-05-19 |

### 2.2 PoC fake-quant reference (W4A8, FFN-only)

| Benchmark | W4A8 fake-quant | W4A16 | Delta |
|-----------|-----------------|-------|-------|
| MMLU | 75.80 | 76.00 | -0.20 |
| GPQA | 43.94 | 42.93 | +1.01 |

The PoC showed W4A8 FFN-only quantization produces essentially flat quality.
The real W4A8 native FP8 (full model, E4M3 with per-row scales) should land
in the same range or better.

### 2.3 Rationale for quick subsets

Running full MMLU (14k questions) and full GPQA-Diamond (198 questions) on
two models via `LynnIncrementalRunner` (sequential generation) takes hours.
A 100-question MMLU subset + 50-question GPQA subset provides sufficient
statistical signal for a ±1pp regression gate while completing in ~30-60 min
per model on Spark.

## 3. Evaluation Recipe

### 3.1 Datasets

| Eval | Subset size | Source | Prompt format |
|------|-------------|--------|---------------|
| MMLU-100 | 100 (deterministic sample, seed=20260519) | MMLU CSV test split, all subjects | 5-shot, subject-specific dev examples |
| GPQA-Diamond-50 | 50 (deterministic sample, seed=20260519) | `gpqa_diamond.csv` | 0-shot, shuffled choices with fixed seed per question |

### 3.2 Generation parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Decode strategy | Greedy (temperature=0) | Deterministic; eliminates sampling variance |
| `max_new_tokens` | 16 (MMLU), 16 (GPQA) | Single-letter answer expected; 16 provides safety margin |
| Batch size | 1 (sequential) | Matches `LynnIncrementalRunner.generate()` interface |

### 3.3 Answer extraction

Identical regex chain used by `openai_mmlu_500_5shot_eval.py` and
`openai_gpqa_diamond_eval.py`:
1. First-char check (A/B/C/D)
2. `^(?:answer\s*[:：]?\s*)?([ABCD])\b`
3. `\banswer\s*(?:is|:|：)?\s*([ABCD])\b`
4. `\(([ABCD])\)`
5. `\b([ABCD])\b`

Parse failures (no match) are counted but excluded from accuracy numerator.

### 3.4 Environment

Both models run through `LynnIncrementalRunner` with the Spark Config D
environment baseline:
```
LYNN_MOE_IMPL=packed_nvfp4
LYNN_LINEAR_ATTN_RECURRENT_INPLACE=1
LYNN_NATIVE_FP4_LM_HEAD=1
LYNN_PACKED_DECODE_BACKEND=native_fast_2d
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

## 4. Acceptance Gates

| Gate | Threshold | Severity |
|------|-----------|----------|
| W4A8 MMLU-100 accuracy within 1pp of W4A16 | \|Δ\| ≤ 0.01 | **HARD** — blocks promotion |
| W4A8 GPQA-Diamond-50 accuracy within 2pp of W4A16 | \|Δ\| ≤ 0.02 | **HARD** — blocks promotion |
| W4A8 MMLU-100 parse_fail ≤ W4A16 parse_fail | ≤ baseline | **SOFT** — investigate if violated |
| W4A8 GPQA-50 parse_fail ≤ 5% of N | ≤ 0.05 × N | **SOFT** — investigate if violated |

**PASS** requires both HARD gates to pass. SOFT gates flag issues for
investigation but do not block.

### 4.1 Rationale for thresholds

- **MMLU 1pp**: With n=100, the 95% CI for accuracy is approximately ±10pp.
  A 1pp threshold is conservative — if the true regression is >1pp, we'll
  see it consistently across multiple runs. If it's <0.5pp, the subset is
  too noisy to detect reliably, but that's acceptable because <0.5pp is
  within measurement noise for full MMLU too.
- **GPQA 2pp**: With n=50 and baseline accuracy ~50%, variance is higher.
  The 2pp threshold provides the same practical sensitivity as MMLU's 1pp.

## 5. How to Run

### 5.1 Prerequisites

1. **Spark access**: `ssh dgx-via-n5`
2. **Eval data on Spark**:
   - MMLU CSV at `/tmp/datasets/mmlu/` (or `--mmlu-data-dir <path>`)
   - GPQA Diamond CSV at `/tmp/datasets/gpqa/gpqa_diamond.csv` (or `--gpqa-csv <path>`)
3. **Lynn-engine worktree** on branch `qwen/fp8-quality-regression-20260520`

### 5.2 Baseline only (W4A8 not yet available)

```bash
ssh dgx-via-n5
cd /path/to/lynn-engine

python scripts/spark_w4a8_vs_w4a16_quality_regression.py \
    --model-a /home/merkyor/models/Qwen3.6-35B-A3B-lynn-native-w4a16-nvfp4-from-r6000 \
    --out-dir reports/mtp/fp8_quality_regression_$(date +%Y%m%d_%H%M%S)
```

This runs the W4A16 baseline leg only and writes the report JSON with
model-A numbers. Model-B is recorded as `"status": "placeholder"`.

### 5.3 Full comparison (W4A8 repack V1 available)

```bash
python scripts/spark_w4a8_vs_w4a16_quality_regression.py \
    --model-a /home/merkyor/models/Qwen3.6-35B-A3B-lynn-native-w4a16-nvfp4-from-r6000 \
    --model-b /home/merkyor/models/Qwen3.6-35B-A3B-lynn-native-w4a8-fp8 \
    --out-dir reports/mtp/fp8_quality_regression_$(date +%Y%m%d_%H%M%S)
```

### 5.4 Custom dataset paths

```bash
python scripts/spark_w4a8_vs_w4a16_quality_regression.py \
    --model-a /path/to/w4a16 \
    --model-b /path/to/w4a8 \
    --mmlu-data-dir /data/mmlu \
    --gpqa-csv /data/gpqa/gpqa_diamond.csv \
    --out-dir reports/mtp/fp8_quality_regression_custom
```

## 6. Output Schema

The report JSON (`quality_regression_report.json`) uses schema
`lynn-fp8-quality-regression-v1`:

```json
{
  "schema_version": "lynn-fp8-quality-regression-v1",
  "timestamp": "2026-05-20T...",
  "models": {
    "a": { "label": "w4a16", "model_dir": "..." },
    "b": { "label": "w4a8", "model_dir": "...", "status": "ok|error|placeholder" }
  },
  "subsets": {
    "mmlu100": {
      "model_a": { "n": 100, "correct": 84, "accuracy": 0.84, "parse_fail": 0, "elapsed_sec": 120.5, "by_subject": { ... } }
    },
    "gpqa50": {
      "model_a": { "n": 50, "correct": 25, "accuracy": 0.50, "parse_fail": 0, "elapsed_sec": 60.2 }
    }
  },
  "comparison": {
    "mmlu_accuracy_delta": -0.01,
    "gpqa_accuracy_delta": 0.02,
    "mean_prefix_agreement": null
  },
  "verdict": {
    "mmlu_within_1pp": true,
    "gpqa_within_2pp": true,
    "pass": true
  }
}
```

## 7. Next Steps

1. **Baseline capture**: Run script with model-A only to capture W4A16
   baseline numbers on the quick subsets.
2. **Repack V1**: Once Claude Code's `spark_pack_w4a8_fp8.py full-dir`
   produces the W4A8 FP8 model dir, re-run with `--model-b` for full
   comparison.
3. **Full eval**: If quick-subset gates pass, optionally run full MMLU
   (14k) + full GPQA-Diamond (198) for the final promotion report.
4. **Triton kernel integration**: Once the fused gate/up kernel + resident_runner
   integration land (Phase 2 steps 2-3), re-run to validate that the
   Triton FP8 path (vs. dequant-to-BF16 fallback) doesn't change quality.

## 8. Files

| File | Purpose |
|------|---------|
| `scripts/spark_w4a8_vs_w4a16_quality_regression.py` | Eval script (this plan's primary deliverable) |
| `reports/mtp/QWEN36_FP8_QUALITY_REGRESSION_PLAN_20260520.md` | This plan document |
| `reports/mtp/fp8_quality_regression_*/quality_regression_report.json` | Generated report (one per run) |
