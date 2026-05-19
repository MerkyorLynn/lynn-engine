# P197 · Qwen3.5-9B W4A16 vs W4A8 Token Drift Probe

**Date:** 2026-05-19
**Author:** Qwen Code (auto-generated)
**Status:** 🟡 PENDING — awaiting R6000 execution

---

## Motivation

P190 uses simple greedy exact-match to compare W4A16 and W4A8 (FP4×FP8) resident
paths. A single argmax flip caused by rounding noise looks the same as a systematic
hidden-state corruption. P197 adds per-step top-5 logit comparison to distinguish
these cases.

## Method

For each prompt, two `LynnIncrementalRunner` instances are loaded sequentially:
- **Reference:** CONVSTRICT_W4A16 (safe baseline profile)
- **Candidate default:** `true_fp4xfp8`, using
  `LYNN_DENSE_FFN_TRUE_FP8=1` plus the packed FP4xFP8 sidecar
- **Candidate optional:** `fake_w4a8`, using
  `LYNN_W4A8_FAKE_QUANT_ACTIVE=1` for quality/noise isolation only

Both generate with `top_k=5`, collecting `{ids, values}` at every step.
Per-step similarity:

```
jaccard        = |ref_ids ∩ cand_ids| / |ref_ids ∪ cand_ids|
shared_cosine  = cosine_similarity(ref_values, cand_values)  # intersection only
combined       = 0.5 * jaccard + 0.5 * shared_cosine
```

Combined is chosen over raw cosine because it captures both **what** changed (ID
set membership) and **how much** the logits shifted (magnitude on shared tokens).

## Thresholds

| Decision  | Meaning |
|-----------|---------|
| STRICT    | 0 drift steps — candidate is greedy-identical at top-5 level |
| AMBER     | combined ≥ 0.80 AND drift_ratio ≤ 0.25 — minor noise, needs human review |
| CLOSED    | combined < 0.80 OR drift_ratio > 0.25 — candidate is not safe for resident |

## P197 JSON Schema

```json
{
  "schema": "lynn-qwen35-9b-fp4xfp8-token-drift-v1",
  "created": "ISO-8601",
  "model": "path to model dir",
  "reference_profile": "convstrict_w4a16",
  "candidate_profile": "convstrict_true_fp4xfp8_dense_ffn",
  "candidate_mode": "true_fp4xfp8",
  "sidecar_dir": "/root/autodl-tmp/reports/qwen35_9b/p192_dense_fp4x_fp8_sidecar",
  "max_seq_len": 4096,
  "n_prompts": 5,
  "elapsed_s": 123.4,
  "steps": [ ... ],
  "first_drift_step": null | int,
  "exact_match_count": 40,
  "top5_jaccard_mean": 0.95,
  "shared_cosine_mean": 0.98,
  "combined_score": 0.96,
  "decision": "STRICT|AMBER|CLOSED"
}
```

## How to Run

### On R6000 (requires GPU)

```bash
# Copy lynn-engine to R6000, then:
cd /root/autodl-tmp/lynn-engine
bash scripts/r6000_qwen35_9b_fp4xfp8_token_drift_probe.sh

# Custom parameters:
LIMIT=70 MAX_NEW=64 bash scripts/r6000_qwen35_9b_fp4xfp8_token_drift_probe.sh

# Fake-quant comparison only, useful when the native FP4 extension is absent:
CANDIDATE_PROFILE=fake_w4a8 bash scripts/r6000_qwen35_9b_fp4xfp8_token_drift_probe.sh
```

### Expected Output

```json
{
  "decision": "STRICT",
  "exact_match_count": 40,
  "total_steps": 40,
  "first_drift_step": null,
  "drift_ratio": 0.0,
  "combined_score": 1.0,
  "top5_jaccard_mean": 1.0,
  "shared_cosine_mean": 1.0
}
```

## Results

**TODO: fill after R6000 run.**

| Metric | Value |
|--------|-------|
| decision | — |
| exact_match_count | — / 40 |
| first_drift_step | — |
| combined_score | — |
| top5_jaccard_mean | — |
| shared_cosine_mean | — |

### Per-prompt drill-down

**TODO: fill after R6000 run.**

| Prompt | Drift steps | first_drift | combined |
|--------|-------------|-------------|----------|
| (fill from JSON) | | | |

---

## Integration with P190

P190 can optionally consume P197 output via `--p197-report`:

```bash
# Run P197 first:
bash scripts/r6000_qwen35_9b_fp4xfp8_token_drift_probe.sh

# Then pass report to P190:
P197_REPORT=/root/autodl-tmp/reports/qwen35_9b/p197_fp4xfp8_token_drift_*.json \
  bash scripts/r6000_qwen35_9b_true_fp8_resident_gate.sh
```

P197 CLOSED/AMBER overrides P190's simple exact-match verdict.

---

## Relationship to Other Work

| Item | Relation |
|------|----------|
| P190 | P197 feeds into P190 as a severity classifier |
| P184 | CONVSTRICT_ENV baseline is from P184 |
| P148 | _run_mode / _compare_modes utility is from P148 |
| P193 | Admission gate — P197 result can be referenced in P193's final verdict |
