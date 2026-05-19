# P193 · Qwen3.5-9B Native Packed Boundary Admission Gate

**Date:** 2026-05-19  
**Stream:** 9B-C / Qwen(MIMO)  
**Scope:** benchmarks, scripts, docs only (no engine/csrc/server edits)

## Purpose

Unify upstream quality/speed/capability gates into a single JSON decision before
any native FP4×FP8 packed boundary reaches resident serving.

## Decision Levels

| Decision | Meaning |
|---|---|
| `CLOSED_NUMERIC` | Cosine/rel_l2 outside safe envelope, or P160/P185 RED |
| `AMBER_FIXTURE_FAST` | Numeric below green but speedup compensates |
| `GREEN_FIXTURE` | All available gates pass, numerics solid |
| `PROMOTE_BLOCKED` | Numerics OK but capability/contract gates block promotion |

## Consumed Reports

| Gate | Report Prefix | Key Fields |
|---|---|---|
| P160 | `p160_` | decision, cosine_min, max_abs_max, passed/total |
| P185 | `p185_` | decision, cosine_min, rel_l2_max, speedup_vs_ref_mean |
| P189 | `p189_` | decision, cute_sm120_e4m3_e2m1_header |
| P191 | `p191_` | mma_compiled, scalar_reference_available, results[].{cosine_vs_bf16_ref, rel_l2_vs_bf16_ref, max_abs_vs_bf16_ref, mma_available} |
| P192 | `p192b_` or `p192_` | overall (GREEN/RED), results[].ok / layers{}.ok |

All reports are optional. Absent reports are skipped (not treated as failures).
P192 auto-discovers `p192b_` (contract) before `p192_` (manifest).

P191 supports both the original flat scalar-reference schema and the real-MMA
nested schema:

```json
{
  "results": [{
    "scalar_reference": {"cosine_vs_bf16_ref": 0.9999, "scalar_ms": 0.526},
    "mma_kernel": {
      "real_compute": true,
      "mma_vs_scalar_cosine": -0.015,
      "mma_ms": 0.049
    }
  }]
}
```

If real MMA runs but `mma_vs_scalar_cosine_min` is below the green threshold,
P193 returns `CLOSED_NUMERIC` with a fragment-layout reason.  Fast but wrong
MMA is never allowed to become AMBER.

## Default Thresholds

| Threshold | Default | Effect |
|---|---|---|
| `cosine_closed` | 0.995 | Below → CLOSED_NUMERIC |
| `cosine_green` | 0.999 | Above → eligible for GREEN |
| `rel_l2_closed` | 0.10 | Above → CLOSED_NUMERIC |
| `rel_l2_green` | 0.05 | Below → eligible for GREEN |
| `speedup_amber` | 1.00 | Speedup ≥ this + amber cosine → AMBER_FIXTURE_FAST |

## Decision Flow

```
1. Load P160/P185/P189/P191/P192 (any may be absent)
2. Numeric envelope:
   - P160 RED → CLOSED_NUMERIC
   - P160 cosine < cosine_closed → CLOSED_NUMERIC
   - P185 RED → CLOSED_NUMERIC
   - P185 cosine < cosine_closed or rel_l2 > rel_l2_closed → CLOSED_NUMERIC
   - P191 cosine < cosine_closed → CLOSED_NUMERIC
   - P191 real-MMA cosine vs scalar < cosine_green → CLOSED_NUMERIC
   - P192 RED/FAIL → CLOSED_NUMERIC
3. Speed check:
   - numeric below green BUT speedup >= speedup_amber → AMBER_FIXTURE_FAST
   - numeric below green AND no speedup → CLOSED_NUMERIC
4. Capability gates:
   - P189 CUTE_REQUIRED → PROMOTE_BLOCKED
   - P191 RED/FAIL → PROMOTE_BLOCKED
   - P192 RED/FAIL → PROMOTE_BLOCKED
5. All clear → GREEN_FIXTURE
```

## Usage

```bash
# Auto-discover latest reports
bash scripts/r6000_qwen35_9b_native_boundary_admission.sh

# Explicit report paths
python3 benchmarks/p193_qwen35_9b_native_boundary_admission.py \
    --report-dir /root/autodl-tmp/reports/qwen35_9b \
    --p160 /path/to/p160_report.json \
    --p185 /path/to/p185_report.json \
    --p189 /path/to/p189_report.json \
    --out /path/to/output.json

# Custom thresholds
python3 benchmarks/p193_qwen35_9b_native_boundary_admission.py \
    --cosine-green 0.9995 --rel-l2-green 0.03
```

## Output Schema

```json
{
  "schema": "lynn-qwen35-9b-p193-native-boundary-admission-v1",
  "created": "2026-05-19T...",
  "thresholds": { ... },
  "sources": {
    "p160": { "gate": "P160", "decision": "...", "cosine_min": ..., ... },
    "p185": { ... },
    "p189": { ... }
  },
  "decision": "GREEN_FIXTURE",
  "reasons": ["all available gates pass"]
}
```

## Decision → Action

| Decision | Next Step |
|---|---|
| `CLOSED_NUMERIC` | Investigate numeric drift; do NOT proceed to resident serving |
| `AMBER_FIXTURE_FAST` | Proceed with caution; monitor generation quality |
| `GREEN_FIXTURE` | Safe to promote to resident serving path |
| `PROMOTE_BLOCKED` | Wait for missing capability gates (P191 kernel, P192 repack) |

---

# P196 · Qwen3.5-9B W4A8 Structured Content Gate

**Date:** 2026-05-19
**Stream:** 9B-C / Qwen(MIMO)
**Scope:** benchmarks, scripts, docs only (no engine/csrc/server edits)

## Purpose

Evaluate structured content generation quality across quantization variants.
Protect W4A16 as the mandatory reference baseline; score W4A8 full and W4A8
gateup independently.  Quality drift → fall back to A16, no debate.

## Verdicts

| Verdict | Meaning |
|---|---|
| `W4A8_CONTENT_GREEN` | W4A8 pass rate ≥ W4A16 × 95% AND TPS not degraded |
| `W4A8_CONTENT_AMBER` | W4A8 pass rate ≥ W4A16 × 80% but below green |
| `RED_FALLBACK_A16` | W4A8 pass rate < W4A16 × 80% or W4A16 itself fails |

## Design Principle

**W4A16 is the non-negotiable reference.** If W4A16 fails any structured
content test, the test suite itself is broken. W4A8 variants are measured
against W4A16 as a ratio. There is no "good enough" absolute score — only
"how close to A16".

## Test Cases

| ID | Validator | Description |
|---|---|---|
| `json_object` | json_parse | Valid JSON with ≥2 keys |
| `json_array` | json_array_parse | Valid JSON array, exactly 3 elements |
| `python_function` | python_syntax | Syntactically valid Python function |
| `markdown_table` | markdown_table | Valid table, ≥3 data rows |
| `yaml_config` | yaml_parse | Valid YAML with ≥2 top-level keys |
| `csv_data` | csv_parse | Valid CSV, ≥3 data rows, ≥3 columns |
| `key_value_pairs` | key_value_lines | ≥4 key=value lines |
| `numbered_list` | numbered_list | ≥4 numbered items |
| `regex_pattern` | regex_lines | ≥2 valid regex patterns |
| `json_nested` | json_nested | Nested JSON, 3 departments |

## Decision Flow

```
1. Run all test cases with W4A16 (no fake-quant)
   → If any fail: RED_FALLBACK_A16 (test suite broken)

2. Run all test cases with W4A8 full (LYNN_W4A8_FAKE_QUANT_ACTIVE=1)
   → Compute pass_rate / w4a16_pass_rate

3. Run all test cases with W4A8 gateup
   → Compute pass_rate / w4a16_pass_rate

4. For each W4A8 variant:
   - rate_ratio < 0.80 → RED_FALLBACK_A16
   - rate_ratio < 0.95 → W4A8_CONTENT_AMBER
   - rate_ratio ≥ 0.95 AND tps_ratio ≥ 0.95 → W4A8_CONTENT_GREEN
   - rate_ratio ≥ 0.95 BUT tps_ratio < 0.95 → W4A8_CONTENT_AMBER

5. Overall verdict = worst of (w4a8_full, w4a8_gateup)
```

## Usage

```bash
# Full run (starts/stops server for each variant)
bash scripts/r6000_qwen35_9b_w4a8_structured_content_gate.sh

# Gate only (from pre-computed results)
python3 benchmarks/p196_qwen35_9b_w4a8_structured_content_gate.py \
    --results /path/to/p196_test_results.json \
    --out /path/to/report.json

# Custom thresholds
python3 benchmarks/p196_qwen35_9b_w4a8_structured_content_gate.py \
    --results /path/to/results.json \
    --green-rate 0.90 --amber-rate 0.75 \
    --out /path/to/report.json

# List built-in test cases
python3 benchmarks/p196_qwen35_9b_w4a8_structured_content_gate.py \
    --list-tests
```

## Output Schema

```json
{
  "schema": "lynn-qwen35-9b-p196-w4a8-structured-content-report-v1",
  "created": "2026-05-19T...",
  "thresholds": {
    "green_rate": 0.95,
    "amber_rate": 0.80,
    "tps_green_ratio": 0.95,
    "tps_amber_ratio": 0.80
  },
  "decision": "W4A8_CONTENT_GREEN",
  "reasons": ["w4a8_full: PASS", "w4a8_gateup: PASS"],
  "per_variant": {
    "w4a16": {
      "total": 10, "pass_count": 10, "pass_rate": 1.0,
      "failure_ids": [], "decode_tps_mean": 42.5
    },
    "w4a8_full": {
      "total": 10, "pass_count": 9, "pass_rate": 0.9,
      "failure_ids": ["yaml_config"], "decode_tps_mean": 40.1,
      "rate_vs_w4a16": 0.9, "tps_vs_w4a16": 0.944
    },
    "w4a8_gateup": {
      "total": 10, "pass_count": 10, "pass_rate": 1.0,
      "failure_ids": [], "decode_tps_mean": 41.8,
      "rate_vs_w4a16": 1.0, "tps_vs_w4a16": 0.984
    }
  },
  "per_prompt": [ ... ]
}
```

## Decision → Action

| Decision | Next Step |
|---|---|
| `W4A8_CONTENT_GREEN` | W4A8 structurally safe; proceed to numeric admission (P193) |
| `W4A8_CONTENT_AMBER` | Investigate failures; may proceed with monitoring |
| `RED_FALLBACK_A16` | **Fall back to W4A16. Do not deploy W4A8.** |

## Fallback Philosophy

> 质量漂移就回 A16，不纠结。
>
> If W4A8 produces structurally broken output — malformed JSON, invalid
> Python, corrupted tables — it does not matter how fast it is. Speed is
> worthless if the output is garbage. Fall back to W4A16 immediately.
