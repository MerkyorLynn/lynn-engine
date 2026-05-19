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
| P191 | `p191_` | decision, scalar_ref_cosine_min, scalar_ref_max_abs_max |
| P192 | `p192_` | decision, cosine_min, max_abs_max |

All reports are optional. Absent reports are skipped (not treated as failures).

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
