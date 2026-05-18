# P140 · Native MoE AMBER Risk Gate

## Overview

P140 is a **read-only harness** that aggregates existing benchmark reports to
produce a three-tier risk verdict for the native MoE slot kernel candidate.
It does **not** execute any CUDA kernels, load models, or modify any files.

## Motivation

The fast native slot kernel (`moe_slot_output_owned_bf16`) achieves ~0.052 ms
latency — well within the 0.059 ms Triton baseline — but its numerical drift
(`slot_max_abs ≈ 0.00293`) exceeds the strict `1e-3` threshold required for
default promotion.  Meanwhile, the strict cuBLAS oracle achieves exact match
(`max_abs = 0`) but is ~8× slower at 0.467 ms.

P140 formalises the decision boundary between:
- **DEFAULT** — safe to promote to serving path
- **AMBER** — worth P37 exploratory investigation, but not default-promotable
- **CLOSED** — does not merit further investment

## Verdict Rules

| Tier    | p136 18/18 GREEN | slot_max_abs | cosine_min | unique_max_abs | latency  |
|---------|-------------------|-------------|------------|----------------|----------|
| DEFAULT | ✓                 | ≤ 1e-3      | ≥ 0.999999 | —              | ≤ 0.059  |
| AMBER   | ✓                 | ≤ 0.003     | —          | ≤ 0.002        | ≤ 0.055  |
| CLOSED  | any violation of AMBER thresholds, or missing reports             |

### AMBER annotations

When AMBER is triggered, the report explicitly annotates:
- "NO default promote — P37 exploratory only"
- The specific metrics that exceed DEFAULT thresholds
- Reference to strict cuBLAS oracle latency (too slow for serving)

## Usage

### On R6000

```bash
bash scripts/r6000_qwen36_native_moe_risk_gate.sh
```

### Manual

```bash
python benchmarks/p140_native_moe_candidate_risk_gate.py \
    --report-dir reports/qwen36_35b \
    --out reports/qwen36_35b/p140_native_moe_risk_gate.json \
    --md-out reports/qwen36_35b/p140_native_moe_risk_gate.md
```

### Override report paths

```bash
python benchmarks/p140_native_moe_candidate_risk_gate.py \
    --report-dir reports/qwen36_35b \
    --p136-report /path/to/p136_report.json \
    --candidate-report /path/to/candidate_report.json
```

## Input Reports

| Report | Discovery pattern | Required |
|--------|-------------------|----------|
| p136 slot-order contract | `p136_slot_repack_contract_slotorder_report*.json` | Yes |
| Native candidate | `native_slot_output_owned_bf16*slotorder*report*.json` | Yes |
| p137 diagnostics | `p137_moe_slot_stage_diagnostics*.json` | No (context) |
| Strict oracle | `native_slot_strict_bf16*slotorder*report*.json` | No (context) |

## Output

### JSON (`p140_native_moe_risk_gate.json`)

```json
{
  "schema": "lynn-native-moe-risk-gate-v1",
  "verdict": "AMBER",
  "recommend_p37_exploratory": true,
  "reasons": ["slot_max_abs 2.93e-03 > 1.00e-03", ...],
  "annotations": ["NO default promote — P37 exploratory only", ...],
  "inputs": { ... },
  "thresholds": { ... }
}
```

### Markdown (`p140_native_moe_risk_gate.md`)

Human-readable report with badge (🟢/🟡/🔴), metrics table, and annotations.

## Design Decisions

1. **No kernel execution** — pure report aggregation, fast and safe
2. **Auto-discovery** — finds reports by naming convention, explicit overrides available
3. **Exit code** — 0 for DEFAULT/AMBER, 1 for CLOSED (CI-friendly)
4. **Immutable** — does not modify csrc/, engine/, server/, or p135/p136 logic
