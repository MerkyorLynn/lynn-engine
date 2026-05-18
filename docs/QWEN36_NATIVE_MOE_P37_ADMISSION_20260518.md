# P143 · Resident P37 Admission Gate

## Overview

`p143_resident_p37_admission_gate.py` is a read-only admission gate that
prevents AMBER_GRAPHSAFE fixture candidates from being promoted to
resident/default.  It consumes a stage report and an optional P37
end-to-end report.

**Never outputs DEFAULT_PROMOTE — maximum admission is P25_ALLOWED.**

## Inputs

| Report | Key fields | Purpose |
|--------|-----------|---------|
| `p142_graphsafe_v31_fixture_report.json` | `verdict`, `max_max_abs`, `avg_latency_ms` | Stage-level fitness |
| P37 report (env/configurable) | `exact`/`exact_match`, `collapse`, `passed`/`total` | End-to-end validation |

## Admission Verdicts

| Verdict | Badge | Condition |
|---------|-------|-----------|
| **CLOSED_STAGE_BLOCK** | 🔴 | Stage verdict not AMBER_GRAPHSAFE/DEFAULT_STAGE, or max_abs/latency exceeded |
| **WAITING_FOR_P37_REPORT** | ⏳ | P37 report not found |
| **CLOSED_GRAPH_COLLAPSE** | 🔴 | P37 collapse/token0_collapse/repetition=true |
| **P25_ALLOWED** | 🟢 | P37 exact=true OR passed=3,total=3 |
| **CLOSED_P37_DRIFT** | 🔴 | P37 exact=false, no collapse |

> **DEFAULT_PROMOTE is never output.** Maximum admission is P25_ALLOWED.

## Thresholds

| Metric | Limit |
|--------|-------|
| Stage max_abs | ≤ 0.003 |
| Stage latency | ≤ 0.06ms |

## Usage

### On R6000

```bash
bash scripts/r6000_qwen36_moe_resident_p37_admission.sh
```

### Direct

```bash
python benchmarks/p143_resident_p37_admission_gate.py \
    --report-dir reports/qwen36_35b
```

### Env vars

| Variable | Default | Description |
|---|---|---|
| `STAGE_REPORT` | _(auto: p142_graphsafe_v31_fixture_report.json)_ | Explicit stage report path |
| `P37_REPORT` | _(auto)_ | Explicit P37 report path |
| `REPORT_DIR` | `reports/qwen36_35b` | Primary report directory |
| `PY` | `/root/.../python` | Python interpreter |

## Decision Matrix

| Stage verdict | P37 state | → Admission |
|---------------|-----------|-------------|
| not AMBER_GRAPHSAFE/DEFAULT_STAGE | — | 🔴 CLOSED_STAGE_BLOCK |
| ok, but max_abs/latency exceeded | — | 🔴 CLOSED_STAGE_BLOCK |
| ok | missing | ⏳ WAITING_FOR_P37_REPORT |
| ok | collapse=true | 🔴 CLOSED_GRAPH_COLLAPSE |
| ok | exact=true OR passed=total | 🟢 P25_ALLOWED |
| ok | exact=false, no collapse | 🔴 CLOSED_P37_DRIFT |

## Output

### JSON (`p143_resident_p37_admission.json`)

- `schema` — `lynn-p143-resident-p37-admission-v1`
- `inputs.stage_report` — echo of stage path and key metrics
- `inputs.p37_report` — echo of P37 path, exact, collapse, passed/total
- `admission.verdict` — one of the five verdicts above
- `admission.reason` — human-readable explanation
- `admission.default_promote_allowed` — always `false`

### Markdown (`p143_resident_p37_admission.md`)

Human-readable report with emoji badges.

## Relationship to other gates

- **P140** — single-candidate risk gate (fast native vs thresholds)
- **P141** — packed NVFP4 stage diagnostics (fixture-level AMBER detection)
- **P142** — packed NVFP4 stage admission (p141 + p140 packed probe)
- **P143** (this) — resident P37 admission (graphsafe stage + P37 e2e)
