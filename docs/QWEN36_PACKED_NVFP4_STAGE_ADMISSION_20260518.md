# P142 · Packed NVFP4 Stage Admission Gate

## Overview

`p142_packed_nvfp4_stage_admission_gate.py` is a read-only admission gate
that prevents fixture-stage AMBER from leaking into the resident/default
promotion path.  It consumes three upstream reports and produces a single
admission verdict.

**No kernel execution, no resident, no model loading — pure report aggregation.**

## Inputs

| Report | Key fields | Purpose |
|--------|-----------|---------|
| `p141_v2_report.json` | `verdict`, `max_max_abs`, `avg_latency_ms` | Fixture-stage AMBER check |
| `p140_packed_nvfp4_probe_report.json` | `verdict`, `max_max_abs`, `avg_latency_ms` | Packed kernel accuracy/latency |
| `native_moe_candidate_summary.json` | `summary.best_verdict` | Overall candidate landscape |

## Admission Verdicts

| Verdict | Badge | Condition | Meaning |
|---------|-------|-----------|---------|
| **DEFAULT_BLOCKED** | 🟡 | p141=AMBER_STAGE OR packed probe=CLOSED | Default/resident promote forbidden; P37 exploratory may still be allowed when the newest p141 stage probe is within loose P37 bounds |
| **P37_ALLOWED** | 🔵 | stage or packed probe is within loose P37 bounds and no hard limit is exceeded | P37 exploratory work permitted, still no default promote |
| **CLOSED** | 🔴 | packed max_abs>0.01 OR latency>0.15ms | Block all further work on this path |

### Decision Matrix

| p141 verdict | packed verdict | → Admission |
|-------------|---------------|-------------|
| AMBER_STAGE within P37 bounds | any | 🟡 DEFAULT_BLOCKED + P37_ALLOWED |
| AMBER_STAGE outside P37 bounds | any | 🟡 DEFAULT_BLOCKED |
| GREEN | CLOSED | 🟡 DEFAULT_BLOCKED |
| GREEN | GREEN/AMBER (within P37 bounds) | 🔵 P37_ALLOWED |
| any | hard limit exceeded | 🔴 CLOSED |

## Thresholds

| Tier | max_abs | latency (ms) |
|------|---------|-------------|
| P37 exploratory | ≤ 0.005 | ≤ 0.10 |
| Hard CLOSED | > 0.01 | > 0.15 |

## Usage

### On R6000

```bash
bash scripts/r6000_qwen36_packed_nvfp4_stage_admission.sh
```

### Direct

```bash
python benchmarks/p142_packed_nvfp4_stage_admission_gate.py \
    --report-dir reports/qwen36_35b \
    --out reports/qwen36_35b/p142_packed_nvfp4_stage_admission.json \
    --md-out reports/qwen36_35b/p142_packed_nvfp4_stage_admission.md
```

### Env vars

| Variable | Default | Description |
|---|---|---|
| `P141_REPORT` | _(auto)_ | Explicit p141 V2 report path |
| `P140_PACKED_REPORT` | _(auto)_ | Explicit p140 packed probe path |
| `CANDIDATE_SUMMARY` | _(auto)_ | Explicit candidate summary path |
| `REPORT_DIR` | `reports/qwen36_35b` | Primary report directory |
| `PY` | `/root/.../python` | Python interpreter |

## Output

### JSON (`p142_packed_nvfp4_stage_admission.json`)

Top-level fields:
- `schema` — `lynn-p142-packed-nvfp4-stage-admission-v1`
- `inputs` — echo of all input report paths and key metrics
- `admission.verdict` — `DEFAULT_BLOCKED` / `P37_ALLOWED` / `CLOSED`
- `admission.reason` — human-readable explanation
- `admission.default_promote_blocked` — boolean flag
- `admission.p37_exploratory_allowed` — boolean flag

### Markdown (`p142_packed_nvfp4_stage_admission.md`)

Human-readable report with emoji badges, suitable for PR descriptions.

## Relationship to other gates

- **P140** — single-candidate risk gate (fast native vs thresholds)
- **P141** — packed NVFP4 stage diagnostics (fixture-level AMBER detection)
- **P142** (this) — admission gate combining p141 + p140 packed probe
  to block default promotion when AMBER_STAGE is present
