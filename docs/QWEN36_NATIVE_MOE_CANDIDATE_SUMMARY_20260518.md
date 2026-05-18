# P140+ · Unified Native MoE Candidate Summary

## Overview

`summarize_qwen36_native_moe_candidates.py` is a C-stream scaffolding tool
that aggregates all native MoE candidate reports into a single unified view.
It complements the P140 risk gate by showing **all** candidates side-by-side
with per-candidate verdict and next-step recommendation.

## Usage

### On R6000 (single worktree)

```bash
bash scripts/r6000_qwen36_native_moe_candidate_summary.sh
```

### With extra report directories (cross-worktree)

```bash
# Via env var (colon-separated):
EXTRA_REPORT_DIR="/path/to/other/reports/qwen36_35b" \
    bash scripts/r6000_qwen36_native_moe_candidate_summary.sh

# Via Python directly:
python scripts/summarize_qwen36_native_moe_candidates.py \
    --report-dir reports/qwen36_35b \
    --extra-report-dir /path/to/other/reports/qwen36_35b \
    --out reports/qwen36_35b/native_moe_candidate_summary.json \
    --md-out reports/qwen36_35b/native_moe_candidate_summary.md
```

### With explicit p138/p139 overrides

```bash
P138_MANIFEST=/path/to/p138_manifest.json \
P139_REPORT=/path/to/p139_contract.json \
    bash scripts/r6000_qwen36_native_moe_candidate_summary.sh
```

### Env vars

| Variable | Default | Description |
|---|---|---|
| `EXTRA_REPORT_DIR` | _(empty)_ | Colon-separated extra report dirs |
| `P138_MANIFEST` | _(auto)_ | Explicit p138 packed-slot manifest path |
| `P139_REPORT` | _(auto)_ | Explicit p139 packed-slot contract path |
| `REPORT_DIR` | `reports/qwen36_35b` | Primary report directory |
| `PY` | `/root/.../python` | Python interpreter |

Reports are searched in order: local `--report-dir` first, then each
`--extra-report-dir` in the order given.  The first match wins.  Missing
candidates are listed as `MISSING`.

## Candidate Discovery

Reports are auto-discovered by naming convention.  Missing reports are listed
as `MISSING` with `recommend_next_step: "no report found"`.

| Candidate ID | Discovery pattern |
|---|---|
| native_slot_output_owned_bf16_fast | `native_slot_output_owned_bf16_report*.json` |
| native_slot_output_owned_bf16_slotorder | `native_slot_output_owned_bf16_slotorder*report*.json` |
| native_slot_output_owned_bf16_dualref | `native_slot_output_owned_bf16_dualref*report*.json` |
| native_slot_strict_bf16 | `native_slot_strict_bf16*slotorder*report*.json` |
| native_slot_tc_bf16 | `native_slot_tc_bf16*report*.json` |
| native_slot_fused_bf16 | `native_slot_fused*report*.json` |
| native_slot_tensorcore_pretransposed_probe | `native_slot_tensorcore_pretransposed_probe*.json` |
| native_slot_packed_nvfp4_probe | `p140_packed_nvfp4_probe_report*.json` |
| packed_dequant_pretransposed_v2 | `p141_v2_report*.json` |

## Verdict Rules

| Verdict | Condition | Next Step |
|---------|-----------|-----------|
| 🟢 DEFAULT | slot_max_abs ≤ 1e-3, cosine_min ≥ 0.999999, latency ≤ 0.059ms | default promote |
| 🟡 AMBER_FAST | latency ≤ 0.055ms, slot_max_abs ≤ 0.003, P140 recommend_p37=true | P37 exploratory |
| 🟡 AMBER_FAST_PRETRANSPOSED | same as AMBER_FAST, candidate uses pretransposed weight layout | P37 exploratory |
| 🟡 AMBER_STAGE | packed-NVFP4 fixture-stage probe is fast but still approximate | build graph-safe resident ABI, then P37 exact |
| 🟢 DEFAULT_STAGE | packed-NVFP4 fixture-stage probe is exact and fast | stage-only exact; still requires P37 exact |
| 🟡 AMBER_FAST / _PRETRANSPOSED | same, but P140 does not clear | await P140 gate clearance |
| 🔵 EXACT_SLOW | slot_max_abs = 0, latency > threshold | research artifact |
| 🔴 CLOSED | thresholds exceeded | no further action |
| ⚪ MISSING | no report found | no report found |

## Output

### JSON (`native_moe_candidate_summary.json`)

Top-level fields:
- `report_dir` — primary report directory
- `extra_report_dirs` — additional directories searched
- `p136` — slot-order contract status
- `p140_gate` — risk gate verdict and recommend_p37 flag
- `p137_diagnostics` — optional diagnostic context
- `packed_slot` — p138/p139 packed-slot readiness (see below)
- `candidates[]` — per-candidate metrics and verdict (includes `report` path)
- `summary` — `best_verdict`, `has_default_candidate`, `has_amber_candidate`

### Packed-Slot section (`packed_slot`)

| Field | Type | Description |
|---|---|---|
| `packed_fixture_mb` | float? | Total packed fixture size in MiB |
| `bf16_equiv_mb` | float? | BF16-equivalent size in MiB |
| `size_reduction_pct` | float? | Compression ratio (percent) |
| `p139_verdict` | string? | `GREEN` / `RED` |
| `p139_max_abs_max` | float? | Max absolute error across all fixtures |
| `packed_ready_for_kernel` | bool | True if p139 GREEN and sizes available |
| `recommend_next_step` | string | Next action recommendation |

### Markdown (`native_moe_candidate_summary.md`)

Human-readable table with emoji badges, suitable for inclusion in PR descriptions
or status reports.

## Relationship to P140

- **P140** is a single-candidate gate (fast native vs thresholds)
- **This tool** is a multi-candidate dashboard (all candidates, side-by-side)
- Both share the same threshold definitions
- P140's `recommend_p37_exploratory` flag is consumed here to determine
  whether AMBER_FAST candidates get "P37 exploratory" or "await clearance"
