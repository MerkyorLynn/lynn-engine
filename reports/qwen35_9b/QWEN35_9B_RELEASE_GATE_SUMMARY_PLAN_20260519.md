# Qwen3.5-9B Release Gate Summary — Plan

Date: 2026-05-19

## Purpose

A GPU-free, report-only aggregator that reads existing MMLU / GPQA / TPS /
structured-content JSON and Markdown files under `reports/qwen35_9b/`, merges
them into a single unified JSON, and emits a three-tier decision:

| Decision | Meaning |
|---|---|
| `PROMOTE_READY` | All key metrics meet thresholds — candidate is promotion-ready |
| `NEEDS_MORE_DATA` | At least one required metric is missing, no hard failure yet |
| `CLOSED` | At least one metric present and below threshold — blocked |

## Thresholds (constants at top of script)

| Constant | Default | Meaning |
|---|---|---|
| `MMLU_MIN` | 0.74 | MMLU accuracy floor |
| `GPQA_MIN` | 0.41 | GPQA accuracy floor |
| `STRUCTURED_PASS` | `true` | Structured content gate must pass |
| `TPS_DECODE_512_MIN` | 155.0 | Single-request 512-token decode TPS floor |

## Output Fields

```jsonc
{
  "schema": "lynn-qwen35-9b-release-gate-summary-v1",
  "created": "ISO-8601",
  "reports_dir": "…",
  "variant": "NVFP4 | Q4_K_M | BF16",
  "quant": "nvfp4 | q4km | bf16",
  "model_size_gb": 8.3,
  "mmlu": 0.752,
  "gpqa": 0.4293,
  "gpqa_thinking32_naive": 0.7234,
  "gpqa_thinking32_ex_parse_fail": 0.8193,
  "parse_fail_rate": 0.117,
  "tps_decode_512": 62.5,
  "structured_pass": true,
  "decision": "PROMOTE_READY | NEEDS_MORE_DATA | CLOSED",
  "decision_reasons": ["…"]
}
```

All fields except `schema`, `created`, `reports_dir`, `variant`, `quant`,
`decision`, and `decision_reasons` may be `null` when the source report is
absent. Missing required data → `NEEDS_MORE_DATA` (never crash).

## Data Source Discovery

The script globs `reports/qwen35_9b/` for the latest matching files per
variant, using the same naming conventions already in the repo:

| Metric | File pattern |
|---|---|
| BF16 MMLU/GPQA | `bf16_*_quality_summary.json` |
| NVFP4 MMLU | `nvfp4*_mmlu*.summary.json` |
| NVFP4 GPQA | `nvfp4*_gpqa*.summary.json` |
| Q4_K_M MMLU/GPQA | `q4km_*_quality_summary.json` |
| TPS (Q4_K_M) | `r6000_qwen35_9b_q4km_baseline_*.json` |
| TPS (NVFP4) | `r6000_qwen35_9b_nvfp4_openai_matrix_full_*.json` |
| GPQA Thinking32 | `p201_gpqa_live_summary_*.json` |
| Structured gate | `P196_W4A8_STRUCTURED_CONTENT_GATE_*.md` + `p196_*.json` |
| Release matrix | `qwen35_9b_release_matrix.json` |

## Shell Wrapper

`scripts/qwen35_9b_release_gate_summary.sh` with:

- `--reports-dir DIR` (default: `reports/qwen35_9b`)
- `--out FILE` (default: `reports/qwen35_9b/qwen35_9b_release_gate_summary_latest.json`)

Passes through to the Python script.

## Verification

Local-only, no GPU, no SSH, no server kill:

```bash
python3 -m py_compile scripts/qwen35_9b_release_gate_summary.py
bash -n scripts/qwen35_9b_release_gate_summary.sh
```
