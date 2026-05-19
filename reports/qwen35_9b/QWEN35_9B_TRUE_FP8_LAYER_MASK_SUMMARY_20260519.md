# Qwen3.5-9B True-FP8 Layer-Mask Summary

Date: 2026-05-19
Source: `p190_qwen35_9b_true_fp8_resident_gate_*.json` on R6000

## How to Read

Each p190 report tests a specific layer mask (which layers use true FP8 matmul
vs the default NVFP4 scalar-bridge path). The summarizer classifies each mask:

| Verdict | Meaning | Promotable? |
|---------|---------|-------------|
| EXACT_FAST | All tokens match reference AND >= 3% speedup | YES |
| EXACT_FLAT | All tokens match but < 3% speedup | Hold |
| AMBER_LATE_DRIFT | Token drift only after position 32 | Research only |
| CLOSED_EARLY_DRIFT | Token drift before position 32 | Reject |

## Promotion Rule

A layer mask can become DEFAULT only if:
1. P190 exact gate: all prompts produce identical greedy tokens vs reference
2. Speedup >= 3% (otherwise not worth the risk)
3. No regression on P25 512/256 decode TPS

## Running the Summarizer

```bash
# On R6000:
bash scripts/r6000_summarize_qwen35_9b_true_fp8_layer_masks.sh

# Locally (with reports copied):
python scripts/summarize_qwen35_9b_true_fp8_layer_masks.py \
    --reports reports/qwen35_9b/p190_*.json \
    --out-json summary.json \
    --out-md summary.md
```

## Output Files

- `qwen35_9b_true_fp8_layer_mask_summary.json` - Machine-readable full results
- `QWEN35_9B_TRUE_FP8_LAYER_MASK_SUMMARY.md` - Human-readable table

## Context

True FP8 (`torch._scaled_mm` with `float8_e4m3fn`) can accelerate linear
projections on Blackwell TensorCores, but introduces quantization noise.
Layer-mask sweeps identify which layers tolerate FP8 without output drift.
