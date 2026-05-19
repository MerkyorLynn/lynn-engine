# Qwen3.6 Official MTP Concat Fix — Smoke Result

**Date:** 2026-05-19
**Commit:** `493b2da` — mtp_logits concat order → embed FIRST
**Status:** AWAITING SPARK (unreachable from this session)

---

## Background

Before the fix, `mtp_logits` had:
```python
torch.cat([hidden_part, embed_part], dim=-1)  # WRONG
```

After fix (493b2da):
```python
torch.cat([embed_part, hidden_part], dim=-1)  # CORRECT per vLLM
```

Pre-fix results: 0.0% shadow accept, 0.26% spec_k1 accept.

## Spark Smoke File

```
/home/merkyor/reports/spark/mtp_smoke_concat_fix_20260519_225724.json
```

## Results (PENDING — Spark unreachable)

| Metric | Value |
|--------|-------|
| baseline_mean_decode_tps | — |
| shadow_mean_decode_tps | — |
| mean_shadow_accept_rate | — |
| spec_k1_effective_tps | — |
| spec_k1_accept_rate | — |
| spec_k1_batched_effective_tps | — |
| spec_k1_batched_accept_rate | — |

## Decision Tree

- **If shadow_accept ≥ 30%**: MTP is working. Write success report, proceed to K=2 probe.
- **If shadow_accept 5-30%**: Partial success. Investigate per-prompt variance, check if certain prompt types have higher accept.
- **If shadow_accept < 5%**: Concat fix alone insufficient. Run diagnostic sweep:
  - pos_offset: is `current_pos` correct or off by ±1?
  - token_embed source: base embed_tokens vs MTP-specific embed?
  - Attention mask: does the MTP layer see the correct causal context?

## Next Steps (when Spark is accessible)

```bash
# 1. Read the result:
ssh dgx "cat /home/merkyor/reports/spark/mtp_smoke_concat_fix_20260519_225724.json"

# 2. If accept < 5%, run diagnostic:
ssh dgx "cd /home/merkyor/lynn-engine && python scripts/mtp_concat_fix_diagnostic_sweep.py \
  --model /home/merkyor/models/Qwen3.6-35B-A3B-FP8 \
  --sidecar /home/merkyor/models/Qwen3.6-35B-A3B-FP8/mtp_lynn_fused.safetensors \
  --out /home/merkyor/reports/spark/mtp_diagnostic_sweep.json"
```
