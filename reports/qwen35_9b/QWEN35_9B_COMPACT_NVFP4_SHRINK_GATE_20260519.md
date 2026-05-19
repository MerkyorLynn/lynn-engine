# Qwen3.5-9B Compact NVFP4 Shrink Gate

**Date:** 2026-05-19
**Schema:** lynn-compact-nvfp4-shrink-gate-v1
**Status:** Gate definition only — no compact tier is promoted.

---

## Background

The current stable NVFP4 W4A16 artifact for Qwen3.5-9B is **8.248 GiB** —
56% larger than the Q4_K_M baseline (5.3 GiB).  P199 identified the cause:

| Component           | Size (GiB) | Format |
|---------------------|------------|--------|
| Quantized weights   | 4.434      | NVFP4 packed + scales |
| embed_tokens        | 1.895      | BF16 kept |
| lm_head             | 1.895      | BF16 kept |
| Other kept (norms, conv1d, visual) | 0.024 | BF16 |
| Non-tensor metadata | 0.022      | — |

**embed_tokens + lm_head = 3.790 GiB = 46% of total artifact.**

To reach Q4_K_M-like size (~5.3–5.6 GiB), one or both of these tensors must
be quantized from BF16 to NVFP4.

## Why compact NVFP4 is NOT transparent repack

A "transparent repack" preserves all tensor values exactly — only the storage
layout changes.  The current 8.248 GiB NVFP4 artifact IS a transparent repack
of the FP4 checkpoint (weights identical, no quality risk).

Quantizing embed_tokens and/or lm_head changes actual values.  This is an
**experimental quantization** with real quality risk:

1. **lm_head** maps hidden states to 248,320 vocabulary logits.  FP4
   quantization adds noise to every logit, potentially changing top-1 token
   selection.  P136b showed lm_head FP4 exact-match fails on 9B (exact 1/3).

2. **embed_tokens** maps 248,320 token IDs to 4096-dim embeddings.  FP4
   quantization adds noise to the first layer's input, compounding through
   all 36 transformer layers.

3. **Neither tensor is tied** (cosine similarity 0.0198).  They must be
   quantized independently — no shortcut via weight sharing.

## Release Gate Tiers

| Tier | Size (GiB) | Verdict | Risk |
|------|-----------|---------|------|
| SAFE_NO_CHANGE | 8.248 | **PASS_STABLE** | none |
| COMPACT_EMBED_ONLY | ~6.97 | NEEDS_QUALITY_GATE | moderate |
| COMPACT_LMHEAD_ONLY | ~6.97 | NEEDS_QUALITY_GATE | high |
| COMPACT_EMBED_LMHEAD | ~5.70 | NEEDS_FULL_GATE | very high |

**No compact tier is PASS.**  All require quality validation before promotion.

## Size math

```
embed_tokens BF16:  1.895 GiB → NVFP4 estimate: ~0.62 GiB  (saves ~1.27)
lm_head BF16:       1.895 GiB → NVFP4 estimate: ~0.62 GiB  (saves ~1.27)

COMPACT_EMBED_ONLY:    8.248 - 1.895 + 0.62 = 6.97 GiB
COMPACT_LMHEAD_ONLY:   8.248 - 1.895 + 0.62 = 6.97 GiB
COMPACT_EMBED_LMHEAD:  8.248 - 3.790 + 1.24 = 5.70 GiB
```

Note: 5.70 GiB is above the Q4_K_M target (5.3 GiB) because NVFP4 scales
add overhead (~0.89 GiB) that Q4_K_M does not have.  Further shrink would
require pruning visual/MTP tensors (text-only build) or a more aggressive
block size — both out of scope for this gate.

## Required quality gates for compact tiers

Before ANY compact tier can be promoted to PASS:

| Gate | Threshold | Why |
|------|-----------|-----|
| **MMLU-500** | ≥ baseline NVFP4, no >1pp regression | Knowledge benchmark — embed/lm_head changes directly affect token selection |
| **GPQA Diamond** | ≥ baseline NVFP4, no >1pp regression | Reasoning benchmark — sensitive to logit noise |
| **Structured/Content** | 10 prompts, overall GREEN | Practical output quality — format adherence, factual content |
| **32K context smoke** | No crash, no garbage output | Long-context stability — embed noise compounds over sequence |
| **R6000 TPS** | ≥ 90% of baseline NVFP4 tokens/sec | Performance regression check — smaller tensors should be faster, not slower |

### Gate decision rules

- **MMLU or GPQA regression >1pp** → tier is CLOSED, do not promote.
- **Structured/Content RED** → tier is CLOSED.
- **32K smoke fails** → tier is CLOSED.
- **TPS drops >10%** → investigate before promoting (unexpected; smaller should be faster).
- All gates pass → promote tier to PASS_COMPACT.

## Recommendation

1. **Ship SAFE_NO_CHANGE now.**  It is known-good.
2. **Run quality gates on COMPACT_EMBED_LMHEAD** in parallel (highest shrink).
3. If COMPACT_EMBED_LMHEAD passes all gates → promote.
4. If it fails → try COMPACT_EMBED_ONLY or COMPACT_LMHEAD_ONLY individually
   to isolate which tensor is responsible for the regression.

## Files

- `scripts/qwen35_9b_compact_nvfp4_shrink_gate.py` — CPU-only gate script
- `scripts/r6000_qwen35_9b_compact_nvfp4_shrink_gate.sh` — R6000 wrapper
- `reports/qwen35_9b/compact_nvfp4_shrink_gate_*.json` — machine-readable gate output
