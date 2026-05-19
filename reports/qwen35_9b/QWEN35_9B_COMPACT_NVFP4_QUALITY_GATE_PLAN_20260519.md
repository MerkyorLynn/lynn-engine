# Qwen3.5-9B Compact NVFP4 Quality Gate Plan

**Date:** 2026-05-19
**Status:** SCAFFOLD READY — awaiting compact model artifact + R6000 GPU time

---

## Context

The current stable NVFP4 W4A16 artifact is **8.248 GiB**. P199/P200 proved that
quantizing `embed_tokens` (1.895 GiB) and/or `lm_head` (1.895 GiB) from BF16 to
FP4 can shrink the artifact to **5.70 GiB** — competitive with Q4_K_M (5.49 GiB).

**Critical fact:** This is NOT a transparent repack. Quantizing `lm_head` changes
the final logit computation. Quantizing `embed_tokens` changes the input
representation. Both can degrade quality in ways that only show up under MMLU/GPQA.

P136b showed that `lm_head` FP4 exact-match gate **FAILS** on 9B (1/3 exact).
Therefore, compact NVFP4 cannot be promoted without a full quality gate.

---

## Tier Definitions

| Tier | What Changes | Expected Size | Risk |
|------|-------------|---------------|------|
| **SAFE_NO_CHANGE** | Nothing (ship current 8.25 GiB) | 8.248 GiB | None — known-good |
| **EMBED_ONLY** | Quantize `embed_tokens` to FP4 | ~6.98 GiB | Low (embed is overcomplete) |
| **LMHEAD_ONLY** | Quantize `lm_head` to FP4 | ~6.98 GiB | **High** (P136b: exact fail) |
| **EMBED_LMHEAD** | Quantize both to FP4 | ~5.70 GiB | **Highest** (compounds both risks) |

Each tier must pass the full quality gate independently. A tier that fails
is rejected; lower tiers remain candidates.

---

## Quality Gate Thresholds

All thresholds are relative to the stable W4A16 8.25 GiB baseline:

| Metric | Stable Baseline | Floor (promote) | Notes |
|--------|----------------|-----------------|-------|
| MMLU 500 5-shot | 75.20% | ≥ 74.20% | Max 1pp regression |
| GPQA Diamond | 42.93% | ≥ 41.93% | Max 1pp regression |
| Structured 70-prompt | GREEN | GREEN | All prompts must parse |
| 32K context | no crash | no crash | Non-empty response required |
| Decode TPS (512) | ~61 TPS | ≥ 55 TPS | ≥ 90% of stable |

If any metric fails, the tier is **REJECTED** for first release.

---

## Promotion Rules

1. Only `SAFE_NO_CHANGE` is currently promoted (the stable 8.25 GiB track).
2. A compact tier advances to `CANDIDATE` only after ALL gate metrics pass.
3. A `CANDIDATE` advances to `PROMOTED` only after manual review confirms
   no regressions in qualitative generation (code, Chinese, JSON, reasoning).
4. Compact NVFP4 is **never** the default first-release track. It can only
   be offered as an optional smaller artifact if promoted.

---

## Execution Plan

### Prerequisites

1. Create the compact artifact (requires `tools/nvfp4_repack.py` or equivalent)
2. Start Lynn server with the compact model on R6000
3. Ensure MMLU dataset + GPQA CSV are available

### Commands

```bash
# Dry-run (see what will run):
DRY_RUN=1 bash scripts/r6000_qwen35_9b_compact_nvfp4_quality_gate.sh

# Real execution (after server is running):
DRY_RUN=0 MODEL_DIR=/root/autodl-tmp/models/Qwen3.5-9B-lynn-native-compact-nvfp4-candidate \
  bash scripts/r6000_qwen35_9b_compact_nvfp4_quality_gate.sh
```

### Output Structure

```
reports/qwen35_9b/compact_quality_gate_<stamp>/
├── mmlu_500_5shot.jsonl           # Per-question MMLU results
├── mmlu_500_5shot.summary.json    # MMLU accuracy summary
├── gpqa_diamond.jsonl             # Per-question GPQA results
├── gpqa_diamond.summary.json      # GPQA accuracy summary
├── structured_smoke.jsonl         # Structured prompt results
├── long_context_32k.json          # 32K smoke result
├── p25_decode_tps.json            # TPS measurements
└── compact_quality_gate_summary.json  # Final gate verdict
```

---

## What This Plan Does NOT Do

1. Does NOT create the compact model artifact (separate tooling required)
2. Does NOT bypass the quality gate for any tier
3. Does NOT promise compact NVFP4 will ship in the first release
4. Does NOT apply to the Mac/Q4_K_M track (which is separate)
5. Does NOT test W4A8/FP4xFP8 (that is Track 3 experimental, separate gates)

---

## References

| Source | Content |
|--------|---------|
| `reports/qwen35_9b/compact_nvfp4_shrink_gate_20260519_live_compact.json` | P200 tier analysis |
| `reports/qwen35_9b/P199_QWEN35_9B_NVFP4_SIZE_AUDIT_20260519.md` | Size breakdown |
| `reports/qwen35_9b/QWEN35_9B_RELEASE_EVIDENCE_INDEX_20260519.md` | Release evidence |
| `scripts/r6000_qwen35_9b_compact_nvfp4_quality_gate.sh` | Gate runner script |
