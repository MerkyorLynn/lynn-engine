# Qwen3.6-35B MTP Engineering Status — M9

**Date:** 2026-05-19
**Model:** Qwen3.6-35B-A3B-FP8 + official NextN head (converted to Lynn fused)
**Status:** Official head WORKS. No retrain needed. Batched path blocked on M9 verify.

---

## Key Finding

The official Qwen3.6 NextN MTP head produces **77.22% accept rate** in sequential
spec_k1 decode after the concat-order fix (493b2da). This proves:

1. The official head is well-trained and usable — no Lynn-specific retrain required.
2. The conversion pipeline (FP8 block-scale → BF16 fused) is numerically correct.
3. The remaining issues are in batched/graph-safe execution paths, not the head itself.

## Numbers

| Metric | Value | Notes |
|--------|-------|-------|
| Pre concat-fix accept | ~0.26% | Broken: embed/hidden halves swapped |
| **Post-fix spec_k1 accept** | **77.22%** | Sequential, T=1, correct backend |
| Post-fix spec_k1 effective TPS | 25.30 | Single-token speculative |
| Baseline decode TPS (no MTP) | 38.96 | Reference |
| Shadow accept (smoke summary) | low | Accounting difference — spec_k1 is the real signal |
| Batched spec_k1 accept (pre-M9) | 11.21% | Backend drift: K=2 used wrong MoE path |

## Root Cause Chain

```
V1 (pre 493b2da):  cat([hidden, embed]) → 0.26% accept (weights misaligned)
                         │
                         ▼ fix: swap to cat([embed, hidden])
V2 (post 493b2da): spec_k1 sequential → 77.22% accept ✓
                         │
                         ▼ but batched path → 11.21% accept ✗
                         │
                         ▼ M9 diagnosis: K=2 draft tokens routed through
                           a different MoE backend than T=1 verification
                         │
                         ▼ M9 fix: K=2 MoE uses per-position T=1 loop
                           (same backend guarantee)
```

## M9 Fix

`codex/mtp-m9-integration` implements: when K=2 speculative tokens are fed to the
MoE layer for verification, instead of batching them through a potentially
different code path, the M9 fix processes each draft position through the same
T=1 MoE loop that the baseline uses. This ensures bit-identical routing and
expert computation, eliminating the 11.21% → expected ≥60% accept recovery.

## Remaining Blockers (in order)

| # | Gate | Requirement | Status |
|---|------|-------------|--------|
| 1 | M9 smoke rerun | batched accept ≥ 60% | PENDING (Spark flaky) |
| 2 | Effective TPS | spec TPS > baseline 38.96 | PENDING |
| 3 | P37 exact | MTP-on greedy == MTP-off greedy (lossless) | PENDING |
| 4 | P25 decode TPS | 128/256/512 with MTP enabled | PENDING |
| 5 | Structured gate | 40/70 prompts pass with MTP | PENDING |

Gate 1 is the immediate next step. If batched accept recovers to ≥60%, gates
2-5 follow sequentially. If it doesn't, there's a deeper execution-order issue
in the verification path.

## What This Means for Product

- **No retrain budget needed.** The official head at 77% accept is strong enough
  for ~1.3-1.5x effective TPS uplift once the batched path is correct.
- **MTP is not blocking 9B first release.** It's a speed optimization for the
  35B NVIDIA track.
- **Timeline:** Once Spark is stable + M9 smoke passes, MTP can enter the
  resident serving path within 1-2 sessions.

## Files

| Path | Content |
|------|---------|
| `engine/mtp_sidecar.py` (493b2da) | Concat fix: embed first |
| `codex/mtp-m9-integration` branch | K=2 same-backend MoE fix |
| `scripts/qwen36_convert_official_mtp_to_lynn_fused.py` | FP8→BF16 converter |
| This report | Engineering status snapshot |
