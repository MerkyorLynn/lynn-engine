# Qwen3.6-35B Native MoE Resident Admission

**Date:** 2026-05-19
**Owner:** Stream M (Native MoE / packed boundary)
**Status:** SCAFFOLD — admission script ready, P37 exact is the current blocker

---

## Purpose

This document tracks the path from fixture-level AMBER to resident-level PASS
for the Lynn native MoE kernel on Qwen3.6-35B-A3B.

"Resident" means the kernel is safe to use in the default serving path:
- Graph-captured, allocation-free
- Bit-exact with the Triton baseline (P37)
- Performance at or above the Triton active path
- Structured content quality preserved

## Current State

| Metric | Status | Notes |
|--------|--------|-------|
| Fixture correctness (P136/P139) | PASS | Slot repack and packed dequant are exact |
| Latency | 0.044ms (graph-safe V3.1) | Faster than Triton active 0.059ms |
| P37 exact match | **AMBER** | BF16 truncation causes argmax flips at low margins |
| Resident promotion | **BLOCKED** | Cannot proceed without P37 exact |

## Admission Script

```bash
scripts/r6000_qwen36_native_moe_resident_admission.sh
```

### Stages (Sequential, Fail-Loud)

| Stage | Gate | Pass Criteria | Stop-on-Fail |
|-------|------|---------------|--------------|
| 1 | P136 fixture contract | Strict numeric match | No (continue) |
| 2 | P139 packed contract | Strict numeric match | No (continue) |
| 3 | P37 graph-on exact | 3p × 128t greedy identical | **YES** → CLOSED |
| 4 | P25 decode TPS | 128/256/512 measured | No (record) |
| 5 | Structured 40/70 | All prompts parse | No (record) |

### Usage

```bash
# Dry-run:
DRY_RUN=1 bash scripts/r6000_qwen36_native_moe_resident_admission.sh

# Real execution with candidate env:
CANDIDATE_NAME=v31_graphsafe \
CANDIDATE_ENV_FILE=scripts/qwen36_candidate_env_moe_repack_scratch.env \
DRY_RUN=0 bash scripts/r6000_qwen36_native_moe_resident_admission.sh
```

## Known Blocker: BF16 Truncation

The native MoE dequant path produces mathematically correct FP32 values from
NVFP4 packed weights. However, the subsequent cuBLAS mm requires BF16 input,
which truncates effective weights to ~7-bit mantissa. Over 8 top-k experts
× 3 projections × position-dependent routing, this accumulated truncation
shifts logits enough to flip argmax at ~5-20% of low-margin tokens.

This is not a kernel bug — it's a fundamental precision mismatch between:
- Triton: FP32 accumulation in inner loop (tl.dot keeps FP32 partials)
- Native: BF16 truncation → cuBLAS FP32 accumulation (different rounding)

## Resolution Candidates

1. **FP32 weight buffer**: Skip BF16 truncation, feed FP32 to matmul
2. **Accept AMBER**: Document as numerical noise, ship with quality gate
3. **True FP4 MMA (SM120a)**: Bypass dequant entirely (P191 blocker)

## Dependencies

- No 9B eval interaction (9B is a different model/track)
- No server/HTTP changes
- No csrc kernel changes in this scaffold (kernel work is separate PRs)

## Output

```
reports/qwen36_35b/native_moe_admission_<candidate>_<stamp>/
├── p136_fixture.json
├── p139_packed.json
├── p37_exact.json
├── p25_decode_tps.json
├── structured_40.json
├── structured_70.json
└── admission_summary.json
```
