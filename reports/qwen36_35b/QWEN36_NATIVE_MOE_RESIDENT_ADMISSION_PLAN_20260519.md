# Qwen3.6-35B Native MoE Resident Admission Plan

**Date:** 2026-05-19
**Status:** SCAFFOLD READY — awaiting P37 exact pass before promotion
**Model:** Qwen3.6-35B-A3B W4A16 NVFP4

---

## Background

The native MoE kernel path (slot/pretransposed/graph-safe) achieves fixture-level
correctness at ~0.044ms latency (vs Triton active ~0.059ms). However, strict
numeric P37 remains AMBER: the native path produces correct outputs at fixture
level but does not achieve bit-exact greedy token match against the Triton
baseline in end-to-end generation.

Root cause (proven in P145/V3.2): BF16 truncation of dequantized weights before
cuBLAS mm introduces ~1e-3 per-element error that accumulates to flip argmax at
low-margin tokens. This is fundamental to any dequant-to-BF16-then-cuBLAS bridge.

## Admission Ladder (Sequential, Fail-Loud)

```
┌─────────────────────────────────┐
│  Stage 1: P136 Fixture Contract │  Slot repack exact-match
└─────────┬───────────────────────┘
          │ PASS
┌─────────▼───────────────────────┐
│  Stage 2: P139 Packed Contract  │  NVFP4 dequant exact-match
└─────────┬───────────────────────┘
          │ PASS
┌─────────▼───────────────────────┐
│  Stage 3: P37 Graph-On Exact    │  3p × 128t greedy identical
└─────────┬───────────────────────┘  ← FAIL = CLOSED, stop here
          │ PASS
┌─────────▼───────────────────────┐
│  Stage 4: P25 Decode TPS        │  128/256/512 single-stream
└─────────┬───────────────────────┘
          │ Record
┌─────────▼───────────────────────┐
│  Stage 5: Structured 40/70      │  Parse correctness + content
└─────────┬───────────────────────┘
          │ PASS
┌─────────▼───────────────────────┐
│  CANDIDATE_READY                │  → Manual review → P25 serving
└─────────────────────────────────┘
```

**Key rule:** P37 exact is the hard gate. If P37 fails, the candidate is CLOSED
and P25/structured are NOT run. This prevents wasting GPU time on candidates that
cannot reproduce the Triton baseline.

## Current Candidates

| Candidate | P136 | P139 | P37 | Notes |
|-----------|------|------|-----|-------|
| `packed_pretransposed_graphsafe_v31` | PASS | PASS | AMBER | Custom reduce, 0.044ms |
| `packed_pretransposed_graphsafe_v32_ordered` | PASS | PASS | AMBER | Sequential mm, still AMBER |

Both fail P37 for the same fundamental reason: BF16 truncation of effective
weights. The native path's dequant is mathematically correct, but the resulting
BF16 matmul accumulates differently than Triton's FP32 inner loop.

## Possible Resolution Paths

| Path | Description | Effort | Expected P37 |
|------|-------------|--------|--------------|
| A: FP32 inner loop | Keep weights in FP32, matmul in FP32 | Medium | PASS (matches Triton) |
| B: Accept AMBER | Document drift as numerical noise, not semantic | Low | N/A |
| C: Triton-native FP4 | Use tl.dot_scaled with E2M1 directly | High | Depends |
| D: SM120a MMA | Use the real E4M3×E2M1 MMA instruction | High | Unknown |

Path A is the most likely to achieve P37 exact: dequant NVFP4 to FP32 (not BF16),
then use FP32 matmul. This matches Triton's behavior where the inner loop
accumulates in FP32 before truncating to output.

## Env Variables for Candidate Control

The admission script accepts candidate environments via file or inline:

```bash
# Via file:
CANDIDATE_ENV_FILE=scripts/qwen36_candidate_env_moe_repack_scratch.env

# Via inline:
CANDIDATE_ENV="LYNN_NATIVE_ACTIVE_MOE_BACKEND=packed_pretransposed_graphsafe_v31"
```

Available backends:
- `packed_pretransposed_graphsafe_v31` — allocation-free, custom reduce
- `packed_pretransposed_graphsafe_v32_ordered` — sequential for P37 attempt
- `triton` — Triton active (the baseline)

## Summary JSON Schema

```json
{
  "schema": "lynn-native-moe-resident-admission-v1",
  "candidate_name": "v31_graphsafe",
  "fixture_status": "PASS",
  "p37_exact": false,
  "p25_512_decode_tps": null,
  "structured_40": null,
  "structured_70": null,
  "decision": "CLOSED",
  "reason": "p37_exact: NOT exact — candidate rejected"
}
```

## Run Commands

```bash
# Dry-run (see the full command chain):
DRY_RUN=1 bash scripts/r6000_qwen36_native_moe_resident_admission.sh

# With specific candidate:
CANDIDATE_NAME=v31_graphsafe \
CANDIDATE_ENV="LYNN_NATIVE_ACTIVE_MOE_BACKEND=packed_pretransposed_graphsafe_v31" \
DRY_RUN=0 bash scripts/r6000_qwen36_native_moe_resident_admission.sh
```

## References

| File | Content |
|------|---------|
| `benchmarks/p136_moe_slot_repack_contract.py` | Stage 1 fixture probe |
| `benchmarks/p139_moe_slot_packed_contract.py` | Stage 2 packed probe |
| `benchmarks/p37_moe_config_generate_gate.py` | Stage 3 exact gate |
| `benchmarks/p25_server_decode_tps_probe.py` | Stage 4 TPS probe |
| `scripts/qwen36_structured_hard_prompts_70.json` | Stage 5 prompts |
| `scripts/r6000_qwen36_candidate_promotion_gate.sh` | Prior art (36B promotion) |
