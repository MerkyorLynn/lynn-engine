# Qwen3.6 35B Native MoE Slot-Order Strict Ground Truth

Date: 2026-05-18

## What Changed

`p135_repack_moe_fixture_slots.py` now stores two routed references:

- `routed_output`: slot-order PyTorch reference, matching the p136 fixture contract.
- `unique_routed_output`: original unique-expert/index_add reference, kept for serving-risk analysis.

This separates two questions that were previously mixed:

1. Is the slot-repacked fixture mathematically self-consistent?
2. How far does slot-order accumulation drift from the resident serving reference?

## R6000 Results

| Probe | Result |
| --- | --- |
| p136 slot-order contract | GREEN |
| p136 fixtures passed | 18/18 |
| p136 max_abs_max | 0.0 |
| p136 slot_repack_ms_mean | 0.473 ms |
| fast native slot latency | 0.0523 ms |
| fast native slot max_abs vs slot reference | 0.0029296875 |
| fast native unique max_abs | 0.001953125 |
| strict cuBLAS slot max_abs | 0.0 |
| strict cuBLAS latency | 0.4667 ms |

## Interpretation

KIMI's fixture-ground-truth diagnosis is correct: p136 was RED because p135 stored a unique-expert/index_add reference while p136 recomputed slot-order output. Regenerating `routed_output` with the slot-order loop makes p136 exact.

Claude's strict cuBLAS reference is also correct: cuBLAS/`torch::mm` can reproduce slot-order PyTorch exactly, but it is about 8x slower than the 0.059 ms Triton active reference and cannot be a serving candidate.

The fast native custom kernel remains useful but not strict. Truncating routing weights to BF16 inside the CUDA down kernel improved the fast path:

- slot max_abs improved from 0.00390625 to 0.0029296875
- unique max_abs improved from 0.00390625 to 0.001953125
- latency stayed about 0.052 ms

Stage diagnostics still show the remaining drift is dominated by scalar native gate/up and down reduction semantics rather than fixture layout.

## Verdict

- `p135/p136` slot-order fixture harness: GREEN and should be kept.
- `native_slot_strict_bf16`: exact research oracle, too slow for serving.
- `native_slot_output_owned_bf16`: fast research candidate, not resident-safe yet.

Do not escalate the fast native slot candidate to P37/P25 until its fixture-level strictness improves or an explicit AMBER drift budget is approved.

Artifacts:

- `reports/qwen36_35b/p135_repacked_fixtures_slotorder_manifest_20260518.json`
- `reports/qwen36_35b/p136_slot_repack_contract_slotorder_report_20260518.json`
- `reports/qwen36_35b/native_slot_output_owned_bf16_slotorder_routebf16_report_20260518.json`
- `reports/qwen36_35b/p137_moe_slot_stage_diagnostics_slotorder_routebf16_20260518.json`
- `reports/qwen36_35b/native_slot_strict_bf16_slotorder_report_20260518.json`
