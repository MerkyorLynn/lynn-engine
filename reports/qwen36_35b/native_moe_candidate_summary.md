# Native MoE Candidate Summary

**Generated:** 2026-05-18T23:17:01

## Prerequisites

| Source | Status |
|--------|--------|
| p136 slot-order contract | GREEN (18/18) |
| P140 risk gate | AMBER (recommend_p37=True) |
| P137 diagnostics | present |

## Packed-Slot Readiness (p138/p139)

- **p138 manifest:** `p138_packed_slot_fixtures_manifest_kimi_20260518.json`
- **p139 contract:** `p139_slot_packed_contract_kimi_gz_20260518.json`

| Field | Value |
|-------|-------|
| num_fixtures | 18 |
| packed_fixture_mb | 270.08 |
| bf16_equiv_mb | 864.00 |
| size_reduction_pct | 68.7% |
| p139_verdict | 🟢 GREEN |
| p139_max_abs_max | 0.0 |
| packed_ready_for_kernel | ✅ |
| recommend_next_step | build native packed NVFP4 kernel probe |

## Candidates

| Candidate | slot_max_abs | unique_max_abs | cosine_min | latency (ms) | Verdict | Next Step |
|-----------|-------------|----------------|-----------|-------------|---------|-----------|
| native_slot_output_owned_bf16 (fast, default ref) | 3.906250e-03 | — | 0.9999802113 | 0.0520 | 🔴 CLOSED | no further action |
| native_slot_output_owned_bf16 (slot-order + route-bf16) | 2.929688e-03 | 1.953125e-03 | 0.9999796748 | 0.0523 | 🟡 AMBER_FAST | P37 exploratory |
| native_slot_output_owned_bf16 (dual-ref) | 3.906250e-03 | 3.906250e-03 | 0.9999786615 | 0.0520 | 🔴 CLOSED | no further action |
| native_slot_strict_bf16 (cuBLAS oracle) | 0.000000e+00 | — | 0.9999998808 | 0.4667 | 🔵 EXACT_SLOW | research artifact — too slow for serving |
| native_slot_tc_bf16 (TensorCore probe) | 1.953125e-03 | — | 0.9999891520 | 0.2456 | 🔴 CLOSED | no further action |
| native_slot_fused_bf16 (fused probe) | 1.953125e-03 | — | 0.9999891520 | 0.1088 | 🔴 CLOSED | no further action |
| native_slot_tensorcore_pretransposed_probe (p139b) | 1.953125e-03 | — | 0.9999890924 | 0.0527 | 🟡 AMBER_FAST_PRETRANSPOSED | P37 exploratory |
| native_slot_packed_nvfp4_probe (p140) | 3.906250e-03 | — | 0.9999736547 | 0.0903 | 🔴 CLOSED | stage diagnostics / v2 kernel only |
| packed_dequant_pretransposed_v2 (p141) | 1.953125e-03 | — | 0.9999890924 | 0.0517 | 🟡 AMBER_STAGE | build graph-safe resident ABI, then P37 exact |

## Overall

**Best verdict: 🟡 AMBER_FAST_PRETRANSPOSED**

> 🟡 AMBER candidates exist — no default promote. P37 exploratory permitted only after graph-safe resident ABI is ready.
