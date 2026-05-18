# Native MoE Candidate Summary

**Generated:** 2026-05-18T22:35:24

## Prerequisites

| Source | Status |
|--------|--------|
| p136 slot-order contract | GREEN (18/18) |
| P140 risk gate | AMBER (recommend_p37=True) |
| P137 diagnostics | present |

## Candidates

| Candidate | slot_max_abs | unique_max_abs | cosine_min | latency (ms) | Verdict | Next Step |
|-----------|-------------|----------------|-----------|-------------|---------|-----------|
| native_slot_output_owned_bf16 (fast, default ref) | 3.906250e-03 | — | 0.9999802113 | 0.0520 | 🔴 CLOSED | no further action |
| native_slot_output_owned_bf16 (slot-order + route-bf16) | 2.929688e-03 | 1.953125e-03 | 0.9999796748 | 0.0523 | 🟡 AMBER_FAST | P37 exploratory |
| native_slot_output_owned_bf16 (dual-ref) | 3.906250e-03 | 3.906250e-03 | 0.9999786615 | 0.0520 | 🔴 CLOSED | no further action |
| native_slot_strict_bf16 (cuBLAS oracle) | 0.000000e+00 | — | 0.9999998808 | 0.4667 | 🔵 EXACT_SLOW | research artifact — too slow for serving |
| native_slot_tc_bf16 (TensorCore probe) | — | — | — | — | ⚪ MISSING | no report found |
| native_slot_fused_bf16 (fused probe) | — | — | — | — | ⚪ MISSING | no report found |

## Overall

**Best verdict: 🟡 AMBER_FAST**

> 🟡 AMBER candidates exist — no default promote. P37 exploratory permitted if P140 gate clears.
