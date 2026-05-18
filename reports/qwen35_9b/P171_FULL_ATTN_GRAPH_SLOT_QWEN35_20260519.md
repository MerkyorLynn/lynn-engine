# P171 Qwen3.5-9B Full-Attention Graph-Slot Probe

Date: 2026-05-19

## Purpose

P170 showed that dense gate/up fusion is exact but too small to move service
TPS on its own.  P171 checks the next larger exact-first boundary for the 9B
dense route: keep the safe reusable 3-layer linear/SSM block graphs, and add
per-request real-state full-attention graph slots.

The probe also fixes older 35B-specific benchmark assumptions so P28/P9V use
the runner's actual `layer_types` and `LynnInferenceState.from_config(...)`.
That matters for Qwen3.5-9B because its KV head shape differs from the 35B
default.

## R6000 Result

| Metric | Result |
|---|---:|
| Full-attn graph slots | 8 |
| Linear block graphs | 8 |
| Strict logit pass | true |
| Greedy pass | true |
| Logit max abs | 0.0 |
| Eager token | 23.8800 ms |
| Graph token | 16.1783 ms |
| Probe speedup | 1.4761x |
| One-shot graph token | 16.0660 ms |
| One-shot graph TPS | 62.24 |

P28 on the same profile reports 10.69 ms/token across linear blocks and
4.31 ms/token across full-attention layers, so full-attention graphing is a
real but secondary island behind linear/SSM.

## Decision

Proceed to a guarded resident experiment for Qwen3.5-9B only: per-request
full-attention graph slots are exact in the P9V shape and have enough local
speedup to justify a P172 implementation probe.  Do not enable cross-request
full-attn graph reuse; earlier 35B P9W showed cross-prompt reuse is unsafe.

## Acceptance For P172

P172 should be opt-in and default-off, then pass:

1. direct greedy exactness against the safe `linear_graph_only` profile;
2. P25 512 decode TPS clearly above the current 61.85 fused-gate/up service
   result, with a practical target around 68 TPS;
3. no cross-request reuse unless a separate parity gate proves it.

## Artifacts

- `benchmarks/p28_hybrid_block_timing_profile.py`
- `benchmarks/p9n_hybrid_full_attn_graph_slots_probe.py`
- `benchmarks/p9v_hybrid_real_state_full_attn_slots_probe.py`
- `reports/qwen35_9b/p171_hybrid_block_timing_qwen35_9b_20260519_0900.json`
- `reports/qwen35_9b/p171_real_state_full_attn_slots_qwen35_9b_20260519_0900.json`
