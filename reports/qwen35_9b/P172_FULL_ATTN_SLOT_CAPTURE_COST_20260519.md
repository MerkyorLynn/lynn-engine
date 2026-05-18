# P172 Qwen3.5-9B Full-Attention Slot Capture Cost

Date: 2026-05-19

## Purpose

P171 proved that real-state full-attention graph slots are numerically strict
for one decode token.  P172 adds the missing service-readiness check: the
capture cost for the full-attention slots.  Because these slots are fixed to
the current sequence length, they cannot be assumed reusable across subsequent
decode positions.

## R6000 Result

| Metric | Result |
|---|---:|
| Strict logit pass | true |
| Greedy pass | true |
| Full-attn slots | 8 |
| Full-slot capture time | 66.70 ms |
| Eager token | 24.37 ms |
| Graph replay token | 16.14 ms |
| Replay-only speedup | 1.51x |
| Single-use graph time with capture | 82.75 ms |
| Single-use speedup with capture | 0.295x |

## Decision

Do not wire full-attention graph slots into the 9B resident service path yet.
Replay is fast and exact, but single-use capture cost dominates.  The next
usable implementation would need a real dynamic-shape or reusable full-attn
graph ABI that handles changing sequence length without recapturing each token.

For now, the 9B speed mainline should return to larger exact dense-FFN or
linear/SSM boundaries.

## Artifacts

- `benchmarks/p9v_hybrid_real_state_full_attn_slots_probe.py`
- `reports/qwen35_9b/p172_full_attn_slot_capture_cost_qwen35_9b_20260519_0920.json`
