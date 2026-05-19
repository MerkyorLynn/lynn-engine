# Qwen3.6-35B-A3B MTP M14 Batched Commit Trace

**Date:** 2026-05-20  
**Host:** Spark  
**Model:** `Qwen3.6-35B-A3B-lynn-native-w4a16-nvfp4-from-r6000`

## What Changed

M13 proved the official MTP head is alive: shadow accept is 81.44% and batched
speculative accept is 75.17%.  However, batched exact parity was only 2/6.

M14 added a smaller event-level trace and a state-diff extension to the K2-vs-T1
probe.

## Findings

1. The first end-to-end divergence happens at speculative event 26.
2. The failing event is `accepted=true`, but the wrong token is the pending
   token carried from the previous event, not the draft token itself.
3. A reject-path safety fix now uses the canonical T1 re-decode argmax after
   reject instead of reusing K2 `argmax_at_pos0`.
4. That fix did not change the first divergence, which means the state had
   already drifted before the failing event.
5. A single-step state-diff probe with `LYNN_FULL_ATTN_K2_BACKEND=t1_loop`
   reports exact K2-vs-two-T1 parity for hidden, logits, KV, recurrent, and
   conv state.

## Key Numbers

| Probe | Result |
|---|---|
| M13 shadow accept | 81.44% |
| M13 spec_k1 accept | 75.13% |
| M13 spec_k1 exact | 6/6 |
| M13 spec_k1_batched accept | 75.17% |
| M13 spec_k1_batched exact | 2/6 |
| M13 spec_k1_batched TPS ratio | 0.766x |
| M14 first divergence | event 26, prefix 39 |
| M14 single-step state diff | hidden/logits/KV/recurrent/conv exact |

## Interpretation

The MTP head and single-step K2 verifier are no longer the blockers.  The
remaining correctness issue is multi-event state-path drift across accepted and
rejected speculative rounds.  The current strict fallback is also slower than
baseline, so it is not a promotion candidate even if fully corrected.

To turn MTP into TPS credit, the next route should be a true batched verifier
path that is both state-stable across many events and faster than two T1
forwards.  Continuing to stack strict T1-loop fallbacks is useful for diagnosis
but not for serving speed.

## Artifacts

- Commit trace before reject fix: `reports/mtp/mtp_batched_commit_trace_20260520_013255.json`
- Commit trace after reject fix: `reports/mtp/mtp_batched_commit_trace_rejectfix_20260520_013926.json`
- State-diff probe: `reports/mtp/mtp_k2_vs_t1_state_diff_20260520_014604.json`
- M13 smoke: `reports/mtp/mtp_smoke_m13_fullattn_t1loop_20260520_011344.json`
