# Qwen3.6 MTP M16 K=2 Bisect Result - 2026-05-20

## Verdict

The batched K=2 verifier drift is not an event-5-only bug. It reproduces with zero advance tokens, so the remaining MTP blocker is a fundamental mismatch between the K=2 verifier path and the canonical two-step T=1 decode path.

This confirms that the official MTP head is usable, but the batched verifier is not yet safe for promotion.

## Source

- Remote Spark JSON: `reports/mtp/remote_spark_20260520/mtp_m16_k2_bisect_20260520_032928.json`
- Probe: `scripts/spark_mtp_m16_k2_bisect_probe.py`

## Result Summary

| Advance Count | First Bad Layer | Layer Type | Pos0 Bad | Pos1 Bad | Worst State Drift |
|---:|---:|---|---|---|---|
| 0 | 5 | linear_attention | true | true | conv layer 20 max_abs 0.6875 |
| 1 | 4 | linear_attention | true | false | conv layer 38 max_abs 0.421875 |
| 2 | 4 | linear_attention | false | true | conv layer 38 max_abs 2.125 |
| 4 | 3 | full_attention | true | false | conv layer 24 max_abs 0.625 |
| 8 | 4 | linear_attention | false | true | conv layer 38 max_abs 0.5 |

The logits drift is already large at zero advance:

- Pos0 logits: max_abs 0.9238, cosine 0.99608
- Pos1 logits: max_abs 0.7422, cosine 0.99787

## Interpretation

M15 localized the production trace's first visible bad boundary to event 5 and layer 32. M16 shows that this was not the root cause; it was only the first place where drift crossed the previous threshold in that particular prefix.

The actual root is earlier:

1. K=2 verifier execution diverges from two canonical T=1 decode steps even immediately after prefill.
2. The first bad layer moves with prefix length, alternating between early linear attention and full attention layers.
3. Conv/recurrent state drift accumulates downstream, often peaking in layer 38.

This means small repairs to MTP bookkeeping, concat order, lm_head dispatch, or MoE backend are already exhausted. The remaining work is to make the K=2 verifier path numerically equivalent to two T=1 steps, then re-introduce batching only where exactness survives.

## Next Patch Direction

Do not run another full MTP smoke until a small probe passes.

Recommended next patch:

1. Add an opt-in strict verifier mode such as `LYNN_MTP_K2_VERIFY_MODE=t1_canonical`.
2. In that mode, compute the two verification positions by executing the exact T=1 decode path twice on a cloned verifier state.
3. Confirm M16 becomes exact for all tested advance counts.
4. If exact, use it as the correctness oracle and re-enable K=2 batching layer-by-layer, starting with full-attention projections and linear-attention state updates.

The expected performance of strict canonical mode is not the final target; it is a correctness scaffold for safely recovering K=2 speed.
