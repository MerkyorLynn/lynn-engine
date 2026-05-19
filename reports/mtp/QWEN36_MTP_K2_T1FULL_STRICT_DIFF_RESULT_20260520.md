# Qwen3.6-35B MTP K2 T1-Full-Attn Strict Diff Result · 2026-05-20

## Verdict

`LYNN_FULL_ATTN_K2_BACKEND=t1_loop` makes the K=2 verifier numerically strict against two sequential T=1 decode steps for the tested prompt.

This proves the M12 batched correctness failure was caused by the true batched full-attention SDPA path, not by the official MTP head, lm_head dispatch, MoE, or linear/GDN state rollout.

## Metrics

- First bad layer: `None`.
- Sequential argmax pos0/pos1: `248068`, `248046`.
- K2 argmax pos0/pos1: `248068`, `248046`.

| Position | logit max_abs | rel_l2 | cosine |
|---|---:|---:|---:|
| pos0 | 0.000000 | 0.000000 | 0.999999881 |
| pos1 | 0.000000 | 0.000000 | 1.000000000 |

## Next

Run M13 smoke with `LYNN_FULL_ATTN_K2_BACKEND=t1_loop`. Promotion still depends on full-prompt exactness and effective TPS, but the K=2 verifier math is now strict in the layer-level probe.
