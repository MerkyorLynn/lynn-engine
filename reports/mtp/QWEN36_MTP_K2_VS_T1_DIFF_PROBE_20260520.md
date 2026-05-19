# Qwen3.6-35B MTP K2-vs-T1 Diff Probe · 2026-05-20

## Verdict

The remaining batched MTP correctness failure is localized to the K=2 linear-attention path, not to the MTP head and not primarily to full-attention QKV/O.

## Key Result

- Pending token: `271` / `'\n\n'`.
- Draft token: `271` / `'\n\n'`.
- Sequential argmax pos0/pos1: `248068`, `248046`.
- Batched argmax pos0/pos1: `248068`, `248046`.
- First bad layer: layer `5` `linear_attention`.

| Position | max_abs | rel_l2 | cosine |
|---|---:|---:|---:|
| pos0 | 0.00195312 | 0.003901 | 0.999992549 |
| pos1 | 0.00097656 | 0.005272 | 0.999986172 |

Final logit drift:

| Position | max_abs | rel_l2 | cosine |
|---|---:|---:|---:|
| pos0 | 0.958984 | 0.071042 | 0.997527659 |
| pos1 | 1.238281 | 0.102565 | 0.995743155 |

## Implication

M12 already proved official MTP head accept is healthy: shadow 81.44%, sequential speculative 75.13%, batched 73.17%. The blocker is K=2 verifier parity. Since first drift appears at a `linear_attention` layer, the next implementation target is a strict K=2 linear-attention fallback that replays two T=1 decode calls with identical recurrent/conv state mutation semantics.

## Next

1. Add an opt-in `LYNN_LINEAR_ATTN_K2_BACKEND=t1_loop` path.
2. In `_decode_layer_k2`, for linear-attention layers only, run two single-token `decode_linear_attn` calls with state updates between positions.
3. Re-run this diff probe. Promotion remains blocked until first_bad_layer is null or below threshold, then rerun MTP smoke.
