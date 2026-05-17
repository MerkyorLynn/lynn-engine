# P119 R6000 MTP In-Place Scratch Parity

Date: 2026-05-17

## Result

`GREEN`: K=2 in-place full-attention KV plus linear-state scratch matches
direct base decode on R6000.

| Metric | Value |
|---|---:|
| Prompt count | 3 |
| Verify events | 24 |
| Passed events | 24 |
| Pass rate | 100% |
| Mean events per prompt | 8.0 |
| Max absolute diff | 0.0 |
| Scratch bytes per token | 64,389,120 |

Raw output:

```text
reports/r6000/p119/p119_mtp_inplace_scratch_parity_r6000_20260517_221000.json
reports/r6000/p119/p119_mtp_inplace_scratch_parity_r6000_20260517_221000.log
```

## Meaning

P119 validates the production-friendly verifier assumption:

- verify can write candidate tokens in-place into full-attention KV;
- reject does not need a full KV rollback;
- `seq_len` clipping makes rejected KV positions invisible;
- only linear recurrent and convolution state need per-token scratch copies.

This lowers the first native verifier target from full-state snapshot/restore to
a narrow scratch/commit boundary for the 30 linear-attention layers.
