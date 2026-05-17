# P118 R6000 MTP Verify State Parity

Date: 2026-05-17

## Result

`GREEN`: Python K=2 verify accept/reject commit semantics match direct base
decode on R6000.

| Metric | Value |
|---|---:|
| Prompt count | 3 |
| Verify events | 24 |
| Passed events | 24 |
| Pass rate | 100% |
| Mean events per prompt | 8.0 |
| Max absolute diff | 0.0 |

Raw output:

```text
reports/r6000/p118/p118_mtp_verify_state_parity_r6000_20260517_220350.json
reports/r6000/p118/p118_mtp_verify_state_parity_r6000_20260517_220350.log
```

## Meaning

P118 does not claim a TPS win by itself. It freezes the commit/rollback
semantics needed for a real K=2/K=3 verifier:

- accepted draft path commits state after `[base, draft]`;
- rejected draft path commits only the base token and discards scratch state;
- the next base decode remains bit-identical to the direct baseline.

This clears the next implementation step: replace Python clone-based scratch
with a native scratch/commit boundary for recurrent, convolution, and KV state.
