# Qwen3.6-35B MTP M18 Layer-Type Sweep Result

Date: 2026-05-20

## Summary

M18 completed on Spark and localized the remaining K=2 batched verifier drift.

Safe combinations:

- `t1_full_attn_only`
- `t1_both`

Unsafe combinations:

- `k2_both`
- `t1_linear_attn_only`

This means the immediate correctness bridge is:

```bash
LYNN_FULL_ATTN_K2_BACKEND=t1_loop
LYNN_MTP_K2_LINEAR_ATTN_MODE=k2
```

In plain terms: full-attention K=2 batching is the unsafe piece; linear-attention
K=2 can remain enabled for this probe.

## Results

| Combination | Full Attention | Linear Attention | Result |
|---|---|---|---|
| `k2_both` | K=2 | K=2 | DRIFT at layer 5 linear-attention |
| `t1_full_attn_only` | 2x T=1 | K=2 | EXACT |
| `t1_linear_attn_only` | K=2 | 2x T=1 | DRIFT at layer 5 linear-attention |
| `t1_both` | 2x T=1 | 2x T=1 | EXACT |

The first bad layer for the unsafe paths is layer 5, but the mode comparison
shows it is caused by upstream full-attention K=2 drift propagating into the
next linear-attention layer. Switching only linear-attention to T=1 does not
fix the issue; switching only full-attention does.

## Artifacts

- `reports/mtp/remote_spark_20260520/mtp_m18_k2_layer_sweep_retry_104335.json`
- Spark source path: `/tmp/mtp_m18_k2_layer_sweep_retry_104335.json`

## Next

Run a focused MTP smoke with:

```bash
LYNN_FULL_ATTN_K2_BACKEND=t1_loop
```

Acceptance criteria before any promotion:

- `spec_k1_batched` accept returns to the sequential head level, roughly 70%+.
- `spec_k1_batched` correctness matches baseline on the smoke prompts.
- Effective TPS beats baseline; otherwise MTP is correctness-unblocked but still
  not a serving-speed win.
