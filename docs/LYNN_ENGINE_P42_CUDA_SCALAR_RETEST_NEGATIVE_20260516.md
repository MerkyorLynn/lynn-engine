# Lynn Engine P42 · CUDA scalar retest negative

Date: 2026-05-16

## Summary

P42 retests the tempting `cuda_scalar` native active-MoE backend after P40
cleaned up the default MoE path.

The retest uses the safer full-attention-only allowlist:

```bash
LYNN_NATIVE_ACTIVE_MOE_BACKEND=cuda_scalar
LYNN_NATIVE_ACTIVE_MOE_LAYERS=full_attention
LYNN_LINEAR_BLOCK_GRAPH=1
LYNN_MOE_FAST_FIXED=0
```

This avoids capturing `cuda_scalar` inside the reusable linear-attention block
graphs. It still fails full-generate parity.

## Evidence

Report:

```text
reports/p16_155/p42_cuda_scalar_full_attention_retest.json
```

| Mode | Median Decode TPS | Greedy IDs |
|---|---:|---|
| Triton reference | 99.74 | reference |
| CUDA scalar full-attn allowlist | 99.43 | mismatch |

```text
pass = false
cuda_scalar prompt matches = 0/3
```

One prompt also regressed badly to 48.50 TPS, pulling cuda-scalar mean down to
82.49 TPS.

## Decision

Keep `cuda_scalar` as a diagnostic contract backend only. Do not use it as a
production bridge, even with the full-attention-only allowlist.

P31 remains valuable because it proved the native extension contract. But P32,
P34, and now P42 agree on the production decision: scalar CUDA is not a safe
runtime speed path. Future native work must replace the scalar inner loops with
a genuinely exact grouped FP4 kernel, not promote the scalar bridge.
