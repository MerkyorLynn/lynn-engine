# Lynn Engine P37 · MoE block-retune line closed

Date: 2026-05-16

## Summary

P37 checks whether another production-safe Triton block-size retune can move
the R6000 27B NVFP4 path beyond the current ~100 TPS serving plateau.

The answer is no. The isolated layer-28 segment profile still shows a tiny
block-size candidate, but the full autoregressive generate gate rejects it:
the candidate is slower and changes greedy token IDs.

## Evidence

Reports:

```text
reports/p16_155/p37_segment_layer28_current.json
reports/p16_155/p37_down_hidden16_generate_gate.json
```

Layer 28 segment profile:

| Segment | Latency |
|---|---:|
| Router top-k | 0.036448 ms |
| Active routed packed NVFP4 experts | 0.107478 ms |
| Shared BF16 expert | 0.059848 ms |
| Current full MoE | 0.188140 ms |

Best isolated active-block candidate:

```text
gate_block_inter=8
gate_block_hidden=256
down_block_hidden=16
down_block_inter=512
isolated active latency=0.106674 ms
diff_vs_default=max_abs 0.0 / cosine 1.0
```

Full generate gate for `LYNN_MOE_DOWN_BLOCK_HIDDEN=16`:

| Mode | Median Decode TPS | Token IDs |
|---|---:|---|
| Current default | 100.25 | reference |
| Candidate | 94.94 | mismatch |

```text
new_ids_all_match = false
median_speedup    = 0.9471x
promote_default   = false
```

All three smoke prompts drifted. The first mismatches happened at token 30,
token 5, and token 7 respectively, so this is not a harmless formatting tail.

## Decision

Do not promote the `down_block_hidden=16` retune. Close the "more block-size
sweeps will get us to 155 TPS" line unless a future hardware-specific profile
shows a materially different bottleneck.

The current MoE wall is real:

- Active routed packed NVFP4 experts: ~0.107 ms/layer.
- Shared BF16 expert: ~0.060 ms/layer.
- Router top-k: ~0.036 ms/layer.

That means the remaining 100 -> 155 TPS gap is not another env-var retune. It
requires a new grouped native-FP4 active expert kernel and/or a graph-owned
serving path that remains strict-parity safe.

## Next

P38 should stop looking for broad Triton grid rearrangements and instead focus
on one of these two tracks:

- Native grouped FP4 expert kernel: replace the inner active expert math while
  preserving the P30/P31 contract.
- Strict graph-owned full-token serving: use the P35 sorted-router contract,
  but do not promote until continuous graph-owned replay matches greedy IDs.
