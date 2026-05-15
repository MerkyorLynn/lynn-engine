# Lynn Engine P20 — Router Top-K Unsorted (2026-05-16)

P20 removes unnecessary sorting from router top-k.

PyTorch `topk` sorts results by default. For MoE decode the final weighted sum
is order-independent as long as expert ids and routing weights stay paired. The
router does not need sorted experts.

## Micro Probe

R6000, 27B NVFP4 step5000, layer 28:

| Router mode | Latency |
|---|---:|
| `sorted=True` | 0.02640 ms |
| `sorted=False` | **0.02099 ms** |

Speedup: **1.26x** for the router top-k micro-segment.

The selected expert set and paired routing weights are identical after sorting
pairs for comparison:

```text
same_expert_set: true
paired_weight_max_abs_after_sort: 0
```

## MoE Parity

Sorted vs unsorted MoE output was checked on representative layers:

```text
layers: 0,7,14,21,28,35,39
max_abs: 0 on every layer
cosine: >= 0.99999994
```

## Full Graph Result

With P19 block retune plus P20 unsorted router:

| Path | P19 | P20 |
|---|---:|---:|
| strict full graph | 115.41 TPS | **117.55 TPS** |
| replay-only graph | 120.25 TPS | **122.43 TPS** |

This is a small but quality-safe gain. It does not change routing semantics,
expert choice, quantization, or approximation policy.
