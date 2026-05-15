# Lynn Engine P23 — Active MoE Accounting and Safe Micro-Optimization (2026-05-16)

P22 moved the R6000 27B NVFP4 graph ceiling to the 123 TPS class. P23 asks a
narrow question before attempting a larger custom kernel:

> Is there still a quality-safe local optimization inside the existing active
> MoE path, or is the remaining 155 TPS gap truly a new-kernel problem?

## Baseline

Model:

```text
/root/autodl-tmp/models/lynn-27b-variable-recovery-step5000-nvfp4-final
```

P22 R6000 baseline:

| Path | TPS |
|---|---:|
| strict full graph | 118.25 |
| replay-only graph | 123.25 |

## P23-A: 40-Layer Active MoE Accounting

`benchmarks/p23_active_moe_layer_sweep.py` loads the resident model once and
times router, active routed experts, shared expert, and full MoE across all 40
layers.

Totals:

| Segment | Total latency |
|---|---:|
| router + top-k + softmax | 2.061 ms |
| packed NVFP4 gate/up | 1.433 ms |
| packed NVFP4 down | 1.400 ms |
| active routed experts end-to-end | 2.773 ms |
| BF16 shared expert | 2.302 ms |
| full MoE path | 8.804 ms |

There is no obvious pathological layer. Active routed experts are almost flat
across the stack:

```text
active MoE per layer ~= 0.069 ms
router per layer     ~= 0.049 ms
shared per layer     ~= 0.057 ms
```

This rules out a cheap "fix two bad layers" route. The remaining cost is broad
and structural.

## P23-B: Router Split

Layer 28 router split:

| Segment | Latency |
|---|---:|
| BF16 router linear | 0.0089 ms |
| `torch.topk(sorted=False)` | 0.0249 ms |
| small-vector softmax | 0.0106 ms |
| full router path | 0.0457 ms |

The router matrix multiply is not the issue. The cost is the small-vector
top-k/softmax orchestration.

Two tempting alternatives were checked:

| Attempt | Result |
|---|---|
| 1D top-k instead of `[1, experts]` 2D top-k | no improvement; full router was slightly slower |
| manual softmax | much slower because it creates multiple pointwise launches |

## P23-C: Triton Top-k + Softmax Probe

`triton_kernels/router_topk.py` adds a one-kernel Triton top-k + softmax probe.
It is numerically close on layer 28:

```text
top-k same set: true
max weight abs diff on common experts: 2.23e-8
```

But it is slower than the PyTorch path:

| Router path | Latency |
|---|---:|
| torch router in probe | 0.0668 ms |
| Triton cached-logits top-k+softmax | 0.0589 ms |
| Triton full router | 0.0751 ms |

Conclusion: do **not** promote the custom Triton router today. It proves the
shape is feasible, but it does not beat the production route.

## P23-D: Safe Expert-ID Cast Removal

Production previously let `torch.topk` return int64 expert ids, then both
gate/up and down wrappers converted those ids to int32 for Triton. P23 changes
the production path to cast once immediately after top-k:

```python
expert_ids = expert_indices[0].to(torch.int32).contiguous()
```

The Triton kernels already consume int32 expert ids, so this is a pure
orchestration cleanup. It does not change routing, quantization, or arithmetic.

R6000 long-run result:

| Path | P22 | P23-D |
|---|---:|---:|
| strict full graph | 118.25 TPS | **118.73 TPS** |
| replay-only graph | 123.25 TPS | **123.78 TPS** |

This is a small but real quality-safe gain.

## Decision

P23 closes three tempting branches:

1. There is no per-layer hotspot to retune.
2. 1D top-k and manual softmax are not useful.
3. A naive Triton fused top-k+softmax does not beat PyTorch.

The only P23 code change promoted to production is the int32 expert-id cleanup.

The 155 TPS target still requires a true custom active routed expert path:

```text
per-16 grouped native-FP4 active expert kernel
```

That kernel must preserve Lynn's per-16 scale contract instead of using the
lossy e8m0/group32 bridges rejected in P18.
