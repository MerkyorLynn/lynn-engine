# Lynn Engine P74: active-MoE inner budget ledger

Date: 2026-05-16
Branch: `codex/p16-r6000-155-tps`

## Why P74

P73 proved that a native-owned scratch boundary is not enough:

```text
P73 active speedup vs Triton: 1.110x
P69 promotion threshold:     1.25x
```

P74 measures the inner budget of the active expert FFN so the next branch
targets the real bottleneck instead of chasing isolated sub-kernel wins.

## Probe

```bash
python benchmarks/p74_active_moe_inner_budget_probe.py \
  --model /root/autodl-tmp/models/lynn-27b-variable-recovery-step5000-nvfp4-final \
  --out reports/p16_155/p74_active_moe_inner_budget_probe.json \
  --layers 2 8 14 20 28 36 \
  --tile-inter 2 \
  --tile-hidden 2 \
  --warmup 4 \
  --iters 30
```

The probe times the same hidden vector, router top-k, and selected experts
through:

- Triton gate/up;
- native tile-inter gate/up;
- Triton down;
- native grouped-per16 down tile;
- P73 native-owned active boundary.

## Result

| Metric | Value |
|---|---:|
| mean Triton gate/up | 0.03181 ms |
| mean Triton down | 0.02606 ms |
| gate/up share of Triton sub-kernels | 54.97% |
| down share of Triton sub-kernels | 45.03% |
| native gate/up tile speedup | 1.044x |
| native down tile speedup | 1.259x |
| P73 active speedup | 1.111x |
| min P73 cosine vs Triton active | 0.99999988 |
| max P73 relative L2 vs Triton active | 4.98e-4 |

The important asymmetry:

```text
down tile is a good local kernel:     1.259x
gate/up tile is only a small local win: 1.044x
full active boundary stays modest:    1.111x
```

## Interpretation

P74 closes the "maybe we just need a better wrapper" theory.

The native-owned active boundary has essentially zero extra overhead relative
to the sum of its native sub-kernels:

```text
mean p73_native_active_ms - mean summed_native_subkernels_ms = -0.00044 ms
```

So the remaining gap is not Python wrapping inside the active-MoE boundary. It
is the inner math / launch shape:

- gate/up is the larger sub-kernel budget at about 55%;
- down has the stronger current native local win at about 1.26x;
- neither scalar tile path is strong enough to clear the P69 1.25x active
  promotion threshold alone.

## Decision

P74 is a measurement milestone, not a runtime candidate.

Next grouped per-16 work should avoid another scalar two-stage wrapper. The
viable branches are:

1. build a stronger gate/up implementation that keeps Lynn's per-16 scale
   contract but replaces scalar accumulation;
2. fuse or persist the proven down-tile win so it does not remain trapped as a
   small sub-kernel gain;
3. test a vendor-compatible second artifact only as a separate layout track,
   never as a post-hoc conversion of the current Lynn-native artifact.

P74 keeps the existing guardrail:

```text
no full-generate/server gate until P69 accepts the active-MoE boundary
```

The next useful implementation should improve the active boundary, not just a
sub-kernel.
