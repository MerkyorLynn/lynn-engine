# Lynn Engine P53: Triton retune review

Date: 2026-05-16
Branch: `codex/p16-r6000-155-tps`

## Why P53

An external independent review suggested that the current Triton active-MoE
kernels might contain easy wins:

- replace the nested E2M1 `tl.where` expression with a cheaper decode form;
- hoist per-16 scales instead of loading them for every column.

P53 tests those claims as opt-in probes only.  No default runtime path changes
from P53 unless both speed and full-generate quality gates pass.

## P53 scale-hoist attempt

`benchmarks/p53_triton_scale_hoist_probe.py` adds opt-in scale-hoisted gate/up
and down kernels.  The first implementation unrolled the hidden/intermediate
loops into per-16 groups so each scale is loaded once per group.

Result: the kernel did not finish JIT/execute within a practical probe window
and was killed.  This is itself a useful signal: the naive scale-hoist rewrite
creates a much heavier Triton compile/body shape than the current production
kernel.  It is not a free 15-20 TPS win in this form.

## P53-LUT lightweight decode expression

`benchmarks/p53_lut_fast_decode_probe.py` keeps the exact production kernel
shape and changes only the E2M1 decode expression:

```text
reports/p16_155/p53_lut_fast_decode_probe.json

mean ref gate/up:      0.03314 ms
mean fast gate/up:     0.03862 ms
mean speedup:          0.936x
min cosine:            0.99999994
max rel_l2:            0
```

Per-layer result:

```text
layer 4:   1.077x faster
layer 16:  1.062x faster
layer 28:  1.069x faster
layer 36:  0.536x slower
```

The lightweight expression is numerically exact, but performance is not stable
across representative layers.  It may be useful as an allowlisted compile
variant after more layer coverage, but it is not a global default.

## Decision

The independent-review idea was worth testing, but P53 does **not** support a
"free 15 TPS" conclusion:

- scale-hoist needs a more careful Triton design or should be folded into the
  CUDA/CUTLASS grouped-kernel work;
- E2M1 expression simplification is exact but uneven and loses on one late
  layer badly enough to make the average slower;
- the 155 TPS path remains P52 grouped native-FP4 active expert FFN or an exact
  graph-owned route.

P53 therefore stays as a research branch of probes, not a runtime promotion.
