# Lynn Engine P21 — BF16 Shared Expert Gate/Up Fusion (2026-05-16)

The shared expert stays BF16 on R6000 because packed shared-expert paths remain
slower. P21 fuses only the BF16 shared expert gate/up projection:

```text
old: gate = linear(x, gate_w); up = linear(x, up_w)
new: gate_up = linear(x, cat(gate_w, up_w)); gate, up = chunk(gate_up)
```

This preserves exact BF16 math and removes one small GEMM launch.

## Micro Probe

Representative layers:

| Layer | Unfused | Fused | Parity |
|---:|---:|---:|---|
| 0 | 0.0600 ms | 0.0546 ms | max_abs 0 |
| 14 | 0.0597 ms | 0.0544 ms | max_abs 0 |
| 28 | 0.0597 ms | 0.0545 ms | max_abs 0 |
| 39 | 0.0600 ms | 0.0556 ms | cosine 0.99999994 |

## Full Graph Result

With P19 + P20 + P21:

| Path | Before P21 | After P21 |
|---|---:|---:|
| strict full graph | 117.55 TPS | **117.71 TPS** |
| replay-only graph | 122.43 TPS | **122.71 TPS** |

The full-graph gain is small but safe. P21 confirms that shared BF16 launch
overhead is no longer the main blocker; active routed experts remain dominant.
