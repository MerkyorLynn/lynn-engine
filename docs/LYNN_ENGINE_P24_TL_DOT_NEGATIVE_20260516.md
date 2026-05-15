# Lynn Engine P24 — `tl.dot` Gate/Up Negative Result (2026-05-16)

P23 proved that the remaining 155 TPS gap is not a single bad layer or a router
bookkeeping issue. P24 tested the next tempting shortcut:

> Can we keep Lynn's per-16 NVFP4 scale contract, dequantize inside Triton, and
> use `tl.dot` for the active expert gate/up reduction?

This would be attractive because it looks like a halfway step toward native FP4
without writing a full custom grouped expert kernel.

## Probe

`benchmarks/p24_gateup_tl_dot_probe.py` compares the production packed NVFP4
scalar gate/up kernel with a Triton `tl.dot` candidate on layer 28.

Model:

```text
/root/autodl-tmp/models/lynn-27b-variable-recovery-step5000-nvfp4-final
```

The probe preserves the current artifact contract:

- packed E2M1 nibbles;
- per-16 scales;
- active top-8 routed experts;
- BF16 output parity against the production scalar packed path.

Large tiles hit shared-memory/resource limits, so the final sweep used smaller
tiles:

```text
BLOCK_INTER  = 4, 8, 16
BLOCK_HIDDEN = 32, 64, 128
num_warps    = 4, 8
```

## Result

Quality is good, but speed is not.

| Path | Latency |
|---|---:|
| production scalar packed gate/up | **0.03345 ms** |
| best `tl.dot` candidate | **0.08065 ms** |

Best candidate:

```text
BLOCK_INTER=8
BLOCK_HIDDEN=32
num_warps=4
```

Diff vs scalar reference:

| Metric | Value |
|---|---:|
| cosine | 0.99999964 |
| max_abs | 0.00390625 |
| rel_l2 | 0.000810 |

The candidate is roughly **2.4x slower** than the current production scalar
kernel despite passing the numeric check.

## Decision

Do **not** promote P24.

The result narrows the route to 155 TPS:

1. Dequantizing per-16 packed weights into `tl.dot` is not enough.
2. The overhead/resource shape of the bridge eats the expected tensor-core win.
3. The correct next step is a real custom active expert kernel that consumes the
   Lynn native per-16 packed contract directly.

In other words, the next kernel should not be:

```text
packed NVFP4 -> dequant inside Triton -> fp16 tl.dot
```

It should be:

```text
packed E2M1 + per-16 scale -> grouped native FP4 expert GEMM
```

P24 is still useful because it prevents us from repeatedly retrying the
half-native bridge. The scalar packed kernels remain the production default until
the real grouped native FP4 expert kernel is available.

