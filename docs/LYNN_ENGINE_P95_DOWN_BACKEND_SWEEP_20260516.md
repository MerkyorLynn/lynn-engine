# P95 Split16 Down Backend Sweep

Date: 2026-05-16

P95 keeps the P93 native gate/up output fixed and sweeps the down half of the
active expert path:

- production Triton down weighted-sum;
- native CUDA scalar down;
- native CUDA tile-hidden down with `TILE_HIDDEN in {1,2,4,8}`.

This answers whether the next speed work should focus on the down projection or
on a larger fused/persistent active expert kernel.

## Result

Report:

```text
reports/p16_155/p95_sm120a_split16_down_backend_sweep.json
```

All variants pass the quantized-activation active-MoE reference contract:

| Variant | Median ms | Speedup vs Triton | Contract |
|---|---:|---:|---|
| Triton down | 0.0521 | 1.00x | PASS |
| Native scalar down | 0.0347 | 1.50x | PASS |
| **Native tile1 down** | **0.0243** | **2.14x** | PASS |
| Native tile2 down | 0.0262 | 1.99x | PASS |
| Native tile4 down | 0.0258 | 2.02x | PASS |
| Native tile8 down | 0.0378 | 1.38x | PASS |

Best variant:

```text
native_down_tile1
```

## Decision

P95 shows that the down half has real headroom. The fastest native down backend
is **2.14x faster** than the current Triton down kernel while keeping the same
active-MoE numerical contract.

This changes the next step:

- do not spend the next iteration only on gate/up;
- build P96 as `P93 native gate/up + native_down_tile1`;
- compare that full composition against the current production active-MoE path;
- if it holds, run strict full-generate parity before any runtime promotion.

