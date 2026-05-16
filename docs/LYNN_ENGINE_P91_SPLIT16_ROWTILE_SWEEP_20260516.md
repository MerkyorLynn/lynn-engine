# P91 Split16 Row-Tile Sweep

Date: 2026-05-16

P91 widens the P90 split16 gate/up kernel from one 8-row tile to multiple
8-row tiles in one launch. It keeps the exact same math:

- current Lynn-native per-16 activation and weight scales;
- two neutral-scale SM120a FP4 MMA passes per K32 group;
- explicit per-group scale accumulation in FP32;
- real Lynn 27B routed expert data.

The goal is to find whether row-level parallelism amortizes launch/atomic
overhead before we attempt a full 512-row active expert gate/up kernel.

## Result

Report:

```text
reports/p16_155/p91_sm120a_split16_gateup_rowtile_sweep.json
```

All variants pass the `1e-5` tolerance gate:

| Row count | Blocks | Median ms | Rows/ms median | max_abs_err | rel_l2 |
|---:|---:|---:|---:|---:|---:|
| 8 | 1 | 0.04568 | 350.3 | 2.38e-07 | 1.53e-07 |
| 16 | 2 | 0.04568 | 700.5 | 2.38e-07 | 1.30e-07 |
| 32 | 4 | 0.04582 | 1396.6 | 2.98e-07 | 1.48e-07 |
| 64 | 8 | **0.04426** | **2892.3** | 3.58e-07 | 1.57e-07 |

P91 is a strong shape signal: 64 rows are not slower than 8 rows. They are
slightly faster in median latency and dramatically better in rows/ms.

## Decision

Use `row_count=64` as the next design point.

P92 should attempt a full gate/up expert shape by launching eight 64-row tiles
for gate and up. The next proof target is:

- exact full 512-intermediate gate/up for one routed expert;
- compare against the existing Triton/scalar reference;
- measure whether the native FP4 kernel can beat the rejected scalar tile
  bridge while preserving full-generate safety before any runtime promotion.

This keeps us on the current Lynn-native artifact route and continues pushing
toward the grouped per-16 active expert FFN without waiting for vendor re-quant.
