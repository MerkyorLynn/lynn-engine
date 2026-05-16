# P90 Split16 Gate/Up Kernel Probe

Date: 2026-05-16

P90 is the first real SM120a native FP4 gate/up kernel probe that consumes the
current Lynn-native NVFP4 artifact directly.

P89 proved one K=32 tile. P90 stretches that into a useful gate/up row tile:

- one real routed expert from Lynn 27B;
- eight consecutive gate rows and eight consecutive up rows;
- full hidden dimension K=2048;
- current Lynn-native activation and weight per-16 FP32 scales;
- SM120a blockscaled FP4 MMA with split16 accumulation.

This is still a contract probe, not a production runtime path. Its job is to
prove the current artifact is valid input for the native kernel line.

## Result

Report:

```text
reports/p16_155/p90_sm120a_split16_gateup_kernel_probe.json
```

Ground truth:

| Field | Value |
|---|---:|
| Layer | 28 |
| Expert id | 116 |
| Top-k experts | `[116,159,49,204,12,192,63,155]` |
| Rows | 8 gate + 8 up |
| K | 2048 |
| Median kernel probe | 0.0621ms |
| Mean kernel probe | 0.0662ms |

Accuracy:

| Metric | Value |
|---|---:|
| max_abs_err | `2.38e-07` |
| mean_abs_err | `1.01e-07` |
| rel_l2 | `1.53e-07` |
| tolerance gate | PASS at `1e-5` |

The remaining error is FP32 accumulation/order tolerance. It is far below the
drift levels that caused greedy decode failures in earlier scalar/tile bridges.

## Decision

The Lynn-native route is now more than a tile proof:

- Current Lynn-native per-16 artifact can feed a real full-K gate/up native FP4
  row tile.
- No official/vendor re-quantization is needed before continuing runtime kernel
  construction.
- P91 should expand the row tile and reduce atomics/launch overhead.
- The production promotion gate remains strict full-generate parity, not just
  tile-level tolerance.

The next engineering target is to turn this exact split16 row tile into a
larger active expert gate/up kernel, then wire it behind an opt-in backend.
