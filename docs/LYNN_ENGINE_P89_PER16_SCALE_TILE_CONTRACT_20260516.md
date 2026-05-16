# P89 Per-16 Scale Tile Contract

Date: 2026-05-16

P89 answers the immediate route question: can the current Lynn-native NVFP4
artifact be consumed directly by the SM120a native FP4 path, or do we need to
stop and re-quantize into a vendor-friendly e8m0/group32 layout first?

The answer is yes for the current artifact, as long as the kernel preserves
Lynn's per-16 scale contract. The safe shape is a split16 tile:

1. Run one neutral-scale SM120a FP4 MMA for K[0:16].
2. Run one neutral-scale SM120a FP4 MMA for K[16:32].
3. Apply the real activation scale and weight scale for each K16 group outside
   the MMA, then accumulate in FP32.

This keeps the current Lynn-native `nvfp4_e2m1_rowwise_per_16` artifact intact.
It does not require an official/vendor re-quantization pass before we start the
real active expert kernel.

## Ground Truth

Report:

```text
reports/p16_155/p89_sm120a_per16_scale_tile_contract.json
```

Representative real tile:

| Field | Value |
|---|---:|
| Layer | 28 |
| Expert id | 116 |
| Top-k experts | `[116,159,49,204,12,192,63,155]` |
| Row offset | 0 |
| K offset | 0 |
| Build time | 55.39s |
| Kernel probe median | 0.0329ms |

Split16 result:

| Metric | Value |
|---|---:|
| max_abs_err | `1.49e-08` |
| mean_abs_err | `4.63e-09` |
| rel_l2 | `1.00e-07` |
| tolerance gate | PASS at `1e-6` |

The tiny non-zero error is FP32 accumulation/order tolerance, not a layout or
scale-contract error.

## Why Not Fold To One K32 Scale?

P89 also tested several simple K32 scale folding policies. These are not
production-safe:

| Fold policy | rel_l2 |
|---|---:|
| mean | 0.0370 |
| max | 0.1606 |
| min | 0.1294 |
| geom | 0.0342 |
| weighted_abs | 0.0227 |

Even the best simple fold drifts by about 2.3% relative L2 on this small real
tile. That is exactly the class of tiny numerical drift that previously flipped
greedy decode in P48-P50 and P56.

## Decision

Continue the Lynn-native artifact route first:

- P90 should implement the first real per-16 split16 active gate/up kernel.
- The kernel must keep the two K16 scale groups explicit.
- Do not collapse Lynn per-16 FP32 scales into a single K32 e8m0/group32 scale
  inside the production path.
- The official/vendor-friendly NVFP4 v2 route remains valid, but it belongs to
  the next BF16-derived re-quantization cycle, ideally alongside MTP work.

This keeps the fast path aligned with the artifact we already trust, instead
of blocking the runtime on a second quantization format.
