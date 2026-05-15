# Lynn Engine P18 — Scale Contract Decision (2026-05-16)

P17 proved that Triton `tl.dot_scaled(e2m1)` can run the raw FP4 gate/up shape
fast enough for the 155 TPS target. P18 tests whether we can connect that fast
path to Lynn's current NVFP4 artifact without changing the quantization format.

## Tested Paths

All probes were run on R6000 against the real 27B NVFP4 step5000 artifact,
layer 28, top-k=8 selected experts.

| Probe | Idea | Raw latency | Quality |
|---|---|---:|---|
| **P18-A fold per-16→group32** | Reuse current packed codes, fold each pair of per-16 scales to one e8m0 group32 scale | **0.0176-0.0183 ms** | **FAIL** best inter cosine 0.894 |
| **P18-B re-quant BF16→e8m0/group32** | Re-quant selected BF16 rows into an engine-native e8m0/group32 shape | **0.0184 ms** | **FAIL** inter cosine 0.980 |
| **P18-C padded per-16** | Expand each per-16 group into a padded group32 group to avoid pair folding | **0.0180 ms** | **FAIL** inter cosine 0.936 |

Baseline current scalar bridge raw gate/up latency for the same selected rows:

```text
scalar_bridge_raw_ms: ~0.050 ms
```

So the native dot is about **2.7-2.9x faster** for raw gate/up, but the tested
scale bridges are not numerically safe.

## Decision

The current Lynn variable NVFP4 artifact uses per-16 floating/e4m3-style scales.
Triton `tl.dot_scaled` on Blackwell expects e8m0 microscaling with group size 32.

The scale mismatch is now proven to be the blocker:

- the FP4 tensor-core dot shape is fast,
- current per-16 scale folding is not accurate enough,
- power-of-two e8m0/group32 re-quantization is still too lossy for the active
  MoE gate/up nonlinearity,
- padding per-16 into group32 removes folding error but still suffers from
  e8m0 scale rounding.

## Path Forward

155 TPS is still physically plausible because P16 showed:

| Path | Replay TPS |
|---|---:|
| current correct path | 107.13 |
| skip active routed experts | 173.84 |
| skip active + shared | 208.78 |

But the production route is **not** a simple `dot_scaled` flag flip.

The next viable paths are:

1. **Custom CUDA/CUTLASS grouped FP4 kernel** that keeps Lynn's per-16 scale
   contract instead of forcing e8m0/group32.
2. **New engine-native quant artifact** with a better microscaling format, but
   only if full-layer / V8 / V9 retention proves acceptable. The P18-B
   selected-row result says naive e8m0/group32 is not enough.
3. **Continue optimizing the current per-16 scalar/Triton active path** while
   the custom kernel is built: reduce launch overhead, fuse active gate/up/down
   scheduling where safe, and avoid quality-changing approximations.

## Guardrail

Do **not** ship P18-A/B/C as production kernels. They are useful negative
evidence and speed probes, not quality-safe execution paths.
