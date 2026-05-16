# Lynn Engine P54: vendor-layout scale-search feasibility

Date: 2026-05-16
Branch: `codex/p16-r6000-155-tps`

## Why P54

P52-B showed that the quality cliff in the generic native-FP4 bridge appears
when Lynn's FP32 per-16 weight-scale contract is compressed into the e8m0
`scale_b` layout required by generic `dot_scaled` / vendor-style kernels.

P54 asks whether that conclusion survives a stronger test:

1. `search_recon_mse`: choose the e8m0 exponent per group32 by minimizing
   reconstruction error. This is close to what a deployable second quant
   artifact could do offline.
2. `search_dot_upper_bound`: choose the e8m0 exponent per group32 using the
   current hidden activation to minimize dot-contribution error. This is an
   optimistic upper bound, not a deployable static artifact.

If even the activation-aware upper bound misses the safety gate, a
vendor-friendly e8m0/group32 layout is unlikely to preserve Lynn 27B quality
without retraining / QAT. The practical path is then a Lynn-native grouped
kernel that consumes the existing per-16 scale contract directly.

## Probe

```text
benchmarks/p54_e8m0_scale_search_probe.py

layers:   4,16,28,36
shape:    selected top-k gate/up rows, [8192,2048]
offsets:  e8m0 exponent search over [-4,+4]
gate:     inter cosine > 0.995 and inter rel_l2 < 0.08
```

Reports:

```text
reports/p16_155/p54_e8m0_scale_search_probe_l28.json
reports/p16_155/p54_e8m0_scale_search_probe_4layers.json
```

## Result

Best method on every layer was the optimistic activation-aware
`search_dot_upper_bound`, but every layer still failed:

| Layer | Best inter cosine | Best inter rel_l2 | Pass |
|---:|---:|---:|---|
| 4 | 0.98754 | 0.15756 | ❌ |
| 16 | 0.99136 | 0.13213 | ❌ |
| 28 | 0.98692 | 0.16274 | ❌ |
| 36 | 0.99184 | 0.13038 | ❌ |

The deployable-style `search_recon_mse` path was lower still:

| Layer | `search_recon_mse` inter cosine | inter rel_l2 |
|---:|---:|---:|
| 4 | 0.97903 | 0.20390 |
| 16 | 0.98564 | 0.17070 |
| 28 | 0.97924 | 0.20355 |
| 36 | 0.98801 | 0.16293 |

## Decision

P54 rejects the simple "make a ModelOpt-like e8m0/group32 side artifact and
use a vendor-style kernel" shortcut for the current Lynn 27B artifact.

The important nuance is that NVIDIA's public ModelOpt NVFP4 checkpoints still
validate the destination: Blackwell native FP4 + MoE linear operators is real.
What P54 rejects is the assumption that Lynn's already-quantized per-16 FP32
scale artifact can be converted into that scale contract by offline exponent
search alone.

## Next path

The main line is now narrower and cleaner:

```text
custom grouped native-FP4 active expert FFN
  input:  Lynn packed E2M1 codes + FP32 per-16 scales
  route:  selected top-k experts only
  output: exact/near-exact active expert contribution
  gate:   full-generate parity or V8/V9/tool-call retention
```

P54 also gives us the boundary for a future compatibility track:

- a vendor-friendly artifact may still be possible if quantized from BF16 with
  calibration / QAT / scale-aware recovery;
- it should be treated as a separate model artifact, not a direct conversion of
  the current Lynn-native NVFP4 weights;
- it must pass the same V8, V9, tool-call, no-think, and long-context gates.

## Practical implication

Do not spend more overnight cycles trying to tune e8m0/group32 bridge scales
for the existing artifact. The next meaningful engineering work is the per-16
grouped kernel, starting from the active expert FFN contract defined in P52.
