# Lynn Engine P48: tile-hidden non-atomic down projection

Date: 2026-05-16
Branch: `codex/p16-r6000-155-tps`

## Goal

P46 showed that a fused active-MoE kernel with fp32 atomics is the wrong
production shape. P48 tried the opposite direction: keep the down projection
non-atomic and reduce native CUDA block overhead by computing multiple hidden
rows per CTA.

The first P48 kernel is intentionally narrow:

```text
inter[top_k, 512] + expert_ids + routing_weights + packed down weights
  -> hidden[2048]
```

Gate/up remains the current Triton production kernel. This isolates the down
projection and answers whether a non-atomic native building block can beat the
existing Triton down segment.

## Isolated kernel result

Report: `reports/p16_155/p48_down_tile_nonatomic_probe.json`

Sampled layers: `2, 8, 14, 20, 28, 36`

| Path | Mean latency |
|---|---:|
| Triton down | 0.025865 ms |
| CUDA scalar down | 0.030855 ms |
| CUDA tile down, tile=1 | 0.020676 ms |
| CUDA tile down, tile=2 | **0.020674 ms** |
| CUDA tile down, tile=4 | 0.020935 ms |
| CUDA tile down, tile=8 | 0.032918 ms |

Best isolated speedup:

```text
tile=2 vs Triton down: 1.251x
tile=2 vs CUDA scalar: 1.493x
min cosine vs Triton: 0.99999988
max rel_l2 vs Triton: 2.68e-05
```

This is the first positive kernel-level P43-P48 result after several negative
routes. It confirms that non-atomic native kernels can win on a real active-MoE
subsegment.

## Runtime gate result

The same kernel was wired behind:

```bash
LYNN_NATIVE_DOWN_BACKEND=cuda_tile
LYNN_NATIVE_DOWN_TILE_HIDDEN=2
```

It stayed opt-in and default-disabled.

Runtime reports:

- `reports/p16_155/p48_cuda_tile_runtime_gate.json`
- `reports/p16_155/p48_cuda_tile1_runtime_gate.json`
- `reports/p16_155/p48_cuda_tile4_runtime_gate.json`
- `reports/p16_155/p48_layer_2_runtime_gate.json`
- `reports/p16_155/p48_layer_8_runtime_gate.json`
- `reports/p16_155/p48_layer_14_runtime_gate.json`
- `reports/p16_155/p48_layer_20_runtime_gate.json`
- `reports/p16_155/p48_layer_28_runtime_gate.json`
- `reports/p16_155/p48_layer_36_runtime_gate.json`
- `reports/p16_155/p48_layer_full_runtime_gate.json`
- `reports/p16_155/p48_layer_linear_runtime_gate.json`

Observed runtime behavior:

| Runtime candidate | Greedy exact match | Median TPS behavior |
|---|---:|---:|
| tile=1, all layers | no | ~1.095x |
| tile=2, all layers | no | ~1.087x |
| tile=4, all layers | no | ~1.085x |
| tile=2, layer 2 only | no | ~0.990x |
| tile=2, layer 8 only | no | ~1.002x |
| tile=2, layer 14 only | no | ~1.009x |
| tile=2, layer 20 only | no | ~1.018x |
| tile=2, layer 28 only | no | ~1.000x |
| tile=2, layer 36 only | no | ~1.014x |
| tile=2, full-attn layers | no | ~1.004x |
| tile=2, linear-attn layers | no | ~1.097x |

Some all-layer / linear-layer runs showed a `!` loop. Single-layer tests avoid
the worst collapse for some prompts, but still fail exact greedy parity.

## P49 true decode-state probe

P49 moved the same comparison from a prefill-derived microbench state to the
actual first incremental-decode MoE input.

Report: `reports/p16_155/p49_decode_state_down_tile_probe.json`

| Metric | Value |
|---|---:|
| Mean Triton down | 0.026208 ms |
| Mean CUDA tile down | 0.020671 ms |
| Mean speedup | **1.268x** |
| Max rel_l2 vs Triton | 8.74e-05 |
| Min cosine vs Triton | 0.99999988 |
| Decode-state parity | pass |

This rules out the simple hypothesis that P48 only worked on the wrong hidden
distribution. On the true decode-state MoE input, the tile-hidden kernel still
looks like a clean local replacement.

## P50 first-divergence probe

P50 then compared complete decode steps with CUDA graphs disabled:

```text
baseline:  LYNN_NATIVE_DOWN_BACKEND=triton
candidate: LYNN_NATIVE_DOWN_BACKEND=cuda_tile
```

Both paths consumed the same Triton/reference greedy token stream.

Report: `reports/p16_155/p50_down_tile_first_divergence_linear.json`

Result:

```text
pass: false
first top-1 divergence: step 5
first visible layer divergence: step 1, layer 27
layer 27 rel_l2: 0.0023449
layer 27 cosine: 0.9999972
triton top1 margin at divergence: 0.03125
```

The important lesson is subtle but decisive: local down parity is not enough.
The candidate changes BF16/FP32 accumulation order slightly. The resulting tiny
differences survive through later layers and can flip low-margin greedy choices
after a handful of decode steps, even without CUDA graph replay.

## Decision

Do not promote P48 to the production default.

The isolated and true decode-state kernel wins are real, but the complete
decode loop fails the current greedy exact-match gate. This is exactly why
P37-style runtime gates exist: microbench parity is necessary but not
sufficient for decode-loop promotion.

P48 should remain as an opt-in research backend. The next exact-match line
should either:

- preserve Triton's accumulation order closely enough to pass P50, or
- move to a larger native grouped active-MoE kernel and validate quality with
  explicit V8/V9/tool-call gates rather than pretending it is exact-greedy.

Promotion rule remains strict for the production default: no runtime default
until greedy IDs match across representative prompts.
