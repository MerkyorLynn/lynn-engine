# Lynn Engine P26 — Merged-TopK Gate/Up Negative Result (2026-05-16)

P24 ruled out the half-native `dequant -> tl.dot` bridge. P26 tested another
Triton-only idea that keeps the current scalar per-16 math unchanged:

> Instead of launching one program per `(expert_slot, inter_block)`, launch one
> program per `inter_block` and loop over all active top-k experts inside the
> program.

The goal was to see whether reducing program-grid count could recover scheduling
overhead without changing numeric behavior.

## Probe

`benchmarks/p26_gateup_merged_topk_probe.py` compares the production packed
gate/up kernel with `nvfp4_grouped_gate_up_silu_merged_topk` across four
representative layers.

Model:

```text
/root/autodl-tmp/models/lynn-27b-variable-recovery-step5000-nvfp4-final
```

Layers:

```text
4, 16, 28, 36
```

Sweep:

```text
BLOCK_INTER  = 4, 8, 16
BLOCK_HIDDEN = 64, 128, 256
num_warps    = 4, 8
```

## Result

Quality is effectively exact, but speed regresses.

| Layer | Production reference | Best merged-top-k | Slowdown | Cosine |
|---:|---:|---:|---:|---:|
| 4 | 0.033524 ms | 0.069402 ms | 2.07x slower | 1.0 |
| 16 | 0.033194 ms | 0.068458 ms | 2.06x slower | 1.0 |
| 28 | 0.033277 ms | 0.069259 ms | 2.08x slower | 1.0 |
| 36 | 0.033212 ms | 0.069274 ms | 2.09x slower | 1.0 |

Best common shape:

```text
BLOCK_INTER=4
BLOCK_HIDDEN=256
num_warps=8
```

## Decision

Do **not** promote P26.

The result tells us the current scalar packed kernel is already close to the
best Triton can do for this math shape. Rearranging the program grid without
changing the underlying compute path is not enough.

Combined with P18 and P24, the boundary is now clear:

| Route | Result |
|---|---|
| e8m0/group32 bridge into `dot_scaled` | fast but quality fails |
| per-16 dequant inside Triton then `tl.dot` | quality passes but slower |
| merged-top-k Triton scheduling | quality passes but slower |

The next 155 TPS attempt should stop treating Triton scheduling as the main
lever and move to a real custom active expert path:

```text
custom per-16 grouped native-FP4 expert kernel
```

That likely means CUDA/CUTLASS or another lower-level Blackwell FP4 path that
can consume Lynn's native per-16 scale contract without first expanding into the
current scalar bridge.

