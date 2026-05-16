# Lynn Engine P55: tile-inter CUDA scalar gate/up probe

Date: 2026-05-16
Branch: `codex/p16-r6000-155-tps`

## Why P55

P48 proved that a non-atomic tiled down projection can win locally, but full
decode still flipped greedy because tiny accumulation-order drift compounds
through 40 layers.

P55 tests the gate/up side with a safer goal:

```text
hidden[2048] + expert_ids[top_k]
  -> inter[top_k,512]
```

The new CUDA scalar variant computes multiple intermediate rows per block
(`tile_inter`) and reuses the hidden-vector load. It is still scalar math, not
the final native FP4 tensor-core kernel. The goal is to identify a useful tile
shape for the real grouped per-16 active expert kernel.

## Implementation

New opt-in extension entry:

```text
gate_up_silu_tile_inter_scalar(..., tile_inter={1,2,4,8})
```

Probe:

```text
benchmarks/p55_gateup_tile_inter_probe.py
reports/p16_155/p55_gateup_tile_inter_probe.json
```

## Result

`tile_inter=2` wins on all four representative layers and stays exact-ish
against the Triton active gate/up reference:

| Layer | Triton ref ms | Best variant | Best ms | Speedup vs Triton | Diff |
|---:|---:|---|---:|---:|---|
| 4 | 0.03473 | tile_inter_2 | 0.03098 | 1.121x | max_abs=0 |
| 16 | 0.03366 | tile_inter_2 | 0.03101 | 1.086x | max_abs=0 |
| 28 | 0.03416 | tile_inter_2 | 0.03092 | 1.105x | max_abs=0 |
| 36 | 0.03407 | tile_inter_2 | 0.02987 | 1.141x | max_abs=0 |

Larger tiles are worse:

```text
tile_inter_4: ~0.72-0.73x vs Triton
tile_inter_8: ~0.40-0.42x vs Triton
```

## Decision

P55 is a positive kernel-shape signal:

- two inter rows per block is the useful scalar tile;
- bigger tiles over-pressure registers/shared memory and lose badly;
- the speedup is local only, around 8.6-14.1%, so it cannot be the 155TPS fix
  alone.

Do not promote this to production by itself yet. P48 taught that local
kernel-level wins need full-generate gates before runtime promotion.

## Next step

Use the `tile_inter=2` shape as the first design point for the real grouped
per-16 active expert kernel:

```text
P56 target:
  grouped gate/up kernel
  tile_inter=2 baseline
  consume Lynn packed E2M1 + FP32 per-16 scales directly
  preserve Triton reference output or pass full task-retention gates
```

If P56 starts with scalar math, it should still treat P55 as an ABI/tile
contract. The actual 155TPS jump requires replacing the scalar inner loop with
true grouped native-FP4 math or an equivalent CUTLASS/CuTe implementation.
