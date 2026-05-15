# Lynn Engine P17 — Triton `dot_scaled` FP4 Feasibility (2026-05-16)

P17 starts the path after P16 proved that the active routed experts are the
real blocker for 155 TPS.

The core question:

> Can Triton express the native FP4 dot shape we need for grouped active MoE, or
> do we need to jump directly to CUTLASS / custom CUDA?

## Result

Triton 3.6 on R6000 exposes:

```python
tl.dot_scaled(lhs, lhs_scale, "e2m1", rhs, rhs_scale, "e2m1", ...)
```

The raw packed-FP4 layout probe passes.

| Shape | Result |
|---|---|
| synthetic small `1x64 @ 64x16` | max_abs 0, cosine ≈ 1 |
| synthetic gate/up shape `1x2048 @ 2048x8192` | max_abs 0, cosine ≈ 1 |
| raw gate/up dot latency | **0.0125 ms** |

The large shape requires K-packed tiling. A single program with
`K_PACKED=1024,N=8192` exceeds Triton's max tensor elements / shared-memory
limits. With `BLOCK_K_PACKED=256` and `BLOCK_N=64`, the same raw dot is both
correct and fast.

## E8M0 Scale Contract Probe

The follow-up scale probe confirms the `tl.dot_scaled` scale layout and the
neutral e8m0 byte:

| Probe | Result |
|---|---|
| `rhs_scale` layout | **`[N, K//32]`** works |
| transposed `rhs_scale` layout | rejected with shape error |
| neutral e8m0 byte when both sides use the same byte | **127** |
| byte +1 on both lhs/rhs scales | output scales by **4x** |

For a raw all-ones `K=64` dot, expected output is 64. The probe measured:

| scale byte | output |
|---:|---:|
| 124 | 1 |
| 126 | 16 |
| **127** | **64** |
| 128 | 256 |
| 129 | 1024 |

This strongly suggests each operand scale behaves like a power-of-two e8m0
factor around byte 127. When both lhs and rhs use the same byte, incrementing
the byte by one doubles each operand and therefore quadruples the dot result.

## Meaning

This is an important positive signal:

- Blackwell tensor-core FP4 math is easily fast enough for 155 TPS.
- Triton can express raw E2M1 packed dot correctly.
- The current 107 TPS ceiling is not a hardware compute ceiling.

But it is not the full solution yet.

## Remaining Scale Problem

The Lynn-native NVFP4 artifact uses per-16 FP8/e4m3-style scales compatible with
the current `torch._scaled_mm` path.

Triton `tl.dot_scaled` is documented around microscaling/e8m0 semantics:

```text
lhs_scale: [M, K//group_size], group_size 32 for e8m0
rhs_scale: [N, K//group_size], do not transpose rhs_scale
neutral scale byte: 127 in the two-sided synthetic probe
```

That differs from Lynn's current scale contract:

```text
weight_scale: [out, K//16], float/e4m3-ish effective scale
activation scale: [M, K//16]
```

Therefore P17 does **not** claim production native active experts yet. It proves
the dot layout and speed, then narrows the next hard problem to scale handling.

## Next Kernel Task

The next P17/P18 task is a true grouped active expert kernel:

1. Accept selected expert ids and routing weights.
2. Use `tl.dot_scaled` for gate/up with K tiling.
3. Solve scale conversion:
   - either convert Lynn per-16 scales to a Triton-compatible scale layout,
   - or regenerate a second e8m0/microscaling artifact for engine-native use,
   - or fall back to a custom CUDA/CUTLASS path that accepts the existing scale
     contract.
4. Avoid dynamic selected-row `scale_b` construction in the hot path.
5. Extend from gate/up to down without dense cross-expert overcompute.

## Why This Matters

P16 showed:

| Path | Replay TPS |
|---|---:|
| current correct path | 107.13 |
| skip active routed experts | 173.84 |
| skip active + shared | 208.78 |

P17 now shows the raw native FP4 dot for gate/up shape is only **0.0125 ms**.
That is the first concrete evidence that a real grouped native FP4 active
expert path can close the gap, provided scale handling is solved correctly.
