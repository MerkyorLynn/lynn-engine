# Lynn Engine P75: gate/up tile thread sweep closed

Date: 2026-05-16
Branch: `codex/p16-r6000-155-tps`

## Why P75

P74 measured the active-MoE inner budget:

```text
gate/up share: ~55%
down share:    ~45%
```

The current native gate/up tile-inter shape used 128 CUDA threads. P75 asks
whether the scalar gate/up branch still has easy launch-shape headroom before
we move to the real grouped per-16 kernel.

## Probe

```bash
python benchmarks/p75_gateup_tile_threads_probe.py \
  --model /root/autodl-tmp/models/lynn-27b-variable-recovery-step5000-nvfp4-final \
  --out reports/p16_155/p75_gateup_tile_threads_probe.json \
  --layers 2 8 14 20 28 36 \
  --tile-inters 1,2 \
  --threads 64,128,256 \
  --warmup 4 \
  --iters 30
```

P75 adds a probe-only native entry point:

```text
gate_up_silu_tile_inter_threads_scalar(..., tile_inter, threads)
```

It does not change the production runtime path.

## Result

| Variant | Mean speedup vs Triton fast gate/up | Min speedup | Decision |
|---|---:|---:|---|
| `tile_inter=1, threads=64` | 0.700x | 0.694x | reject |
| `tile_inter=1, threads=128` | 1.004x | 0.979x | no gain |
| `tile_inter=1, threads=256` | 0.944x | 0.916x | reject |
| `tile_inter=2, threads=64` | 0.720x | 0.709x | reject |
| **`tile_inter=2, threads=128`** | **1.032x** | **1.029x** | best, small |
| `tile_inter=2, threads=256` | 1.028x | 1.022x | small |

Numerics for the best variant:

```text
min cosine vs Triton: 0.99999988
max relative L2:      1.32e-4
```

## Decision

P75 closes scalar gate/up launch-shape tuning.

The best safe-looking scalar variant is still the existing P55 shape:

```text
tile_inter=2, threads=128
```

But on the stricter P74 six-layer set it is only **1.03x** faster than Triton
fast gate/up. That is not enough to move the active-MoE boundary toward 155
TPS. More scalar gate/up sweeps are unlikely to pay for themselves.

## Next path

The next implementation branch should not be another scalar tile sweep. It
should be one of:

1. a true grouped per-16 gate/up kernel using CuTe/CUTLASS/native FP4 math while
   preserving Lynn's FP32 per-16 scale contract; or
2. a fused/persistent active-MoE schedule that turns P67's down-tile win into
   active-boundary speed without changing accumulation enough to fail
   full-generate gates; or
3. a separate vendor-compatible NVFP4 v2 artifact, guarded by P59 layout
   dispatch and V8/V9/tool/no-think/long-context retention.

P75 confirms that the scalar bridge is useful as a correctness scaffold, not
as the final 155 TPS route.
