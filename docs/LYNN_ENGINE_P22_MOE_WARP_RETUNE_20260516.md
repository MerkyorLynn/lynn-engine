# Lynn Engine P22 — MoE Kernel Warp Retune (2026-05-16)

P22 makes Triton `num_warps` configurable for the packed NVFP4 active MoE
gate/up and down kernels, then sweeps R6000.

## Best Config

```text
LYNN_MOE_GATE_NUM_WARPS=4
LYNN_MOE_DOWN_NUM_WARPS=8
```

The gate/up kernel stays at 4 warps; the down projection benefits from 8 warps.

## Full Graph Result

R6000, 27B NVFP4 step5000, P19/P20/P21 defaults:

| Path | P21 | P22 |
|---|---:|---:|
| strict full graph | 117.71 TPS | **118.25 TPS** |
| replay-only graph | 122.71 TPS | **123.25 TPS** |

This is another small but quality-safe scheduling gain. The remaining gap to
155 TPS is still active expert math/scale handling, not router sorting, shared
expert launch overhead, or generic block/warp configuration.
