# Route Decision: Official Qwen3.6 35B First

Date: 2026-05-17 22:50 CST

## Decision

Promote the official `Qwen/Qwen3.6-35B-A3B` route ahead of the temporary
V4-Pro/V4-Flash distillation route.

The working target is now:

```text
Qwen/Qwen3.6-35B-A3B BF16
  -> Lynn-native W4A16 NVFP4
  -> official Qwen3.6-35B-A3B MTP sidecar
  -> R6000 native serving/MTP probes
```

## Why

The new quality framing makes the win condition clearer:

- Q4_K_M-class quality already beats the older FP8 teacher comparison on MMLU
  and keeps GPQA within a small tradeoff band.
- If Lynn-native NVFP4/W4A16 can match that quality band, it keeps the runtime
  advantages that GGUF cannot provide: Lynn native kernels, native MTP wiring,
  and direct P107/P119 serving probes.
- The official 35B model is the cleanest match for the official 35B MTP sidecar.
  A distilled model can still be a product variant, but it should no longer
  block the core 155 TPS route.

## Immediate R6000 Plan

`scripts/r6000_qwen36_35b_native_w4a16_mtp_pipeline.sh` now owns the R6000
official route:

1. download official BF16 from `Qwen/Qwen3.6-35B-A3B`;
2. validate every safetensors shard before use;
3. pack Lynn-native W4A16 NVFP4;
4. run BF16-vs-W4A16 logit smoke;
5. run W4A16/W4A8 generation matrix;
6. run official MTP shape, forward, iterative accept, and P107 shadow probes.

V4-Pro partial downloads and watchers were stopped and removed on R6000 before
starting this route.
