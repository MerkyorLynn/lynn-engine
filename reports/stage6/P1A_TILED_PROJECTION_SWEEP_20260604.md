# Stage 6 Phase 1-A follow-up — tiled packed-NVFP4 projection sweep

**Date:** 2026-06-04  
**Host:** Spark GB10 (`dgx-via-n5`)  
**Model:** `/home/merkyor/models/Qwen3.6-35B-A3B-lynn-native-w4a16-nvfp4-from-bf16-20260526`  
**Projection:** `model.language_model.layers.0.linear_attn.in_proj_qkv.weight`  
**Runner:** `scripts/spark_stage6_p1a_batched_projection_poc.py`  
**Remote run dir:** `/home/merkyor/lynn-engine/reports/stage6/p1a_tiled_projection_sweep_20260604_002348`

## Verdict

**P1-A tiled scalar-dequant bridge is NOT promoted.**

The tiled kernel is correct and memory-clean, and it is a real improvement over
the naive P1-A row bridge: up to **25.93x** faster than the naive packed kernel.
But it still loses to Spark's BF16 `F.linear` tensor-core path for M>1. The best
observed tiled config is still **0.742x** vs BF16 for M=16 and **0.359x** vs
BF16 for M=64.

This closes the scalar-dequant dense-prefill bridge on Spark. Dense M>1 prefill
needs either a native FP4-MMA/CUTLASS-style path or should be kept as a
correctness probe while Stage 6 moves to grouped MoE prefill.

## Command

The sweep ran nine tile shapes:

```bash
for bt in 8 16 32; do
  for bm in 16 32 64; do
    docker run --rm --gpus all --ipc=host \
      -e PYTHONNOUSERSITE=1 -e PYTHONUNBUFFERED=1 \
      -v /home/merkyor:/home/merkyor \
      -w /home/merkyor/lynn-engine \
      lynn-eval-base:cu13 \
      python3 -u scripts/spark_stage6_p1a_batched_projection_poc.py \
        --batches 16,64 \
        --block-t "$bt" --block-m "$bm" --block-n 128 \
        --warmup 8 --iters 25 --repeats 2 \
        --json-out "reports/stage6/p1a_tiled_projection_sweep_20260604_002348/bt${bt}_bm${bm}_bn128.json"
  done
done
```

## Numeric And Memory Gates

All nine configs passed the numeric gate against FP32 dequant and the no-shadow
gate.

Representative best-shape numeric (`BLOCK_T=32`, `BLOCK_M=16`, `BLOCK_N=128`):

| batch | cosine vs FP32 dequant | rel L2 | argmax |
|---:|---:|---:|---|
| 16 | 0.999999979 | 2.571e-04 | match |
| 64 | 0.999999979 | 2.583e-04 | match |

Representative memory after deleting BF16/FP32 references:

| metric | value |
|---|---:|
| memory before packed bench | 0.0180 GiB |
| memory after packed bench | 0.0180 GiB |
| peak during packed bench | 0.0200 GiB |
| single projection BF16 shadow | 0.03125 GiB |

## Sweep Results

Sorted by M=16 speedup vs BF16:

| BLOCK_T | BLOCK_M | M=16 tiled | M=16 BF16 | M=16 vs BF16 | M=16 vs naive | M=64 tiled | M=64 BF16 | M=64 vs BF16 | M=64 vs naive |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 16 | 209.11 us | 156.60 us | 0.749x | 11.31x | 806.03 us | 152.34 us | 0.189x | 11.69x |
| 32 | 16 | 211.61 us | 157.01 us | 0.742x | 11.16x | 421.43 us | 151.17 us | 0.359x | 22.36x |
| 32 | 64 | 364.33 us | 161.08 us | 0.442x | 12.60x | 755.91 us | 151.37 us | 0.200x | 24.08x |
| 16 | 32 | 361.94 us | 152.12 us | 0.420x | 12.99x | 1301.14 us | 150.24 us | 0.115x | 14.31x |
| 32 | 32 | 377.28 us | 155.54 us | 0.412x | 12.46x | 718.27 us | 153.53 us | 0.214x | 25.93x |
| 8 | 16 | 393.34 us | 157.19 us | 0.400x | 6.01x | 1526.61 us | 150.83 us | 0.099x | 6.17x |
| 16 | 64 | 387.83 us | 152.97 us | 0.394x | 11.84x | 1356.96 us | 152.02 us | 0.112x | 13.41x |
| 8 | 32 | 648.28 us | 167.47 us | 0.258x | 7.25x | 2515.67 us | 151.81 us | 0.060x | 7.40x |
| 8 | 64 | 748.35 us | 159.37 us | 0.213x | 6.13x | 2689.25 us | 151.40 us | 0.056x | 6.77x |

## Decision

Keep `nvfp4_tiled_batched_matmul_packed` as a correctness/regression probe, but
do not wire it into `resident_runner` and do not count it as a serving win.

Stage 6 should stop spending Spark cycles on scalar-dequant dense M>1 bridges.
Next useful work:

| path | why |
|---|---|
| P2 grouped MoE prefill | MoE dominates the reload/prefill cost and needs a grouped M>1 kernel anyway. |
| Native FP4-MMA/CUTLASS bridge | Required if dense M>1 packed prefill must beat BF16 tensor-core GEMM. |
| M=1 decode fusion | Still relevant for launch-cut / low-dispatch runtime work, but it is not the packed-prefill bottleneck. |
