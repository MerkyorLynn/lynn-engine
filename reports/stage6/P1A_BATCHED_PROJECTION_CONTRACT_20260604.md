# Stage 6 Phase 1-A — batched packed projection contract

**Date:** 2026-06-04
**Scope:** M>1 prefill projection for the same real
`model.language_model.layers.0.linear_attn.in_proj_qkv.weight` target banked in
P1.

## Goal

Replace the P0.1 `LYNN_PACKED_PREFILL_SLOW` Python row loop for dense
projections with one packed-NVFP4 kernel launch over multiple prompt tokens.

The first P1-A kernel is intentionally conservative:

- input `x`: BF16 `[tokens, 2048]`
- weight: packed E2M1 `uint8[8192, 1024]`
- scale: checkpoint-native FP16 `float16[8192, 128]`
- global scale: FP32 scalar
- output: FP32 `[tokens, 8192]`

It computes one token row and one output-row block per Triton program, so it
removes Python launch loops but does not yet reuse activation tiles across
tokens. If this is correct but slow, the next optimization is a true
`BLOCK_T x BLOCK_OUT x BLOCK_K` tiled kernel.

## Gate

Run:

```bash
docker run --rm --gpus all --ipc=host \
  -e PYTHONNOUSERSITE=1 -e PYTHONUNBUFFERED=1 \
  -v /home/merkyor:/home/merkyor \
  -w /home/merkyor/lynn-engine \
  lynn-eval-base:cu13 \
  python3 -u scripts/spark_stage6_p1a_batched_projection_poc.py \
    --batches 1,4,16,64 \
    --json-out reports/stage6/p1a_batched_projection_poc_$(date +%Y%m%d_%H%M%S).json
```

Promotion requires all four evidence classes:

| evidence | requirement |
|---|---|
| byte-count | real checkpoint packed/scale bytes reported |
| numeric parity | each batch cosine `>=0.99999`, rel L2 `<=2e-3`, argmax match vs FP32 dequant oracle |
| no hidden BF16 shadow | timed packed path runs after deleting BF16/FP32 references |
| microbench | report packed vs BF16 for every batch; promotion only if relevant batch sizes are not slower |

Do not wire P1-A into `resident_runner` until this gate has Spark evidence.
