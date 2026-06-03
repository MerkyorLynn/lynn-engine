# Stage 6 Phase 2-L — linear-attn block-kernel integration smoke

Date: 2026-06-04

Verdict: **PASS as an opt-in `prefill_linear_attn` integration smoke; not yet promoted to serving.**

P2-KB passed the gated-delta core-kernel gate. P2-L wires that kernel into
`engine.incremental_decode.prefill_linear_attn()` behind:

```bash
LYNN_LINEAR_ATTN_PREFILL_BLOCK_GQA=1
```

Default behavior is unchanged when the flag is unset.

## Scope

This smoke compares the same `prefill_linear_attn()` layer-level path with the
flag off vs on:

- reference: existing `_chunk_gated_delta_with_state`;
- candidate: `recurrent_gated_delta_block_gqa`;
- unchanged segments: qkv/z/a/b projections, causal conv, RMSNormGated, out_proj;
- checked outputs: final layer output, recurrent state, and conv state.

## Commands

Short run:

```bash
docker run --rm --gpus all --ipc=host \
  -e PYTHONNOUSERSITE=1 -e PYTHONUNBUFFERED=1 \
  -v /home/merkyor:/home/merkyor \
  -w /home/merkyor/lynn-engine \
  lynn-eval-base:cu13 \
  python3 scripts/spark_stage6_p2l_linear_attn_block_integration_smoke.py \
    --layer 0 \
    --seq-lens 16,64,128 \
    --repeats 1 \
    --json-out /home/merkyor/lynn-engine/reports/stage6/p2l_linear_attn_block_integration_20260604_025327/result.json
```

Long run:

```bash
docker run --rm --gpus all --ipc=host \
  -e PYTHONNOUSERSITE=1 -e PYTHONUNBUFFERED=1 \
  -v /home/merkyor:/home/merkyor \
  -w /home/merkyor/lynn-engine \
  lynn-eval-base:cu13 \
  python3 scripts/spark_stage6_p2l_linear_attn_block_integration_smoke.py \
    --layer 0 \
    --seq-lens 256,512 \
    --repeats 1 \
    --json-out /home/merkyor/lynn-engine/reports/stage6/p2l_linear_attn_block_integration_long_20260604_025348/result.json
```

Artifacts:

- `reports/stage6/p2l_linear_attn_block_integration_20260604_025327/result.json`
- `reports/stage6/p2l_linear_attn_block_integration_20260604_025327/run.log`
- `reports/stage6/p2l_linear_attn_block_integration_long_20260604_025348/result.json`
- `reports/stage6/p2l_linear_attn_block_integration_long_20260604_025348/run.log`

## Speed

| Seq len | Reference | Block opt-in | Ratio |
|---:|---:|---:|---:|
| 16 | 3.31 ms | 1.20 ms | 2.760x |
| 64 | 3.19 ms | 1.07 ms | 2.984x |
| 128 | 3.37 ms | 0.90 ms | 3.739x |
| 256 | 4.35 ms | 1.70 ms | 2.560x |
| 512 | 5.58 ms | 2.18 ms | 2.563x |

## Numeric

All output/recurrent-state/conv-state comparisons pass with cosine > 0.999 and
argmax match.

| Run | Min cosine | Max cosine | Max rel_l2 | Argmax |
|---|---:|---:|---:|---|
| T16/T64/T128 | 0.999983974 | 1.000000000 | 0.005845026 | match |
| T256/T512 | 0.999990103 | 1.000000000 | 0.004459088 | match |

The layer output cosine is lower than the core-kernel cosine because the result
has passed through RMSNormGated and out projection, but argmax remains stable.

## Decision

Bank P2-L as an opt-in layer-level integration pass.

Next gate: P2-M should run selected-layer/full-prefill smoke with
`LYNN_LINEAR_ATTN_PREFILL_BLOCK_GQA=1` plus existing P2-E MoE opt-in, then check
full `_prefill_layer` numeric/speed before any server/default promotion.
