# Stage 6 Phase 2-KA — gated-delta native recurrent-loop PoC

Date: 2026-06-04

Verdict: **numeric PASS, speed FAIL; do not promote.**

P2-J showed that `chunk_gated_delta_with_state` is the linear-attention prefill wall
(71-76% of traced wall time across T16..512). P2-KA tested the cheapest possible
native reuse path: call the existing single-token Triton decode recurrent kernel
(`recurrent_gated_delta_fused_prepare_gqa`) in a loop over prefill tokens.

This lower-bound answers one narrow question: can the decode recurrent kernel
match the chunk reference math if reused for prefill? It can. But it launches once
per token, so it is not a viable prefill implementation.

## Commands

Short run:

```bash
docker run --rm --gpus all --ipc=host \
  -e PYTHONNOUSERSITE=1 -e PYTHONUNBUFFERED=1 \
  -v /home/merkyor:/home/merkyor \
  -w /home/merkyor/lynn-engine \
  lynn-eval-base:cu13 \
  python3 scripts/spark_stage6_p2k_gated_delta_native_loop_poc.py \
    --layer 0 \
    --seq-lens 16,64,128 \
    --repeats 1 \
    --json-out /home/merkyor/lynn-engine/reports/stage6/p2ka_gated_delta_native_loop_20260604_023639/result.json
```

Long run:

```bash
docker run --rm --gpus all --ipc=host \
  -e PYTHONNOUSERSITE=1 -e PYTHONUNBUFFERED=1 \
  -v /home/merkyor:/home/merkyor \
  -w /home/merkyor/lynn-engine \
  lynn-eval-base:cu13 \
  python3 scripts/spark_stage6_p2k_gated_delta_native_loop_poc.py \
    --layer 0 \
    --seq-lens 256,512 \
    --repeats 1 \
    --json-out /home/merkyor/lynn-engine/reports/stage6/p2ka_gated_delta_native_loop_long_20260604_023700/result.json
```

Artifacts:

- `reports/stage6/p2ka_gated_delta_native_loop_20260604_023639/result.json`
- `reports/stage6/p2ka_gated_delta_native_loop_20260604_023639/run.log`
- `reports/stage6/p2ka_gated_delta_native_loop_long_20260604_023700/result.json`
- `reports/stage6/p2ka_gated_delta_native_loop_long_20260604_023700/run.log`

## Speed

Ratio is `chunk_reference_ms / native_loop_inplace_ms`; values below 1.0 are regressions.

| Seq len | Chunk reference | Native loop alloc | Native loop inplace | Inplace ratio | Estimated recurrent launches |
|---:|---:|---:|---:|---:|---:|
| 16 | 2.72 ms | 0.84 ms | 0.54 ms | 4.991x | 16 |
| 64 | 2.54 ms | 2.08 ms | 1.96 ms | 1.296x | 64 |
| 128 | 2.46 ms | 4.21 ms | 4.05 ms | 0.608x | 128 |
| 256 | 3.40 ms | 8.90 ms | 8.03 ms | 0.424x | 256 |
| 512 | 4.16 ms | 16.28 ms | 15.62 ms | 0.266x | 512 |

The small T16/T64 wins are fixed-overhead artifacts: the torch chunk reference has
enough constant overhead that a short token loop can look faster. Once T reaches
128+, the one-launch-per-token design dominates and speed falls linearly.

## Numeric

All output/state comparisons pass with cosine > 0.999 and matching argmax.

| Run | Min cosine | Max cosine | Argmax |
|---|---:|---:|---|
| T16/T64/T128 | 0.999989555 | 0.999996625 | match |
| T256/T512 | 0.999993027 | 0.999995884 | match |

The math contract is therefore good enough to use the existing decode recurrent
kernel as a reference oracle for P2-KB.

## Decision

Bank P2-KA as a rejected implementation path:

- **Keep:** existing recurrent decode kernel validates gated-delta recurrence semantics.
- **Reject:** prefill by looping T single-token launches; it regresses to 0.266x by T512.
- **Next:** P2-KB must be a true chunk/block-level gated-delta prefill kernel that processes multiple tokens per launch.
- **Promotion gate remains closed:** no packed-prefill server default until P2-KB/P2-L and RC quality pass.
