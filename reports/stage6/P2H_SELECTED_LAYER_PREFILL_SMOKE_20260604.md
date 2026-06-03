# Stage 6 Phase 2-H — selected-layer full transformer prefill smoke

**Date:** 2026-06-04
**Host:** Spark GB10 (`dgx-spark`)
**Model:** `/home/merkyor/models/Qwen3.6-35B-A3B-lynn-native-w4a16-nvfp4-from-bf16-20260526`
**Runner:** `scripts/spark_stage6_p2h_selected_layer_prefill_smoke.py`

## Verdict

**P2-H passes for selected-layer full transformer prefill.** The P2E
`p2e_hybrid` path is now verified inside `_prefill_layer`, not only in a
standalone MoE harness: RMSNorm, linear/full attention cache population,
residuals, and MoE FFN all run through the engine prefill chain while active
BF16 expert shadows are deleted.

This is still not full tokenized end-to-end all-layer prefill. It is the bridge
gate between P2-G MoE-only smoke and all-layer/server promotion.

## Commands

### Full-attention layer smoke

Remote run dir:
`/home/merkyor/lynn-engine/reports/stage6/p2h_fullattn_layer_smoke_20260604_020519`

```bash
docker run --rm --gpus all --ipc=host \
  -e PYTHONNOUSERSITE=1 -e PYTHONUNBUFFERED=1 \
  -v /home/merkyor:/home/merkyor \
  -w /home/merkyor/lynn-engine \
  lynn-eval-base:cu13 \
  python3 scripts/spark_stage6_p2h_selected_layer_prefill_smoke.py \
    --layers 3 --seq-lens 16,64 \
    --warmup 0 --iters 1 --repeats 1 \
    --json-out reports/stage6/p2h_fullattn_layer_smoke_20260604_020519/result.json
```

### Linear-attention layer smoke

Remote run dir:
`/home/merkyor/lynn-engine/reports/stage6/p2h_linearattn_layer_smoke_20260604_020700`

```bash
docker run --rm --gpus all --ipc=host \
  -e PYTHONNOUSERSITE=1 -e PYTHONUNBUFFERED=1 \
  -v /home/merkyor:/home/merkyor \
  -w /home/merkyor/lynn-engine \
  lynn-eval-base:cu13 \
  python3 scripts/spark_stage6_p2h_selected_layer_prefill_smoke.py \
    --layers 0 --seq-lens 16 \
    --warmup 0 --iters 1 --repeats 1 \
    --json-out reports/stage6/p2h_linearattn_layer_smoke_20260604_020700/result.json
```

### Mixed four-layer mini-chain

Remote run dir:
`/home/merkyor/lynn-engine/reports/stage6/p2h_mixed4_layer_smoke_20260604_020809`

```bash
docker run --rm --gpus all --ipc=host \
  -e PYTHONNOUSERSITE=1 -e PYTHONUNBUFFERED=1 \
  -v /home/merkyor:/home/merkyor \
  -w /home/merkyor/lynn-engine \
  lynn-eval-base:cu13 \
  python3 scripts/spark_stage6_p2h_selected_layer_prefill_smoke.py \
    --layers 0-3 --seq-lens 16 \
    --warmup 0 --iters 1 --repeats 1 \
    --json-out reports/stage6/p2h_mixed4_layer_smoke_20260604_020809/result.json
```

## Coverage

| run | layers | layer types | seq lens | status |
|---|---:|---|---|---|
| P2-Ha | `3` | full_attention | `16,64` | PASS |
| P2-Hb | `0` | linear_attention | `16` | PASS |
| P2-Hc | `0-3` | linear, linear, linear, full | `16` | PASS |

The first attempted `0-3, T=16/64` run was manually stopped before clean
result collection after exposing harness trace gaps and the old torch-only
linear-attention prefill wall. It is not counted as a P2E numeric/perf failure.

## Memory

| run | BF16 active experts | packed active experts | after deleting BF16 active | P2E peak | stream peak |
|---|---:|---:|---:|---:|---:|
| full-attn L3 T16 | 1.500 GiB | 0.563 GiB | 0.629 GiB | 0.690 GiB | 12.689 GiB |
| full-attn L3 T64 | 1.500 GiB | 0.563 GiB | 0.629 GiB | 0.696 GiB | 12.691 GiB |
| linear-attn L0 T16 | 1.500 GiB | 0.563 GiB | 0.640 GiB | 0.721 GiB | 12.701 GiB |
| mixed L0-3 T16 | 6.000 GiB | 2.250 GiB | 2.525 GiB | 2.606 GiB | 14.585 GiB |

No-active-shadow gate: **PASS**.

## Numeric Gate

| run | comparison | cosine | rel L2 | max abs | argmax |
|---|---|---:|---:|---:|---|
| full-attn L3 T16 | P2E vs BF16 | 0.999999649 | 8.383e-04 | 0.0078125 | match |
| full-attn L3 T64 | P2E vs BF16 | 0.999999681 | 7.991e-04 | 0.0078125 | match |
| linear-attn L0 T16 | P2E vs BF16 | 0.999999834 | 5.755e-04 | 0.0156250 | match |
| mixed L0-3 T16 | P2E vs BF16 | 0.999983027 | 5.828e-03 | 0.0468750 | match |

`stream_bf16` remains exact vs BF16 for all banked runs.

Numeric gate: **PASS**.

## Latency Gate

| run | BF16 prefill | stream_bf16 | P2E hybrid | P2E vs BF16 | P2E vs stream |
|---|---:|---:|---:|---:|---:|
| full-attn L3 T16 | 12.21 ms | 976.01 ms | 8.68 ms | 1.407x | 112.44x |
| full-attn L3 T64 | 20.75 ms | 941.14 ms | 20.51 ms | 1.012x | 45.90x |
| linear-attn L0 T16 | 15.71 ms | 1000.90 ms | 13.45 ms | 1.168x | 74.42x |
| mixed L0-3 T16 | 58.57 ms | 2394.57 ms | 45.97 ms | 1.274x | 52.09x |

Speed gate: **PASS** for selected-layer full prefill.

## Caveats

- This is synthetic-hidden selected-layer prefill, not tokenized full-model e2e
  prefill.
- The old linear-attention prefill path is torch-only and CPU/dispatch-heavy;
  long `T=64` mixed-chain runs should not be promoted without a separate
  linear-attention prefill kernel/trace gate.
- Default remains off. `p2e_hybrid` only runs when
  `LYNN_PACKED_PREFILL_SLOW=1`, `LYNN_PACKED_PREFILL_SLOW_MODE=p2e_hybrid`,
  and `LYNN_PACKED_PREFILL_P2E_LAYERS` selects the layer.

## Decision

Bank P2-H. The next meaningful gate is P2-I:

| next gate | requirement |
|---|---|
| P2-I all-MoE selected prefill | Expand `LYNN_PACKED_PREFILL_P2E_LAYERS` beyond the first 4 layers; keep hidden agreement, memory, and latency gates explicit. |
| P2-J linear-attn prefill trace | Separate the torch-only linear-attn prefill wall from P2E MoE behavior; decide whether a native linear-attn prefill kernel is needed before server promotion. |
| P3 serving A/B | Only after P2-I/P2-J: remove per-request reload in server path and measure multi-request behavior. |

Keep default off until full prefill and RC quality pass.
