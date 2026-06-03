# Stage 6 Phase 2-I — selected-MoE expansion smoke

**Date:** 2026-06-04
**Host:** Spark GB10 (`dgx-spark`)
**Model:** `/home/merkyor/models/Qwen3.6-35B-A3B-lynn-native-w4a16-nvfp4-from-bf16-20260526`
**Runner:** `scripts/spark_stage6_p2h_selected_layer_prefill_smoke.py`
**Remote run dir:** `/home/merkyor/lynn-engine/reports/stage6/p2i_mixed8_layer_smoke_20260604_021658`

## Verdict

**P2-I passes for an 8-layer selected-MoE expansion smoke.** After P2-H proved
the P2E path inside `_prefill_layer`, this run expands selected layers from
`0-3` to `0-7` while still deleting active routed-expert BF16 shadows.

This is synthetic-hidden selected-layer prefill, not full tokenized e2e prefill.
It is a scale-up gate, not a server promotion.

## Command

```bash
docker run --rm --gpus all --ipc=host \
  -e PYTHONNOUSERSITE=1 -e PYTHONUNBUFFERED=1 \
  -v /home/merkyor:/home/merkyor \
  -w /home/merkyor/lynn-engine \
  lynn-eval-base:cu13 \
  python3 scripts/spark_stage6_p2h_selected_layer_prefill_smoke.py \
    --layers 0-7 --seq-lens 16 \
    --warmup 0 --iters 1 --repeats 1 \
    --json-out reports/stage6/p2i_mixed8_layer_smoke_20260604_021658/result.json
```

## Coverage

| item | value |
|---|---|
| Layers | `0,1,2,3,4,5,6,7` |
| Layer types | linear, linear, linear, full, linear, linear, linear, full |
| Seq len | 16 |
| Active BF16 shadow deleted | yes |

## Memory

| item | value |
|---|---:|
| 8-layer BF16 active experts | 12.000 GiB |
| 8-layer packed active experts | 4.500 GiB |
| after deleting BF16 active shadows | 5.041 GiB |
| P2E peak | 5.123 GiB |
| `stream_bf16` peak | 17.102 GiB |

No-active-shadow gate: **PASS**.

## Numeric Gate

| comparison | cosine | rel L2 | max abs | argmax |
|---|---:|---:|---:|---|
| stream vs BF16, T=16 | 1.000000000 | 0 | 0 | match |
| P2E vs BF16, T=16 | 0.999948277 | 1.017e-02 | 0.0625 | match |

Numeric gate: **PASS**.

## Latency Gate

| seq len | BF16 prefill | `stream_bf16` | P2E hybrid | P2E vs BF16 | P2E vs stream |
|---:|---:|---:|---:|---:|---:|
| 16 | 113.82 ms | 4154.68 ms | 88.96 ms | 1.279x | 46.70x |

Speed gate: **PASS**.

## Decision

Bank P2-I. The next gate is no longer "does P2E survive selected-layer prefill";
it does. The next hard questions are:

| next gate | requirement |
|---|---|
| P2-J linear-attn prefill trace | Separate the old torch-only linear-attn prefill wall from P2E MoE cost; decide whether a native linear-attn prefill kernel is required before server promotion. |
| P2-K larger/all selected-MoE sweep | If P2-J is understood, scale selected layers beyond `0-7` and track cumulative drift. |
| P3 serving A/B | Only after P2-J/P2-K: remove per-request reload in the server path and measure multi-request behavior. |

Keep default off until full prefill and RC quality pass.
