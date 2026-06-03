# Stage 6 Phase 2-G — multi-layer p2e_hybrid MoE smoke

**Date:** 2026-06-04  
**Host:** Spark GB10 (`dgx-spark`)  
**Model:** `/home/merkyor/models/Qwen3.6-35B-A3B-lynn-native-w4a16-nvfp4-from-bf16-20260526`  
**Layers:** `0-3`  
**Runner:** `scripts/spark_stage6_p2g_multilayer_moe_smoke.py`  
**Remote run dir:** `/home/merkyor/lynn-engine/reports/stage6/p2g_multilayer_moe_smoke_20260604_014613`

## Verdict

**P2-G passes as a 4-layer MoE-only no-reload smoke.** The P2-F
`p2e_hybrid` engine path remains correct and memory-clean across consecutive
MoE layers with residual addition on synthetic hidden states.

This is not full transformer prefill yet; attention, layernorms, and residual
from the surrounding block are still outside this smoke. But it is the first
multi-layer evidence that the packed active-MoE replacement can beat both
`stream_bf16` and resident BF16 MoE while keeping BF16 active shadows deleted.

## Command

```bash
docker run --rm --gpus all --ipc=host \
  -e PYTHONNOUSERSITE=1 -e PYTHONUNBUFFERED=1 \
  -v /home/merkyor:/home/merkyor \
  -w /home/merkyor/lynn-engine \
  lynn-eval-base:cu13 \
  python3 -u scripts/spark_stage6_p2g_multilayer_moe_smoke.py \
    --layers 0-3 --batches 16,64 \
    --warmup 0 --iters 1 --repeats 1 \
    --json-out reports/stage6/p2g_multilayer_moe_smoke_20260604_014613/result.json
```

## Memory

| item | bytes |
|---|---:|
| 4-layer BF16 active experts | 6.000 GiB |
| 4-layer packed active experts | 2.250 GiB |
| after deleting BF16 active shadows | 2.525 GiB |

| batch | P2E peak | `stream_bf16` peak |
|---:|---:|---:|
| 16 | 2.526 GiB | 14.525 GiB |
| 64 | 2.527 GiB | 14.526 GiB |

No-active-shadow gate: **PASS**.

## Numeric Gate

The first attempted raw-MoE chain collapsed synthetic hidden states toward zero,
making cosine meaningless. This recorded run uses residual addition:

```python
h = h + moe(h)
```

| comparison | cosine | rel L2 | max abs | argmax |
|---|---:|---:|---:|---|
| stream vs BF16, M=16 | 1.000000000 | 0 | 0 | match |
| stream vs BF16, M=64 | 1.000000000 | 0 | 0 | match |
| P2E vs BF16, M=16 | 0.999999869 | 5.114e-04 | 0.00390625 | match |
| P2E vs BF16, M=64 | 0.999999840 | 5.651e-04 | 0.00781250 | match |

Numeric gate: **PASS** for this MoE-only multi-layer smoke.

## Latency Gate

| batch | BF16 4-layer MoE | `stream_bf16` 4-layer | P2E 4-layer | P2E vs BF16 | P2E vs stream |
|---:|---:|---:|---:|---:|---:|
| 16 | 47.23 ms | 2388.18 ms | 34.20 ms | 1.381x | 69.82x |
| 64 | 85.12 ms | 2432.64 ms | 80.97 ms | 1.051x | 30.04x |

Speed gate: **PASS** for this 4-layer MoE-only smoke.

## Decision

Bank P2-G and move to P2-H:

| next gate | requirement |
|---|---|
| P2-H full prefill selected-layer smoke | Run full transformer prefill with selected MoE layers using `p2e_hybrid`; compare text/token or hidden agreement, memory, and latency. |
| P2-I all-MoE no-reload prefill | If P2-H holds, expand `LYNN_PACKED_PREFILL_P2E_LAYERS` to all MoE layers and measure end-to-end prefill without reload. |
| P3 serving A/B | Only after full prefill gates: remove per-request reload in the server path and measure multi-request behavior. |

Keep default off until full prefill and RC quality pass.
