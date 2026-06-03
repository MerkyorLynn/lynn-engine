# Stage 6 Phase 2-J — linear-attention prefill trace

**Date:** 2026-06-04
**Host:** Spark GB10 (`dgx-spark`)
**Model:** `/home/merkyor/models/Qwen3.6-35B-A3B-lynn-native-w4a16-nvfp4-from-bf16-20260526`
**Layer:** `0` (`linear_attention`)
**Runner:** `scripts/spark_stage6_p2j_linear_attn_prefill_trace.py`

## Verdict

**P2-J passes as a trace gate and identifies the next kernel target.** The
linear-attention prefill wall is not QKV/out projection, depthwise conv, or
RMSNormGated. It is the torch-only `chunk_gated_delta_with_state` recurrence:
~71-76% of traced wall time across T=16..512.

The traced implementation is numerically exact against `prefill_linear_attn()`
for output, recurrent state, and conv state.

## Commands

Short trace:

Remote run dir:
`/home/merkyor/lynn-engine/reports/stage6/p2j_linear_attn_prefill_trace_20260604_022743`

```bash
docker run --rm --gpus all --ipc=host \
  -e PYTHONNOUSERSITE=1 -e PYTHONUNBUFFERED=1 \
  -v /home/merkyor:/home/merkyor \
  -w /home/merkyor/lynn-engine \
  lynn-eval-base:cu13 \
  python3 scripts/spark_stage6_p2j_linear_attn_prefill_trace.py \
    --layer 0 --seq-lens 16,64,128 --repeats 1 \
    --json-out reports/stage6/p2j_linear_attn_prefill_trace_20260604_022743/result.json
```

Long trace:

Remote run dir:
`/home/merkyor/lynn-engine/reports/stage6/p2j_linear_attn_prefill_trace_long_20260604_022808`

```bash
docker run --rm --gpus all --ipc=host \
  -e PYTHONNOUSERSITE=1 -e PYTHONUNBUFFERED=1 \
  -v /home/merkyor:/home/merkyor \
  -w /home/merkyor/lynn-engine \
  lynn-eval-base:cu13 \
  python3 scripts/spark_stage6_p2j_linear_attn_prefill_trace.py \
    --layer 0 --seq-lens 256,512 --repeats 1 \
    --json-out reports/stage6/p2j_linear_attn_prefill_trace_long_20260604_022808/result.json
```

## Numeric Gate

| seq len | output vs `prefill_linear_attn` | state vs `prefill_linear_attn` | conv vs `prefill_linear_attn` |
|---:|---|---|---|
| 16 | exact, argmax match | exact, argmax match | exact, argmax match |
| 64 | exact, argmax match | exact, argmax match | exact, argmax match |
| 128 | exact, argmax match | exact, argmax match | exact, argmax match |
| 256 | exact, argmax match | exact, argmax match | exact, argmax match |
| 512 | exact, argmax match | exact, argmax match | exact, argmax match |

Numeric gate: **PASS**.

## Segment Wall Time

| seq len | full prefill wall | traced total | `chunk_gated_delta_with_state` | chunk share |
|---:|---:|---:|---:|---:|
| 16 | 2.80 ms | 3.36 ms | 2.56 ms | 76.15% |
| 64 | 2.73 ms | 3.19 ms | 2.42 ms | 75.75% |
| 128 | 2.95 ms | 3.59 ms | 2.68 ms | 74.74% |
| 256 | 3.58 ms | 4.68 ms | 3.47 ms | 74.16% |
| 512 | 5.86 ms | 5.70 ms | 4.08 ms | 71.51% |

The second-largest segments are small:

| seq len | next largest non-chunk segment |
|---:|---|
| 16 | `qkv_projection` 0.165 ms |
| 64 | `qkv_projection` 0.176 ms |
| 128 | `qkv_projection` 0.185 ms |
| 256 | `causal_depthwise_conv_silu` 0.238 ms |
| 512 | `causal_depthwise_conv_silu` 0.393 ms |

## Interpretation

P2-J separates the old linear-attention prefill wall from P2E MoE behavior:

- P2E MoE replacement remains worth carrying forward: P2-H/P2-I already showed
  selected-layer full prefill speed and memory wins after deleting active BF16
  expert shadows.
- Linear-attention prefill is now a distinct native-kernel target. The dominant
  segment is the chunked gated-delta recurrence, not matmul projections.
- Continuing to tune projection bridges will not remove the prefill wall. The
  next meaningful kernel should target `chunk_gated_delta_with_state` /
  `chunk_gated_delta_rule_torch` with a native/fused prefill implementation.

## Decision

Bank P2-J. Before server promotion, run P2-K:

| next gate | requirement |
|---|---|
| P2-K gated-delta prefill kernel PoC | Implement or wire a native/fused prefill path for `chunk_gated_delta_with_state`; compare output/state/conv exactness and segment wall time. |
| P2-L larger/all selected-MoE sweep | After P2-K, scale selected layers beyond `0-7` and track cumulative hidden drift. |
| P3 serving A/B | Only after P2-K/P2-L: remove per-request reload in the server path and measure multi-request behavior. |

Keep default off until full prefill and RC quality pass.
