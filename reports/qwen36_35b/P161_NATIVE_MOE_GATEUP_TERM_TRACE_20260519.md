# P161 Native MoE Gate/Up Term Trace

Date: 2026-05-19

## Verdict

The packed NVFP4 native gate/up drift is **reduction-tree drift**, not FP4 decode, scale/global arithmetic, or BF16 input conversion drift.

P161 traced the worst P160 coordinate down to 256 individual FP32 products. Every term matched Triton exactly.

## Target

Taken from P160 worst coordinate:

| Field | Value |
|---|---|
| fixture | `layer_16_prompt_00_slot_packed.safetensors` |
| slot | 4 |
| hidden block | 4 |
| kind | gate |
| row | 356 |
| P160 partial max abs | `4.76837158203125e-07` |

## Result

| Metric | Value |
|---|---:|
| term max abs | 0.0 |
| term mean abs | 0.0 |
| term rel L2 | 0.0 |
| term exact | 1 |
| diagnosis | REDUCTION_TREE_DRIFT |

The first 8 term values were identical, and the full 256-term vector was exact. A normal Torch sum of both vectors also matched exactly:

| Sum | Value |
|---|---:|
| Triton terms torch sum | `-1.3349268436431885` |
| Native terms torch sum | `-1.3349268436431885` |
| abs diff | `0.0` |

This differs from the actual P160 partials because both Triton `tl.sum` and the native CUDA shared-memory reduction use their own reduction trees.

## Implication

For exact serving, there are only two credible paths:

1. Reproduce Triton's 256-element `tl.sum` reduction tree inside the native gate/up kernel.
2. Keep Triton as the exact gate/up authority and fuse around it instead of replacing it.

The unsafe path is to keep iterating on FP4 decode or scale math; P161 shows those terms already match bit-for-bit.

Report JSON: `reports/qwen36_35b/p161_moe_gateup_term_trace_worst_20260519.json`
