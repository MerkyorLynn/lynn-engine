# P160 Native MoE Gate/Up Partial-Sum Trace

Date: 2026-05-19

## Verdict

`native_grouped_per16_nonatomic` / packed-slot NVFP4 gate/up drift starts **inside each 256-hidden block reduction**, not in the final hidden-block accumulation.

This closes the "maybe the 8 block sums are being added differently" hypothesis. Triton partials reconstruct P147 exactly, and native partials reconstruct native raw exactly, but native-vs-Triton per-block partials differ at FP32 `1e-7` scale.

## R6000 Run

```bash
LYNN_NATIVE_CUDA_BUILD_DIR=/tmp/lynn_engine_native_build/p160_partial \
/root/autodl-tmp/conda-envs/r6000-eval/bin/python \
  benchmarks/p160_native_packed_moe_gateup_partial_trace.py \
  --packed-fixtures /root/autodl-tmp/reports/qwen36_35b/p138_packed_slot_fixtures_kimi_20260518 \
  --p147-reference-dir /root/autodl-tmp/reports/qwen36_35b/p147_triton_stage_reference_20260519_0318 \
  --out /root/autodl-tmp/reports/qwen36_35b/p160_moe_gateup_partial_trace_all_20260519.json
```

## Result Summary

| Metric | Value |
|---|---:|
| selected fixtures | 18 |
| partial exact | 0/18 |
| partial max abs max | 4.76837158203125e-07 |
| diagnosis | WITHIN_HIDDEN_BLOCK_DRIFT |
| Triton partial -> P147 inter max | 0.0 |
| Native partial -> native raw max | 0.0 |
| Native partial -> P147 inter max | 2.44140625e-04 |

Worst fixture:

| Field | Value |
|---|---|
| fixture | `layer_16_prompt_00_slot_packed.safetensors` |
| max index | `[slot=4, hidden_block=4, kind=gate, row=356]` |
| Triton partial | `-1.3349272012710571` |
| Native partial | `-1.334926724433899` |
| diff | `-4.76837158203125e-07` |

## Implication

The native kernel is mathematically close, but exact serving promotion is blocked by the 256-term FP32 reduction contract inside the gate/up dot product. The next useful probe is term/reduction-tree level:

1. Emit decoded per-term products for the worst `(slot, hidden_block, gate/up, row)`.
2. Check whether terms are bit-identical and only `tl.sum` reduction order differs.
3. If terms match, either reproduce Triton's reduction tree or keep Triton as the exact gate/up authority and fuse only around its boundary.
4. If terms differ, isolate FP4 decode / scale-div-global / multiply order and mirror Triton's lowering.

Report JSON: `reports/qwen36_35b/p160_moe_gateup_partial_trace_all_20260519.json`
