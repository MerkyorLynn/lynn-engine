# P162 Native MoE Gate/Up Reduction Tree Search

Date: 2026-05-19

## Verdict

Simple FP32 reduction trees did **not** reproduce Triton `tl.sum` exactly for the P160 worst row.

The current native CUDA tree is identified: `pairwise_halving` exactly matches the native partial. Triton's reduction lands one FP32 ULP beyond the nearest simple candidates tested here.

## Target

Same P160/P161 coordinate:

| Field | Value |
|---|---|
| fixture | `layer_16_prompt_00_slot_packed.safetensors` |
| slot | 4 |
| hidden block | 4 |
| kind | gate |
| row | 356 |

## Key Values

| Item | Value |
|---|---:|
| Triton partial | `-1.3349272012710571` |
| Native partial | `-1.334926724433899` |
| abs diff | `4.76837158203125e-07` |

Best simple candidates:

| Candidate | Value | Diff vs Triton | Diff vs Native |
|---|---:|---:|---:|
| `left_fold` | `-1.3349270820617676` | `1.1920928955078125e-07` | `3.5762786865234375e-07` |
| `chunk4_pairwise_then_left` | `-1.3349270820617676` | `1.1920928955078125e-07` | `3.5762786865234375e-07` |
| `right_fold` | `-1.334926962852478` | `2.384185791015625e-07` | `2.384185791015625e-07` |
| `pairwise_halving` | `-1.334926724433899` | `4.76837158203125e-07` | `0.0` |

## Implication

The native packed-MoE exact-first route should stop trying random FP4/scale/reduction tweaks. P160-P162 now show:

1. FP4 decode and term products are bit-exact.
2. Current native reduction is deterministic and understood.
3. Triton's internal `tl.sum` tree is not matched by common simple reductions.

Practical next move: keep Triton active-MoE as the exact authority and optimize the boundary around it, or invest in a deliberate Triton-lowering/IR-level reproduction effort. The former is higher ROI for the 122 TPS push.

Report JSON: `reports/qwen36_35b/p162_moe_gateup_reduction_tree_search_20260519.json`
