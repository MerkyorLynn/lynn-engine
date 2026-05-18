# P170 Native FP4 Activation Scratch Probe

Date: 2026-05-19

## Verdict

`LYNN_NATIVE_FP4_ACT_SCRATCH=1` is numerically safe but not a serving
promotion.

It reuses caller-owned activation FP4 quantization scratch for fused native FP4
linear projections, preserving the current Triton quantization kernel and
`torch._scaled_mm` math. P169 and P37 prove exactness, but the P25/structured
gain is below the current safe default.

## R6000 Results

| Gate | Result |
|---|---:|
| P169 linear-core fixtures | 20/20 exact |
| P169 max_abs_max | 0.0 |
| P37 exact | true |
| P37 median speedup | 1.0125x |
| P25 512 decode TPS | 107.07 |
| hard structured | 40/40 |
| hard structured mean decode TPS | 107.43 |
| promotion decision | CLOSED |

P168 micro-census with the scratch path showed a real but small local win:

| Segment | Default Sum | Scratch Sum | Delta |
|---|---:|---:|---:|
| fused native FP4 in-proj | 2.107 ms/token | 1.906 ms/token | -0.201 ms |
| full linear core | 8.993 ms/token | 8.799 ms/token | -0.194 ms |

The local in-proj saving did not convert into a safe P25 promotion.

## Keep / Do Not Promote

Keep the opt-in implementation because it is exact and useful for future larger
linear-core boundaries. Do not enable it by default because current service
gates remain below the safe line.

## Artifacts

- `scripts/qwen36_candidate_env_native_fp4_act_scratch.env`
- `reports/qwen36_35b/p169_linear_core_act_scratch_contract_20260519_0810.json`
- `reports/qwen36_35b/p168_qwen36_linear_core_segment_census_20260519_0811.json`
- `reports/qwen36_35b/r6000_qwen36_w4a16_native_fp4_act_scratch_20260519_0814_promotion_summary.json`
