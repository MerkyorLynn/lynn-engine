# Stage 6 R5-C2 Selected-Expert Gate/Up Smoke Summary

| Field | Value |
|---|---|
| Result | `/root/autodl-tmp/src/lynn-engine-r5c2-codex/reports/stage6/r5c2_selected_expert_gateup_smoke_20260604_192904/result.json` |
| Decision | `PASS_R5C2_SELECTED_EXPERT_GATEUP_NUMERIC_SMOKE` |
| CUTLASS dir | `/root/autodl-tmp/src/cutlass` |
| CUTLASS git | `2599f2975b06a67d5ee25e4a7292afeda1475c9b` (`main`) |
| Example | `/root/autodl-tmp/src/cutlass/examples/79_blackwell_geforce_gemm/79d_blackwell_geforce_nvfp4_grouped_gemm.cu` |
| Benchmark file | `/root/autodl-tmp/src/lynn-engine-r5c2-codex/reports/stage6/r5c2_selected_expert_gateup_smoke_20260604_192904/result.benchmark.txt` |
| Selected tokens/top_k/experts | `128 / 2 / 4` |
| Tokens per expert | `[32, 64, 64, 96]` |
| Gate/up shape | `M=tokens_per_expert[e] N=128 K=256` |
| Groups seen | `4` |
| Temporary CUDA atomic patch applied/restored | `None` / `None` |
| Selected-expert gate/up smoke banked | `True` |
| Grouped-MoE FP4-MMA POC banked | `False` |
| Kernel speed banked | `False` |
| Default promotion banked | `False` |

## Run Gates

| Gate | Value |
|---|---:|
| Route tokens match | `True` |
| Route top-k unique | `True` |
| Tokens per expert match | `True` |
| Benchmark shapes aligned to 32 | `True` |
| Benchmark groups match experts | `True` |
| Groups seen match experts | `True` |
| Cooperative schedule passed | `True` |
| Pingpong schedule passed | `True` |
| Host reference seen | `True` |
| Disposition passed count >= 2 | `True` |
| Avg runtime ms | `[0.02, 0.018016]` |
| TFLOPS | `[0.838861, 0.93124]` |

## Boundary

- This R5-C2 artifact banks only `banked_selected_expert_gate_up_smoke=true`.
- It maps `tokens_per_expert` to CUTLASS 79d per-group `M` shapes and checks host-reference numeric correctness.
- It does not bank Lynn slot-preserving gather/scatter, down projection, full grouped-MoE speed, kernel speed, or runtime default promotion.
- The next gate is R5-C2B slot-preserving selected-output bridge, not R5-C3 full grouped-MoE speed.
