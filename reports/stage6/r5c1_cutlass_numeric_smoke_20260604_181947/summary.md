# Stage 6 R5-C1 CUTLASS Numeric Smoke Summary

| Field | Value |
|---|---|
| Result | `/root/autodl-tmp/src/lynn-engine-r5c1-codex/reports/stage6/r5c1_cutlass_numeric_smoke_20260604_181947/result.json` |
| Decision | `PASS_R5C1_CUTLASS_NVF4_UE4M3_NUMERIC_SMOKE` |
| CUTLASS dir | `/root/autodl-tmp/src/cutlass` |
| CUTLASS git | `2599f2975b06a67d5ee25e4a7292afeda1475c9b` (`main`) |
| Example | `/root/autodl-tmp/src/cutlass/examples/79_blackwell_geforce_gemm/79d_blackwell_geforce_nvfp4_grouped_gemm.cu` |
| Shape | `M=256 N=128 K=256 groups=2 iterations=1` |
| Build invoked | `True` |
| Build succeeded | `True` |
| Temporary CUDA atomic patch applied/restored | `True` / `True` |
| Numeric smoke banked | `True` |
| Grouped-MoE FP4-MMA POC banked | `False` |
| Kernel speed banked | `False` |
| Default promotion banked | `False` |

## Run Gates

| Gate | Value |
|---|---:|
| Cooperative schedule passed | `True` |
| Pingpong schedule passed | `True` |
| Host reference seen | `True` |
| Disposition passed count >= 2 | `True` |
| No no-op device gate | `True` |
| Avg runtime ms | `[0.021184, 0.01808]` |
| TFLOPS | `[1.58395, 1.85589]` |

## Boundary

- This R5-C1 artifact banks only `banked_numeric_smoke=true` for CUTLASS native NVF4 + UE4M3.
- It does not bank a Lynn grouped-MoE FP4-MMA kernel, speed claim, or default runtime promotion.
- The next gate is selected expert gate/up numeric smoke, not a grouped-MoE speed headline.

