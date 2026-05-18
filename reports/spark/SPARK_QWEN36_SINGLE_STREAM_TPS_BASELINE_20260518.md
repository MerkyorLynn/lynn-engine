# Spark Qwen3.6-35B-A3B Single-Stream TPS Baseline

Date: 2026-05-18

## Summary

Spark `sm_121` was benchmarked across the three current official
Qwen3.6-35B-A3B serving routes:

| Quant | Stack | Load | TPS mean, 3 single | Median | Peak | Stddev, 10 mixed | Versus R6000 116.9 |
|---|---|---:|---:|---:|---:|---:|---:|
| BF16 official | SGLang dev-cu13 | 190s | 30.14 | 30.19 | 30.30 | 0.43 | 26% |
| Q4_K_M-imatrix GGUF | llama.cpp server-cuda | 50s | 69.77 | 69.76 | 70.08 | 7.56, one outlier | 60% |
| W4A16 NVFP4 Lynn-native | lynn-engine Config D | 318s | 38.96 | 38.85 | 39.18 | 0.09 | 33% |

Spark original JSON reports:

```text
~/reports/bench-20260518/qwen36_bf16_sp01.json
~/reports/bench-20260518/qwen36_q4km_sp01.json
~/reports/bench-20260518/qwen36_w4a16_nvfp4_sp01.json
```

## Interpretation

Q4_K_M through llama.cpp is the current best Spark single-stream serving route.
It reaches about `70 TPS`, is already quality-competitive
(`83.00%` MMLU / `50.00%` GPQA in the paired quality run), and loads quickly.

Lynn-native W4A16 NVFP4 is stable but not fast on Spark: its decode TPS stddev
is only `0.09`, which confirms the kernel path is deterministic, but `38.96 TPS`
is far below the llama.cpp Q4_K_M route. This is expected after the hardware
audit: Spark `sm_121` does not expose the native FP4 MMA path used by R6000.

The R6000 gap is therefore not just a software tuning issue:

| Capability | Spark GB10 `sm_121` | R6000 `sm_120a` |
|---|---|---|
| Native FP4 MMA | not available in current toolchain probes | available through the R6000 path |
| W4A16 implementation | dequant to BF16 plus BF16 TC MMA | native E2M1 / FP4-oriented path |
| Effective memory bandwidth | LPDDR5X class, roughly 200 GB/s attainable | GDDR7 class, much higher |

## Strategy Update

Spark is no longer the primary Lynn-native single-stream TPS battlefield.

Recommended Spark serving default:

```text
official Qwen3.6-35B-A3B Q4_K_M-imatrix GGUF
  -> llama.cpp server-cuda
  -> about 70 TPS single-stream
```

Recommended Lynn Engine work:

```text
R6000 first
  -> official Qwen3.6-35B-A3B W4A16 NVFP4
  -> strict native kernel islands
  -> W4A8 only after structured/code/tool-call quality gates pass
```

Spark Lynn-native remains useful for compatibility, determinism, and fallback
validation, but it should not consume the main 155 TPS loop unless a true
Spark-specific FP8 mirror path is being tested.

