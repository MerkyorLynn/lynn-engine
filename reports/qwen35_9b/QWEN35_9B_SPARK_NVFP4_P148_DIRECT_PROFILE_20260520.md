# Qwen3.5-9B Spark NVFP4 Direct Profile 2026-05-20

## Purpose

The 9B release track needs a clear Spark/NVIDIA answer. The HTTP P150 service
gate on Spark reported only about 12 decode TPS, which could have been a server
wrapper or logging artifact. P148 reran the same 9B Lynn-native W4A16 NVFP4
model through `LynnIncrementalRunner` directly with `verbose=False` to isolate
the resident runtime from OpenAI HTTP overhead.

## Artifact

- `reports/qwen35_9b/remote_spark_20260520/p148_spark_qwen35_9b_nvfp4_fast_profile_20260520_021704.json`

Spark command:

```bash
PYTHONPATH=/home/merkyor/lynn-engine \
/home/merkyor/comfyui/ComfyUI/.venv/bin/python \
  benchmarks/p148_qwen35_9b_nvfp4_fast_profile.py \
  --model /home/merkyor/models/Qwen3.5-9B-lynn-native-w4a16-nvfp4-v0 \
  --max-seq-len 4096 \
  --max-new 64 128 \
  --out /home/merkyor/reports/qwen35_9b/p148_spark_qwen35_9b_nvfp4_fast_profile_20260520_021704.json
```

## Result

| Mode | Decode TPS mean | Min | Max | Exact vs baseline |
|---|---:|---:|---:|---:|
| Conservative | 10.69 | 10.66 | 10.71 | reference |
| Fast profile | 11.92 | 11.12 | 12.10 | 2/6 |

Verdict: `FAST_PROFILE_DRIFT`.

## Readout

This confirms the 10-12 TPS Spark number is not caused by HTTP logging. The
direct resident runner is also slow. The underlying reason is architectural:

- Qwen3.5-9B dense activates the full dense model every token.
- Qwen3.6-35B-A3B activates only a small MoE slice per token, so it can be
  faster despite being a larger checkpoint on disk.
- Spark sm_121 has FP8 MMA, but no native FP4 MMA. The current Lynn-native
  W4A16 NVFP4 path therefore cannot get the R6000 native FP4 benefit and falls
  back to a slow dequant/BF16 style path.
- The existing true-FP8 resident probes show only small speedups when exactness
  is relaxed, and broad true-FP8 layer replacement drifts.

## Release Implication

Spark/Mac first release should use Q4_K_M through llama.cpp as the stable local
runtime track. That path already has the right user-facing behavior and clears
the 50 TPS class target on Spark/Mac-class deployments.

Lynn-native NVFP4 for 9B remains the NVIDIA/R6000 track and a Spark research
track until a true FP8 exact dense GEMM boundary exists. Do not claim Spark
NVFP4 9B performance from the current W4A16 path.

## Next Work

1. Keep Q4_K_M/llama.cpp as the 9B release default for Mac and Spark fallback.
2. Keep Lynn-native NVFP4 9B for R6000 and Linux NVIDIA users where native FP4
   or future exact FP8 kernels can matter.
3. If Spark NVFP4 must be promoted, implement a true FP8 exact dense FFN
   boundary instead of tuning the current W4A16 path.
