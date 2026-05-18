# Qwen3.5-9B Overnight Status — 2026-05-19

## Current Matrix

| Variant | Stack | Size | MMLU 500 5-shot | GPQA Diamond | R6000 Speed |
|---|---|---:|---:|---:|---:|
| BF16 official | Transformers direct eval | 19G | 77.20% (386/500) | 44.95% (89/198) | not served by Lynn yet |
| Q4_K_M GGUF | llama.cpp CUDA, `--reasoning off` for quality | 5.5G | 76.00% (380/500) | 37.37% (74/198) | 168.23 TPS single 512, 420.63 TPS concurrent 8 |
| Lynn-native W4A16 NVFP4 | packed artifact ready | 8.3G | blocked | blocked | blocked by dense runtime |

## Key Readout

Q4_K_M is the immediate R6000 speed reference for 9B dense: it reaches 168.23 tok/s single-stream at 512 tokens and 420.63 tok/s total at concurrency 8. Long-context smoke passes at 4k and 16k prompt chars, while 32k currently returns HTTP 400 from the llama.cpp server configuration.

BF16 direct quality remains the best confirmed 9B quality number in this local harness. Q4_K_M is close on MMLU but loses GPQA versus BF16 in no-thinking A/B/C/D mode.

Lynn-native W4A16 NVFP4 packing is complete, but serving is blocked because the current Lynn resident runtime assumes MoE fields and raises `KeyError: 'num_experts'` on dense Qwen3.5-9B configs. The next unblock is dense runtime support or a fail-loud dense path, not more quantization.

## Artifacts

| Artifact | Path |
|---|---|
| BF16 quality summary | `reports/qwen35_9b/bf16_transformers_20260519_0102_quality_summary.json` |
| Q4_K_M speed JSON | `reports/qwen35_9b/r6000_qwen35_9b_q4km_baseline_20260519_0107.json` |
| Q4_K_M speed report | `reports/qwen35_9b/r6000_qwen35_9b_q4km_baseline_20260519_0107.md` |
| Q4_K_M quality summary | `reports/qwen35_9b/q4km_llamacpp_reasoning_off_20260519_0115_quality_summary.json` |
| Q4_K_M quality report | `reports/qwen35_9b/q4km_llamacpp_reasoning_off_20260519_0115_quality_summary.md` |
