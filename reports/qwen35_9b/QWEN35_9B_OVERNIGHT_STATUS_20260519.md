# Qwen3.5-9B Overnight Status — 2026-05-19

## Current Matrix

| Variant | Stack | Size | MMLU 500 5-shot | GPQA Diamond | R6000 Speed |
|---|---|---:|---:|---:|---:|
| BF16 official | Transformers direct eval | 19G | 77.20% (386/500) | 44.95% (89/198) | not served by Lynn yet |
| Q4_K_M GGUF | llama.cpp CUDA, `--reasoning off` for quality | 5.5G | 76.00% (380/500) | 37.37% (74/198) | 168.23 TPS single 512, 420.63 TPS concurrent 8 |
| Lynn-native W4A16 NVFP4 | Lynn Engine CUDA | 8.3G | 75.20% (376/500) | 42.93% (85/198) | 40.9 TPS single 512, 40.2 TPS concurrent 8 |

## Key Readout

Q4_K_M is the immediate R6000 speed reference for 9B dense: it reaches 168.23 tok/s single-stream at 512 tokens and 420.63 tok/s total at concurrency 8. Long-context smoke now passes at 4k, 16k, and 32k prompt chars when the llama.cpp server runs with a single long-context slot.

BF16 direct quality remains the best confirmed 9B quality number in this local harness. Q4_K_M is close on MMLU but loses GPQA versus BF16 in no-thinking A/B/C/D mode.

Lynn-native W4A16 NVFP4 is no longer blocked: dense resident smoke, OpenAI serving matrix, MMLU, and GPQA all completed. Its current value is NVIDIA-native compatibility and quality retention; its speed is not competitive with Q4_K_M yet, so the next 9B runtime target is dense NVFP4 kernel/launch optimization rather than additional quantization.

An overnight fast-profile sweep found the first safe NVFP4 speed candidate:
`linear_graph_only` stayed exact on the 3-prompt direct-runner gate at both
128 and 512 generated tokens, improving 128-token decode from 40.05 to
59.73 TPS and 512-token decode from 40.71 to 60.16 TPS. The full 35B fast
profile is not safe for 9B yet (`FAST_PROFILE_DRIFT`), so only the linear graph
slice should be promoted to the next OpenAI/P25 serving gate.

That serving gate is now complete. With only `LYNN_LINEAR_STATE_UPDATE=inplace`
and `LYNN_LINEAR_BLOCK_GRAPH{,_REUSE,_PREWARM}=1`, P150 reports decode TPS of
60.80 / 61.47 / 61.69 at 128 / 256 / 512 tokens, with linear-block graph reuse
true on all 9 P25 requests. This upgrades the safe NVFP4 R6000 service line from
~40.9 to ~61.7 decode TPS. Concurrent serving still needs a separate matrix
before comparing against Q4_K_M's 420.63 TPS x8 result.

The follow-up P151 serving matrix confirms the single-stream upgrade but also
shows the current Lynn server does not yet batch concurrent 9B requests:
single 512 wall TPS is 60.09, x2/x4/x8 concurrent total TPS is
60.03/60.08/60.11, and long-context 4k/16k/32k is 56.11/51.38/45.02 TPS.
So 9B NVFP4 now has a stronger NVIDIA single-stream story, while Q4_K_M remains
the multi-request throughput reference.

## Artifacts

| Artifact | Path |
|---|---|
| BF16 quality summary | `reports/qwen35_9b/bf16_transformers_20260519_0102_quality_summary.json` |
| Q4_K_M speed JSON | `reports/qwen35_9b/r6000_qwen35_9b_q4km_baseline_20260519_0107.json` |
| Q4_K_M speed report | `reports/qwen35_9b/r6000_qwen35_9b_q4km_baseline_20260519_0107.md` |
| Q4_K_M quality summary | `reports/qwen35_9b/q4km_llamacpp_reasoning_off_20260519_0115_quality_summary.json` |
| Q4_K_M quality report | `reports/qwen35_9b/q4km_llamacpp_reasoning_off_20260519_0115_quality_summary.md` |
| NVFP4 speed matrix | `reports/qwen35_9b/r6000_qwen35_9b_nvfp4_openai_matrix_full_codex_20260519_022023.json` |
| NVFP4 fast-profile report | `reports/qwen35_9b/QWEN35_9B_NVFP4_FAST_PROFILE_P148_P149_20260519.md` |
| NVFP4 linear-graph P25 report | `reports/qwen35_9b/QWEN35_9B_NVFP4_LINEAR_GRAPH_SERVING_P150_20260519.md` |
| NVFP4 linear-graph matrix | `reports/qwen35_9b/QWEN35_9B_NVFP4_LINEAR_GRAPH_MATRIX_P151_20260519.md` |
| NVFP4 MMLU summary | `reports/qwen35_9b/nvfp4_openai_quality_20260519_022635_mmlu_n500.summary.json` |
| NVFP4 GPQA summary | `reports/qwen35_9b/nvfp4_openai_quality_20260519_022635_gpqa.summary.json` |
| Release gate summary | `reports/qwen35_9b/qwen35_9b_release_gate_summary.json` |
