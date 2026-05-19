# Qwen3.6-35B-A3B Q4_K_M Thinking-On GPQA50

Date: 2026-05-20

## Result

| Model | Quant | Mode | Max tokens | N | Correct | Naive accuracy | Parse fail | Accuracy excl. parse fail |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen3.6-35B-A3B | Q4_K_M imatrix GGUF | thinking-on | 32768 | 50 | 39 | 78.00% | 2 | 81.25% |

## Notes

- Run host: Spark.
- Endpoint: llama.cpp OpenAI-compatible server on port 18196.
- Artifact: `/home/merkyor/models/Qwen3.6-35B-A3B-GGUF-imatrix/Qwen3.6-35B-A3B-Q4_K_M-imatrix.gguf`.
- Runtime: single concurrency, 32K thinking budget.
- Elapsed: 10536.681 seconds.

## Interpretation

This is the first completed Qwen3.6-35B-A3B thinking-on GPQA sample in the current grid. Q4_K_M with 32K thinking moves from the thinking-off 50.00% GPQA full-set reference into an 78.00% naive / 81.25% parse-clean GPQA50 band. It is a sample, not the full 198-question number, but the signal is strong enough to justify finishing the broader thinking-on grid for BF16, Lynn-native NVFP4, and Q4_K_M.

Raw artifacts:

- `reports/qwen36_35b/remote_spark_20260520/qwen36_q4km_gpqa50_thinking32_20260520_043414.summary.json`
- `reports/qwen36_35b/remote_spark_20260520/qwen36_q4km_gpqa50_thinking32_20260520_043414.jsonl`
