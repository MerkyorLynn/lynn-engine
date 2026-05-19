# Qwen3.5-9B Lynn-Native NVFP4 Thinking-On GPQA50

Date: 2026-05-20

## Result

| Model | Quant | Mode | Max tokens | N | Correct | Naive accuracy | Parse fail | Accuracy excl. parse fail |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen3.5-9B | Lynn-native W4A16 NVFP4 | thinking-on | 32768 | 50 | 28 | 56.00% | 10 | 70.00% |

## Notes

- Run host: R6000.
- Endpoint: Lynn engine OpenAI-compatible server on port 18192.
- Artifact: `/root/autodl-tmp/models/Qwen3.5-9B-lynn-native-w4a16-nvfp4-v0`.
- Runtime: single concurrency, 32K thinking budget.
- Elapsed: 15494.029 seconds.

## Interpretation

This result is directionally positive versus the 9B NVFP4 thinking-off GPQA reference of 42.93%, but it does not match the Q4_K_M thinking-on parse-clean band. Parse failures remain meaningful even with a 32K token budget, and the native NVFP4 server is slower on this long reasoning workload than the Mac/Spark llama.cpp Q4_K_M path.

For first release, this reinforces the current split:

- Mac/local-first track: Q4_K_M through llama.cpp remains the stable 9B user path.
- NVIDIA track: Lynn-native NVFP4 is viable, but its 9B long-thinking quality and throughput still need dedicated follow-up before it becomes the preferred endpoint.

Raw artifacts:

- `reports/qwen35_9b/remote_r6000_20260520/qwen35_9b_nvfp4_gpqa50_thinking32_20260520.summary.json`
- `reports/qwen35_9b/remote_r6000_20260520/qwen35_9b_nvfp4_gpqa50_thinking32_20260520.jsonl`
