# Qwen3.5-9B Lynn-native NVFP4 Thinking + TPS Check

Date: 2026-05-20

## Quality

| Benchmark | Mode | Result |
|---|---|---:|
| MMLU 100-sample | thinking-on 32K | 91/100 = 91.00% |
| MMLU 100-sample parse-clean | thinking-on 32K | 91/99 = 91.92% |

## R6000 Serving TPS

| Max Tokens | Decode TPS |
|---|---:|
| 128 | 60.76 |
| 256 | 61.61 |
| 512 | 61.70 |

Verdict: `P25_READY`.

## Artifacts

- `reports/qwen35_9b/remote_r6000_20260520/qwen35_9b_nvfp4_mmlu100_thinking32_20260520_090113.summary.json`
- `reports/qwen35_9b/remote_r6000_20260520/p150_qwen35_9b_nvfp4_linear_graph_summary_20260520_1115_after_mmlu.json`
- `reports/qwen35_9b/remote_r6000_20260520/p150_qwen35_9b_nvfp4_linear_graph_p25_20260520_1115_after_mmlu.json`
