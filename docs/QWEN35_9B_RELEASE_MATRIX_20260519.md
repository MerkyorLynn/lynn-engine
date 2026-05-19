# Qwen3.5-9B Release Matrix · 2026-05-19

## Release Decision

Qwen3.5-9B Dense is the first-release model line.

| Track | Default Runtime | Default Artifact | Status |
|---|---|---|---|
| Mac / general local agents | llama.cpp | Q4_K_M GGUF | stable |
| NVIDIA Linux / Blackwell | Lynn Engine | Lynn-native NVFP4 W4A16 | safe |
| NVIDIA Linux / speed research | Lynn Engine | W4A8 / FP4xFP8 resident | experimental |
| Windows | WSL2 or Docker | same as NVIDIA Linux | beta |

The release should not be framed as a single experimental engine.  It should
ship two practical paths: Q4_K_M for the llama.cpp/Mac ecosystem and NVFP4 for
the NVIDIA/Lynn Engine ecosystem.

## Quality Matrix

These are thinking-off, apples-to-apples quality numbers unless explicitly
marked otherwise.

| Variant | Size | MMLU 500 | GPQA Diamond | Release Role |
|---|---:|---:|---:|---|
| BF16 official | ~18-19 GB | 77.20% | 44.95% | reference |
| Q4_K_M GGUF | 5.49 GB | 76.00% | 37.37% | Mac stable |
| Lynn-native W4A16 NVFP4 | 8.25 GB | 75.20-76.00% | 42.93% | NVIDIA safe |
| Lynn W4A8 fake-quant | 8.25 GB | 75.80% | 43.94% | NVIDIA experimental |
| Q4_K_M thinking-on 50-sample | 5.49 GB | 81.00% sample | 50.00% naive / 83.33% excl. parse-fail | capability signal |

Quality takeaways:

- Q4_K_M is the smallest stable artifact, but GPQA drops more than NVFP4 in the
  thinking-off grid.
- Lynn-native NVFP4 preserves GPQA much better than Q4_K_M while keeping MMLU
  within roughly 1-2 pp of BF16.
- W4A8 fake-quant quality is essentially flat versus W4A16 in the Spark grid.
  It is not a default until the resident FP4xFP8 path passes exact/structured
  runtime gates.

## R6000 Runtime Matrix

| Variant | Runtime | 512-token Single | Concurrent / Batch | Long Context | Notes |
|---|---|---:|---:|---:|---|
| Q4_K_M GGUF | llama.cpp CUDA, `parallel=8` | 165.8 TPS | 413.5 TPS at 8 requests | 16K chars 114.4 TPS | best competitive baseline |
| Q4_K_M GGUF | llama.cpp CUDA, `parallel=1` | 164.5 TPS | no batching gain | 32K chars 55.5 TPS | true single-request 32K |
| NVFP4 W4A16 | Lynn Engine safe profile | ~60-62 decode TPS | flat single-prompt service | 32K path measured in earlier gate | safe NVIDIA default |
| W4A8 FP4xFP8 | Lynn Engine resident | not promoted | not promoted | not promoted | P190/P197 drift investigation |

Important context:

- The earlier `parallel=8` 32K failure on Q4_K_M is a llama.cpp slot partition
  issue, not proof that the model cannot handle 32K.  The `parallel=1` rerun
  passed 32K chars.
- Q4_K_M is currently much faster than Lynn-native NVFP4 on 9B because
  llama.cpp has a mature repacked CUDA path and batching.  Lynn's NVIDIA track
  remains valuable because it gives us a native NVFP4/FP4xFP8 engine path and
  better GPQA preservation than Q4_K_M.

## Distribution

| Domain | Purpose |
|---|---|
| `engine.merkyorlynn.com` | docs, install entrypoint, model cards, API guides |
| `dl.merkyorlynn.com` | model bundles, GGUF, NVFP4 packs, wheels, checksums |
| `mirror.merkyorlynn.com` | optional mainland-China mirror if needed |

Suggested public layout:

```text
https://engine.merkyorlynn.com/docs/qwen35-9b/
https://engine.merkyorlynn.com/docs/qwen35-9b/model-card
https://dl.merkyorlynn.com/models/qwen35-9b/q4_k_m/
https://dl.merkyorlynn.com/models/qwen35-9b/nvfp4-w4a16/
https://dl.merkyorlynn.com/models/qwen35-9b/checksums.sha256
```

## Promotion Rules

| Candidate | Promotion Rule |
|---|---|
| Q4_K_M Mac path | llama.cpp endpoint smoke passes `/v1/models`, chat, and JSON prompt |
| NVFP4 W4A16 | remains NVIDIA safe default while exact/structured gates pass |
| W4A8 / FP4xFP8 | can only move beyond experimental after P197/P190 token drift is resolved and P196 structured gate holds |
| 35B A3B | side track; Spark MTP reproduction must prove no-MTP vs MTP uplift, accept rate, accept length, and quality smoke |

## Source Reports

- `reports/qwen35_9b/r6000_qwen35_9b_q4km_cuda_baseline_20260519_1731.md`
- `reports/qwen35_9b/r6000_qwen35_9b_q4km_cuda_baseline_20260519_1732.md`
- `reports/qwen35_9b/P196_W4A8_STRUCTURED_CONTENT_GATE_20260519.md`
- `reports/qwen35_9b/qwen35_9b_release_matrix.md`
- `docs/QWEN36_35B_SPARK_MTP_REPRO_GATE_20260519.md`
