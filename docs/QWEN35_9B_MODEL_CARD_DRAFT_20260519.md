# Qwen3.5-9B Lynn Release Model Card Draft · 2026-05-19

## Summary

Qwen3.5-9B Dense is the practical first-release model for Lynn local agents.
It is small enough for desktop use, strong enough to justify a local coding
assistant track, and can be distributed in both ecosystem-friendly GGUF and
NVIDIA-focused Lynn-native NVFP4 formats.

## Artifacts

| Artifact | Size | Target Users | Default Runtime | Status |
|---|---:|---|---|---|
| Q4_K_M imatrix GGUF | 5.49 GB | Mac, llama.cpp, fallback Linux users | llama.cpp | stable |
| Lynn-native W4A16 NVFP4 | 8.248 GiB package | NVIDIA Blackwell users | Lynn Engine | safe |
| Lynn W4A8 / FP4xFP8 resident | 8.25 GB class | NVIDIA speed research | Lynn Engine | experimental, not default |
| BF16 official | ~18-19 GB | calibration and reference only | Transformers | internal/reference |

The NVFP4 W4A16 package is 8.248 GiB because BF16 `embed_tokens` and `lm_head`
are intentionally kept in the release artifact.

## Quality

| Artifact | MMLU 500 | GPQA Diamond | Notes |
|---|---:|---:|---|
| BF16 official | 77.20% | 44.95% | quality ceiling in current grid |
| Q4_K_M imatrix GGUF | 76.00% | 37.37% | smallest stable Mac release artifact |
| Lynn-native W4A16 NVFP4 | 75.20-76.00% | 42.93% | better GPQA preservation than Q4_K_M |
| Lynn W4A8 / FP4xFP8 resident | pending promoted report | pending promoted report | experimental; P193 blocks promotion if W4A8 quality report is missing |

Thinking-on 50-sample GPQA for Q4_K_M reached 50.00% naive and 83.33%
excluding parse-fail cases at a 16K generation budget. This is a capability
signal, not a replacement for the thinking-off grid. The 32K thinking-on GPQA
run is still long-running; do not claim a final full 198-question score from
p201 live summaries.

## Runtime

R6000 Q4_K_M imatrix GGUF llama.cpp CUDA:

- 512-token single stream: 165.8 TPS
- 8-request total throughput: 413.5 TPS
- true single-request 32K chars: 55.5 TPS

R6000 Lynn-native NVFP4 W4A16:

- safe profile: roughly 60-62 decode TPS
- stronger GPQA preservation than Q4_K_M
- current package is 8.248 GiB because BF16 `embed_tokens` and `lm_head` are kept
- remains the NVIDIA engine default while W4A8 / FP4xFP8 resident is gated

## Recommended Defaults

| Platform | Recommendation |
|---|---|
| macOS Apple Silicon | Q4_K_M imatrix GGUF through llama.cpp |
| NVIDIA Linux Blackwell | Lynn-native W4A16 NVFP4 through Lynn Engine |
| Windows | WSL2/Docker beta only |
| Spark / GB10 | Q4_K_M or SGLang/Lynn experiments; not the main Lynn-native speed target yet |

## Checksum Tools

- Generate release manifests with `scripts/qwen35_9b_release_checksums.py`.
- Verify downloaded Q4_K_M and NVFP4 artifacts with `scripts/local_qwen35_9b_verify_checksums.sh`.

## Known Limits

- Lynn Engine 9B NVFP4 currently trails llama.cpp Q4_K_M in raw R6000 TPS.
- W4A8 / FP4xFP8 resident remains experimental, not default; P193 requires
  P185/P190/P197/P198-style gates and blocks promotion if the W4A8 quality
  report is missing.
- Native Windows is not promised for the first release.
- 35B A3B remains a side track until Spark MTP reproduction is proven with
  accept-rate and quality data.
