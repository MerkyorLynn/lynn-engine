# Qwen3.5-9B Lynn Release Model Card Draft · 2026-05-19

## Summary

Qwen3.5-9B Dense is the practical first-release model for Lynn local agents.
It is small enough for desktop use, strong enough to justify a local coding
assistant track, and can be distributed in both ecosystem-friendly GGUF and
NVIDIA-focused Lynn-native NVFP4 formats.

## Artifacts

| Artifact | Size | Target Users | Default Runtime | Status |
|---|---:|---|---|---|
| Q4_K_M GGUF | 5.49 GB | Mac, llama.cpp, fallback Linux users | llama.cpp | stable |
| Lynn-native W4A16 NVFP4 | 8.25 GB | NVIDIA Blackwell users | Lynn Engine | safe |
| Lynn W4A8 / FP4xFP8 | 8.25 GB class | NVIDIA speed research | Lynn Engine | experimental |
| BF16 official | ~18-19 GB | calibration and reference only | Transformers | internal/reference |

## Quality

| Artifact | MMLU 500 | GPQA Diamond | Notes |
|---|---:|---:|---|
| BF16 official | 77.20% | 44.95% | quality ceiling in current grid |
| Q4_K_M GGUF | 76.00% | 37.37% | smallest stable release artifact |
| Lynn-native W4A16 NVFP4 | 75.20-76.00% | 42.93% | better GPQA preservation than Q4_K_M |
| Lynn W4A8 fake-quant | 75.80% | 43.94% | quality looks viable; runtime not promoted |

Thinking-on 50-sample GPQA for Q4_K_M reached 50.00% naive and 83.33%
excluding parse-fail cases at a 16K generation budget.  This is a capability
signal, not a replacement for the thinking-off grid.

## Runtime

R6000 Q4_K_M llama.cpp CUDA:

- 512-token single stream: 165.8 TPS
- 8-request total throughput: 413.5 TPS
- true single-request 32K chars: 55.5 TPS

R6000 Lynn-native NVFP4 W4A16:

- safe profile: roughly 60-62 decode TPS
- stronger GPQA preservation than Q4_K_M
- remains the NVIDIA engine default while W4A8 resident drift is unresolved

## Recommended Defaults

| Platform | Recommendation |
|---|---|
| macOS Apple Silicon | Q4_K_M GGUF through llama.cpp |
| NVIDIA Linux Blackwell | Lynn-native W4A16 NVFP4 through Lynn Engine |
| Windows | WSL2/Docker beta only |
| Spark / GB10 | Q4_K_M or SGLang/Lynn experiments; not the main Lynn-native speed target yet |

## Known Limits

- Lynn Engine 9B NVFP4 currently trails llama.cpp Q4_K_M in raw R6000 TPS.
- W4A8/FP4xFP8 quality is promising, but resident generation still needs token
  drift diagnosis before promotion.
- Native Windows is not promised for the first release.
- 35B A3B remains a side track until Spark MTP reproduction is proven with
  accept-rate and quality data.
