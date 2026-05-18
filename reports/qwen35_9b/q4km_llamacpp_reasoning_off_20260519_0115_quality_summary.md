# Qwen3.5-9B Q4_K_M Quality — llama.cpp CUDA

**Run:** `20260519_0115`  
**Mode:** llama.cpp server with `--reasoning off`  
**Why:** disable `<think>` output so MMLU/GPQA answer parsing is clean.

| Variant | MMLU 500 5-shot | GPQA Diamond | Parse Fail |
|---|---:|---:|---:|
| Q4_K_M llama.cpp | 76.00% (380/500) | 37.37% (74/198) | 0 / 0 |
| BF16 Transformers baseline | 77.20% (386/500) | 44.95% (89/198) | 3 / 2 |

## Readout

Q4_K_M is very strong on R6000 speed, but this no-thinking quality run shows a real GPQA drop versus BF16. MMLU is close enough to treat as near-noise for a 500-sample slice; GPQA is not.

The release matrix should therefore present Q4_K_M as the fastest and most portable 9B path, while Lynn-native NVFP4 remains the NVIDIA-specialized path once the dense runtime blocker is cleared.
