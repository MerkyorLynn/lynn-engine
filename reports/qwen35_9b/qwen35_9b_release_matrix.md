# Qwen3.5-9B Dense Release Matrix (Report)

**Date:** 2026-05-19  
**Schema:** `lynn-qwen35-9b-dense-release-matrix-v2`  
**Generated from:** `reports/qwen35_9b/qwen35_9b_release_matrix.json`

---

## Matrix

| Platform | Runtime | Format | Size (GiB) | Single TPS | Batch TPS | Long Context | Status |
|---|---|---|---|---:|---:|---:|---|
| Mac (Apple Silicon) | llama.cpp | Q4_K_M-imatrix | 5.5 | — | — | — | stable |
| NVIDIA Linux (R6000) | llama.cpp CUDA | Q4_K_M-imatrix | 5.5 | 166 | 413 | 4K→174 | baseline |
| NVIDIA Linux (R6000) | Lynn Engine | NVFP4 W4A16 | 8.2 | 62 | — | 4K→60 | safe |
| NVIDIA Linux (R6000) | Lynn Engine | W4A8 fake-quant | 8.2 | 58 | — | 4K→58 | experimental |
| Windows (WSL2/Docker) | Lynn Engine Docker | NVFP4 W4A16 | 8.2 | — | — | — | beta |
| DGX Spark (GB10) | Lynn Engine | NVFP4 W4A16 | 8.2 | — | — | — | pending |

## Evidence Links

- **Q4_K_M baseline 1731**: `reports/qwen35_9b/r6000_qwen35_9b_q4km_cuda_baseline_20260519_1731.md`
- **Q4_K_M baseline 1732**: `reports/qwen35_9b/r6000_qwen35_9b_q4km_cuda_baseline_20260519_1732.md`
- **NVFP4 P25 serving gate**: `reports/qwen35_9b/p150_qwen35_9b_nvfp4_linear_graph_summary_20260519_120000_convstrict.json`
- **NVFP4 exact-fast isolation**: `reports/qwen35_9b/p183_qwen35_9b_nvfp4_exact_fast_isolation_20260519_115751.json`
- **NVFP4 convstrict exact gate**: `reports/qwen35_9b/p184_qwen35_9b_nvfp4_convstrict_exact_gate_20260519_120255.json`
- **W4A8 structured gate**: `reports/qwen35_9b/p196_qwen35_9b_w4a8_structured_content_gate_20260519_1718_p196_chat70.json`

## Distribution

- Docs: `https://engine.merkyorlynn.com/docs/qwen35-9b/`
- Downloads: `https://dl.merkyorlynn.com/models/qwen35-9b-dense/`
- Mirror: `https://mirror.merkyorlynn.com/models/qwen35-9b-dense/`
