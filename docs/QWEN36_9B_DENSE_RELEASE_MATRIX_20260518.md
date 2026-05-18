# Qwen3.6 9B Dense Release Matrix

**Date:** 2026-05-18T21:30:00+08:00
**Schema:** `lynn-qwen36-9b-dense-release-matrix-v1`

> ⚠️ **Disclaimer:** This matrix is a living document. Numbers marked 🔶 PROVISIONAL are preliminary human benchmarks and must be confirmed by automated Spark/R6000 pipelines before release. Do not use provisional values as final marketing claims.

## Product Positioning

- **9B Dense** = endpoint / 16 GB VRAM / 端侧候选. 目标是在消费级 Mac 和 24 GB NVIDIA 卡上跑满 quality + speed.
- **35B-A3B MoE** = 高质量服务候选. 目标是在 R6000 / B200 上提供 100+ TPS 的 NVFP4 服务.
- **Q4_K_M / GGUF** = Mac + llama.cpp 阵营. 端侧首选, 质量可接受, 体积最小.
- **NVFP4 / Lynn Engine** = NVIDIA / Blackwell 阵营. 服务端首选, 质量最高, TensorCore 加速.

## 9B Dense Endpoint Matrix

| Model | Quant | Runtime | Device | Size (GB) | MMLU | GPQA | Single TPS | Verdict |
|-------|-------|---------|--------|-----------|------|------|------------|---------|
| Qwen3.6-9B-Dense | Q4_K_M | llama.cpp / GGUF | Mac (Apple Silicon, 16–36 GB unified) | 5.20 | 81.00 | — | — | 🔶 PROVISIONAL |
| Qwen3.6-9B-Dense | BF16 | Lynn Engine | NVIDIA (RTX 4090/5080, 24 GB+) | 18.00 | — | — | — | ⏳ PENDING (Spark) |
| Qwen3.6-9B-Dense | W4A16 / NVFP4 | Lynn Engine | NVIDIA (RTX PRO 6000 / B200, Blackwell) | 5.50 | — | — | — | ⏳ PENDING (R6000) |

## 35B MoE Reference (high-quality serving candidate)

| Model | Quant | Runtime | Device | Size (GB) | MMLU | GPQA | Single TPS | Verdict |
|-------|-------|---------|--------|-----------|------|------|------------|---------|
| Qwen3.6-35B-A3B-MoE | Q4_K_M | llama.cpp / GGUF | Mac/R6000 (reference baseline) | 20.50 | 83.00 | 50.00 | 207.00 | 📌 REFERENCE |
| Qwen3.6-35B-A3B-MoE | W4A16 / NVFP4 | Lynn Engine | NVIDIA R6000 (production target) | 23.00 | 84.40 | 49.49 | 107.00 | 📌 REFERENCE |

## Context Notes

🔶 **Qwen3.6-9B-Dense (Q4_K_M)**: 4K–32K context via llama.cpp KV cache; imatrix optional for quality lift. Endpoint/端侧首选量化格式.
• **Qwen3.6-9B-Dense (BF16)**: Full BF16 serving path; no quantization loss. Pending Spark/user quality sweep.
• **Qwen3.6-9B-Dense (W4A16 / NVFP4)**: Packed NVFP4 decode; TensorCore MMA on SM100+. Pending R6000/Lynn engine benchmark.
• **Qwen3.6-35B-A3B-MoE (Q4_K_M)**: Reference baseline from 35B MoE Q4_K_M llama.cpp sweep. R6000 512-token single wall TPS ~207; concurrent 8 total TPS ~501. Spark single-stream reference is ~69.77 TPS.
• **Qwen3.6-35B-A3B-MoE (W4A16 / NVFP4)**: Lynn-native W4A16 NVFP4 serving path. R6000 safe-default decode ~107 TPS target.

## How to Update

1. Edit `reports/qwen36_9b/qwen36_9b_dense_matrix_schema_v1.json` with new benchmark results.
2. Run `bash scripts/qwen36_9b_dense_matrix.sh` to regenerate this Markdown.
3. Backfill `mmlu` / `gpqa` / `single_tps` from Spark (Mac) or R6000 (NVIDIA) benchmark pipelines.
4. Once a row is verified, set `provisional: false` and update `source` with pipeline reference.

## Next Pipelines

- **Spark (Mac/llama.cpp)**: Q4_K_M MMLU/GPQA full-suite; single-thread / concurrent TPS on M3 Max / M4 Max.
- **R6000 (NVIDIA/Lynn)**: BF16 / NVFP4 MMLU/GPQA; decode TPS at 4K/32K context; concurrent batch.
- **Atlas AGPL** (reference only, do not merge): Q4_K_M MMLU baseline for cross-check.
