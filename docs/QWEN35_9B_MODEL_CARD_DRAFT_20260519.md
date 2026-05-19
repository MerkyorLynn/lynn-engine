# Qwen3.5-9B Dense — Lynn Model Card Draft

**Date:** 2026-05-19  
**Model:** Qwen3.5-9B-Dense  
**Publisher:** MerkyorLynn (Lynn Engine)  
**License:** Qwen3.5-9B 原始许可（本卡描述 Lynn 量化产物）

---

## 1. 模型概述

Qwen3.5-9B-Dense 是阿里云通义千问团队发布的 **9B 参数 Dense（非 MoE）模型**。Lynn Engine 将其作为 **首期 release 的主线模型**，提供以下量化格式：

| 格式 | 大小 | 适用平台 | 状态 |
|---|---|---|---|
| BF16 | ~18 GB | NVIDIA Blackwell (reference) | internal |
| NVFP4 W4A16 | ~8.2 GB | NVIDIA Blackwell sm_120+ | **safe default** |
| Q4_K_M-imatrix | ~5.5 GB | Mac / NVIDIA / CPU | **stable fallback** |
| W4A8 (fake-quant) | ~8.2 GB | NVIDIA Blackwell (research) | experimental |

---

## 2. 硬件要求

### 2.1 推荐配置

| 平台 | GPU | 显存 | CUDA | 备注 |
|---|---|---|---|---|
| **NVIDIA Linux** | RTX 5090 / RTX PRO 6000 / B200 | 24–96 GB | 12.8+ | NVFP4 W4A16 safe default |
| **Mac** | M3 Max / M4 Max | 36–128 GB unified | N/A | Q4_K_M stable |
| **边缘/桌面** | DGX Spark (GB10) | 119 GB unified | 12.8+ | pending validation |

### 2.2 不支持

- **RTX 4090 / Ada**：无 FP4 tensor cores
- **A100 / H100 / Hopper**：无 native FP4 MMA
- **Apple Silicon Intel 版**：无 Metal Performance Shader 优化
- **原生 Windows**：仅 WSL2/Docker beta

---

## 3. 性能指标（R6000 实测）

### 3.1 与 llama.cpp Q4_K_M 对比

| 指标 | llama.cpp Q4_K_M (parallel=1) | Lynn Engine NVFP4 W4A16 | 比率 |
|---|---|---|---|
| 模型大小 | 5.49 GB | 8.25 GB | 1.50× |
| 单流 decode TPS | ~165 | ~62 | 0.38× |
| 首 token 延迟 (4K) | ~0.3 s | ~0.3 s | ~1× |
| 并发 batch 8 TPS | 166 (无收益) | — (单 prompt) | — |

> **说明**：NVFP4 当前速度低于 Q4_K_M，因为 Lynn Engine 的 dense FFN native FP4 kernel 尚未落地。P155+ 已定位瓶颈，后续更新有望显著缩小差距。

### 3.2 质量指标

| 指标 | BF16 | NVFP4 W4A16 | Q4_K_M | 来源 |
|---|---|---|---|---|
| MMLU (500-shot) | — | — | — | 待跑 |
| GPQA Diamond | — | — | — | 待跑 |
| 6-prompt greedy exact | ✅ | ✅ | — | P184 convstrict gate |
| 70-prompt structured pass | — | 90% | — | P196 W4A8 gate |

---

## 4. 分发与安装

### 4.1 快速开始

```bash
# 1. 安装 Lynn Engine CLI
curl -fsSL https://engine.merkyorlynn.com/install.sh | bash

# 2. 下载模型
lynn-engine pull qwen35-9b-dense-w4a16

# 3. 启动服务
lynn-engine serve --model qwen35-9b-dense-w4a16

# 4. 配置 agent
lynn-engine agents setup
```

### 4.2 Docker（推荐云环境）

```bash
docker run --rm --gpus all --ipc=host \
  -p 127.0.0.1:18099:18099 \
  -v ~/.lynn-engine/models:/models \
  dl.merkyorlynn.com/docker/lynn-engine:latest \
  serve --model qwen35-9b-dense-w4a16
```

### 4.3 域名分工

| 域名 | 用途 |
|---|---|
| `engine.merkyorlynn.com` | 文档、安装脚本、模型卡、API 参考 |
| `dl.merkyorlynn.com` | 模型 bundle、Docker 镜像、wheel、checksums |
| `mirror.merkyorlynn.com` | 国内镜像（阿里云 OSS） |

---

## 5. 限制与注意事项

1. **单 prompt only**：Lynn Engine 不做 batching / PagedAttention。并发场景请用 llama.cpp Q4_K_M。
2. **Greedy only**：首期 release 只支持 temperature=0（top-1）。采样（temperature>0）在 roadmap 中。
3. **Tool-call 未实现**：JSON mode 可用，但结构化 tool_call parser 在 Phase 4。
4. **Streaming SSE stub**：代码中有占位，但非生产级。
5. **32K context 未验证**：4K/16K 已通过；32K 列入 RC gate。

---

## 6. 引用报告

- `reports/qwen35_9b/r6000_qwen35_9b_q4km_cuda_baseline_20260519_1731.md`
- `reports/qwen35_9b/r6000_qwen35_9b_q4km_cuda_baseline_20260519_1732.md`
- `reports/qwen35_9b/p150_qwen35_9b_nvfp4_linear_graph_summary_20260519_120000_convstrict.json`
- `reports/qwen35_9b/p183_qwen35_9b_nvfp4_exact_fast_isolation_20260519_115751.json`
- `reports/qwen35_9b/p184_qwen35_9b_nvfp4_convstrict_exact_gate_20260519_120255.json`
- `reports/qwen35_9b/p196_qwen35_9b_w4a8_structured_content_gate_20260519_1718_p196_chat70.json`

---

## 7. 修订历史

| 日期 | 变更 |
|---|---|
| 2026-05-19 | 初版模型卡草案，对应 release matrix v2 |
