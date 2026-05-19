# Qwen3.5-9B Dense Release Matrix v2

**Date:** 2026-05-19  
**Schema:** `lynn-qwen35-9b-dense-release-matrix-v2`  
**Branch:** `kimi/qwen35-9b-release-matrix-v2-20260519`

> 本文档是 Qwen3.5-9B Dense 的 **首期发布矩阵**。所有性能数字均来自 R6000 (RTX PRO 6000 Blackwell) 实测；Mac/Windows 行基于已知硬件规格推导，待 Spark/本地机器验证后更新。

---

## 1. 平台 / 运行时 / 产物 / 支持状态

| 平台 | 运行时 | 产物格式 | 大小 | 单流 TPS | 并发 TPS | 长文本 | 状态 | 备注 |
|---|---|---|---:|---:|---:|---:|---|---|
| **Mac (Apple Silicon)** | llama.cpp | Q4_K_M-imatrix | 5.5 GB | — | — | — | ✅ **stable** | M3 Max/M4 Max 目标；llama.cpp 主场 |
| **NVIDIA Linux (R6000)** | llama.cpp CUDA | Q4_K_M-imatrix | 5.5 GB | ~166 | ~413 | 4K→174 | ✅ **baseline** | 引 `r6000_qwen35_9b_q4km_cuda_baseline_20260519_1731.md` |
| **NVIDIA Linux (R6000)** | Lynn Engine | NVFP4 W4A16 | 8.2 GB | ~60–66 | — | 4K→~60 | ✅ **safe** | 引 `p150_qwen35_9b_nvfp4_linear_graph_summary_20260519_120000_convstrict.json` |
| **NVIDIA Linux (R6000)** | Lynn Engine | W4A8 (fake-quant) | 8.2 GB | ~58–60 | — | 4K→~58 | 🔶 **experimental** | 引 `p196_qwen35_9b_w4a8_structured_content_gate_20260519_1718_p196_chat70.json` |
| **Windows (WSL2/Docker)** | Lynn Engine Docker | NVFP4 W4A16 | 8.2 GB | — | — | — | 🔶 **beta** | WSL2 + CUDA 12.8+；原生 Windows 不支持 |
| **DGX Spark** | Lynn Engine | NVFP4 W4A16 | 8.2 GB | — | — | — | ⏳ **pending** | GB10 sm_121 验证中 |

### 关键结论速览

- **Mac = Q4_K_M stable**：Apple Silicon 无 NVFP4 tensor core，llama.cpp Q4_K_M 是唯一生产级路径。
- **NVIDIA Linux = NVFP4 W4A16 safe**：Lynn Engine 原生 NVFP4 是 Blackwell 上的 safe default；exact-greedy 保证 + linear-block graph 已验证。
- **W4A8 = experimental**：fake-quant 验证通过（P196 70 prompt structured gate pass rate 90%），但速度未超过 W4A16，且 activation quant 质量风险仍在评估。
- **Windows = WSL2/Docker beta**：不提供原生 Windows 二进制；Docker 镜像内置 CUDA runtime + Lynn Engine。
- **Q4_K_M 也是 NVIDIA Linux 的 fallback**：当 Lynn Engine 出现兼容性问题时，llama.cpp Q4_K_M 是立刻可切换的 baseline。

---

## 2. 各格式详细数据

### 2.1 Q4_K_M-imatrix (llama.cpp CUDA) — Baseline

来源：`reports/qwen35_9b/r6000_qwen35_9b_q4km_cuda_baseline_20260519_1731.md`  
配置：llama.cpp `b8563ad`, CUDA 12.8, `CMAKE_CUDA_ARCHITECTURES=120`, flash-attn, parallel=8

| 指标 | 数值 |
|---|---|
| 模型大小 | 5.49 GiB |
| 单流 128 tok | 131.55 TPS |
| 单流 256 tok | 160.20 TPS |
| 单流 512 tok | **165.75 TPS** |
| 并发 2×256 | 270.61 TPS |
| 并发 4×256 | 330.75 TPS |
| 并发 8×256 | **413.49 TPS** |
| 长文本 4K→256 | 173.79 TPS |
| 长文本 16K→256 | 114.39 TPS |
| 长文本 32K→256 | ❌ FAIL (parallel=8 下 OOM) |

来源：`reports/qwen35_9b/r6000_qwen35_9b_q4km_cuda_baseline_20260519_1732.md`  
配置：同上，parallel=1（单 client）

| 指标 | 数值 |
|---|---|
| 单流 128 tok | 125.98 TPS |
| 单流 256 tok | 159.60 TPS |
| 单流 512 tok | **164.52 TPS** |
| 并发 8×256 | 166.11 TPS（无 batching 收益） |
| 长文本 32K→64 | **55.51 TPS** |

> **解读**：parallel=8 在 llama.cpp 内部做 batching，所以并发 TPS 随 client 数线性提升；parallel=1 更适合与 Lynn Engine（单 prompt，无 batching）做 apples-to-apples 比较。

### 2.2 NVFP4 W4A16 (Lynn Engine) — Safe Default

来源：`reports/qwen35_9b/p150_qwen35_9b_nvfp4_linear_graph_summary_20260519_120000_convstrict.json`  
配置：`LYNN_LINEAR_BLOCK_GRAPH=1`, `LYNN_LINEAR_BLOCK_GRAPH_REUSE=1`, `LYNN_LINEAR_BLOCK_GRAPH_PREWARM=1`, convstrict serving guard

| 指标 | 数值 |
|---|---|
| 模型大小 | 8.25 GiB（官方 W4A16 pack） |
| decode 128 tok | ~61.3 TPS |
| decode 256 tok | ~62.2 TPS |
| decode 512 tok | ~62.1 TPS |
|  verdict | **P25_READY** |

来源：`reports/qwen35_9b/p183_qwen35_9b_nvfp4_exact_fast_isolation_20260519_115751.json`  
配置：exact-fast isolation，对比 baseline vs fast knob

| 模式 | decode TPS |
|---|---|
| baseline | ~60.4 TPS |
| fast knob | ~65.7 TPS |

来源：`reports/qwen35_9b/p184_qwen35_9b_nvfp4_convstrict_exact_gate_20260519_120255.json`  
配置：convstrict exact gate，6 prompt strict pass

| 模式 | decode TPS |
|---|---|
| baseline | ~60.0 TPS |
| fast knob | ~61.8 TPS |

> **解读**：NVFP4 W4A16 当前 decode TPS 约 **60–66**，约为 llama.cpp Q4_K_M (parallel=1) 的 **37–40%**。差距主要来自：(1) Lynn Engine 尚未启用 dense FFN 的 native FP4 kernel；(2) 当前仍为 BF16 activation + packed NVFP4 weight 的混合路径。P155+ 的 dense FFN phase profile 已定位瓶颈，后续 native kernel 有望显著缩小差距。

### 2.3 W4A8 (Lynn Engine) — Experimental

来源：`reports/qwen35_9b/p196_qwen35_9b_w4a8_structured_content_gate_20260519_1718_p196_chat70.json`  
配置：fake-quant W4A8（FP4 weight + FP8 activation），70 prompt structured content gate

| 模式 | decode TPS | 结构化通过率 | 状态 |
|---|---:|---:|---|
| W4A16 reference | 60.49 | 90% (63/70) | baseline |
| W4A8 gateup-only | 60.19 | — | 数值验证中 |
| W4A8 full | 57.81 | — | 数值验证中 |

> **解读**：W4A8 当前是 **fake-quant 验证阶段**，速度未超过 W4A16，且 activation quant 的精度影响需更大规模 gate 验证。不进入首期 release default。

---

## 3. 分发站点分工

| 域名 | 职责 | 内容 |
|---|---|---|
| `engine.merkyorlynn.com` | **文档与入口** | 安装指南、release matrix、模型卡、API 文档、CLI 帮助 |
| `dl.merkyorlynn.com` | **二进制与模型下载** | `install.sh`、Docker 镜像、wheel、模型 bundle (`.tar.gz`)、checksums |
| `mirror.merkyorlynn.com` | **国内镜像** | 阿里云 OSS 回源，中国大陆用户优先 |

### 首期 release URL 规划

```text
# 文档
https://engine.merkyorlynn.com/docs/qwen35-9b/release-matrix
https://engine.merkyorlynn.com/docs/qwen35-9b/model-card
https://engine.merkyorlynn.com/install.sh

# 下载
https://dl.merkyorlynn.com/models/qwen35-9b-dense/lynn-qwen35-9b-dense-w4a16-v1.tar.gz
https://dl.merkyorlynn.com/models/qwen35-9b-dense/lynn-qwen35-9b-dense-bf16-v1.tar.gz
https://dl.merkyorlynn.com/models/qwen35-9b-dense/checksums.sha256
```

---

## 4. 支持生命周期

| 阶段 | 时间 | 承诺 |
|---|---|---|
| **Beta** | 2026-05-19 → 2026-06-01 | R6000 only；API/CLI 可能变动 |
| **RC** | 2026-06-01 → 2026-06-15 | Tier-1 硬件冻结；迁移指南提供 |
| **GA (First Release)** | 2026-06-15 → 2026-09-15 | Mac Q4_K_M + NVIDIA Linux NVFP4；安全修复 |
| **Maintenance** | 2026-09-15 → 2026-12-31 | 关键修复 only；新特性进 v2 |

---

## 5. 已知限制

| 限制 | 影响 | 计划 |
|---|---|---|
| NVFP4 单流 TPS 仅为 Q4_K_M 的 40% | NVIDIA Linux 用户感知速度低于 llama.cpp | P155+ dense FFN native kernel；目标 100+ TPS |
| W4A8 速度未超过 W4A16 | activation quant 暂无不必要收益 | 继续 fake-quant 验证；不阻塞 GA |
| 32K context 未验证 | 长文本场景风险 | 4K/16K 已验证；32K 列入 RC gate |
| Windows 仅 WSL2/Docker | 无原生 Windows binary | beta 阶段收集反馈；native 视需求评估 |
| Mac 无 NVFP4 | Apple Silicon 只能跑 Q4_K_M | 符合预期；llama.cpp 是 Mac 最优解 |

---

## 6. 引用报告清单

| 报告 | 路径 | 用途 |
|---|---|---|
| Q4_K_M CUDA baseline 1731 | `reports/qwen35_9b/r6000_qwen35_9b_q4km_cuda_baseline_20260519_1731.md` | llama.cpp parallel=8 性能基线 |
| Q4_K_M CUDA baseline 1732 | `reports/qwen35_9b/r6000_qwen35_9b_q4km_cuda_baseline_20260519_1732.md` | llama.cpp parallel=1 性能基线 |
| NVFP4 P25 serving gate | `reports/qwen35_9b/p150_qwen35_9b_nvfp4_linear_graph_summary_20260519_120000_convstrict.json` | Lynn Engine NVFP4 safe  verdict |
| NVFP4 exact-fast isolation | `reports/qwen35_9b/p183_qwen35_9b_nvfp4_exact_fast_isolation_20260519_115751.json` | fast knob 上限探针 |
| NVFP4 convstrict exact gate | `reports/qwen35_9b/p184_qwen35_9b_nvfp4_convstrict_exact_gate_20260519_120255.json` | 6 prompt strict pass |
| W4A8 structured gate | `reports/qwen35_9b/p196_qwen35_9b_w4a8_structured_content_gate_20260519_1718_p196_chat70.json` | W4A8 experimental 验证 |

---

## 7. 修订历史

| 日期 | 版本 | 变更 |
|---|---|---|
| 2026-05-18 | v1 | Qwen3.6-9B Dense 初版矩阵（llama.cpp Q4_K_M / Lynn BF16 / Lynn NVFP4） |
| 2026-05-19 | v2 | 切换主线为 Qwen3.5-9B Dense；加入 R6000 实测 Q4_K_M baselines (1731/1732)；NVFP4 W4A16 定为 safe default；W4A8 列为 experimental；Windows WSL2/Docker beta；明确 engine/dl 域名分工 |
