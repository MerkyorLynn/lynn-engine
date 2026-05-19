# Qwen3.5-9B 首发 QA Status · 2026-05-19

**Generated:** 2026-05-19
**Branch:** `deepseek/qwen35-9b-release-qa-checklist-20260519`
**Overall decision:** `PENDING_QA`

---

## Decision

QA 尚未执行。本文件随 QA 推进持续更新。每条检查项的状态与备注应反映最近一次执行的结果。

**锁定清单:** `docs/QWEN35_9B_RELEASE_QA_CHECKLIST_20260519.md`

---

## Track A: Mac Q4_K_M (llama.cpp)

**默认端口:** `18099`
**模型路径:** `~/Models/Lynn/Qwen3.5-9B/q4_k_m/Qwen3.5-9B-Q4_K_M.gguf`
**QA 环境:** macOS Apple Silicon

### A.1 下载 & Checksum

| # | 检查项 | 状态 | 证据 / 备注 |
|---|--------|------|-------------|
| A.1.1 | GGUF 文件存在 | PENDING | |
| A.1.2 | GGUF 文件大小 (~5.5 GiB) | PENDING | |
| A.1.3 | SHA256 checksum 验证 | PENDING | `checksums.sha256` 尚未发布（TODO） |
| A.1.4 | GGUF 元信息可读 | PENDING | |

### A.2 llama.cpp Server 启动

| # | 检查项 | 状态 | 证据 / 备注 |
|---|--------|------|-------------|
| A.2.1 | llama-server binary 存在 | PENDING | |
| A.2.2 | Server 启动 | PENDING | port 18099, model qwen35-9b-q4km |
| A.2.3 | /health 返回 200 | PENDING | |
| A.2.4 | 启动无 crash | PENDING | |

### A.3 /v1/models 端点

| # | 检查项 | 状态 | 证据 / 备注 |
|---|--------|------|-------------|
| A.3.1 | /v1/models 返回 200 | PENDING | |
| A.3.2 | 响应体含 `qwen35-9b-q4km` | PENDING | |
| A.3.3 | 响应体 `object: list` | PENDING | |

### A.4 普通 Chat 补全

| # | 检查项 | 状态 | 证据 / 备注 |
|---|--------|------|-------------|
| A.4.1 | 单轮 chat 返回非空 | PENDING | |
| A.4.2 | 单轮中文 chat | PENDING | |
| A.4.3 | 多轮上下文保持 | PENDING | |
| A.4.4 | 空消息不崩溃 | PENDING | |

### A.5 JSON 输出

| # | 检查项 | 状态 | 证据 / 备注 |
|---|--------|------|-------------|
| A.5.1 | JSON response_format 返回合法 JSON | PENDING | |
| A.5.2 | JSON 可被 python json.loads | PENDING | |
| A.5.3 | 代码块 JSON 输出 | PENDING | |
| A.5.4 | Structured output smoke script | PENDING | `scripts/local_qwen35_9b_q4km_smoke.sh` |

### A.6 32K Context Smoke

| # | 检查项 | 状态 | 证据 / 备注 |
|---|--------|------|-------------|
| A.6.1 | 32K 长提示不崩溃 | PENDING | |
| A.6.2 | 32K 补全不空洞 | PENDING | |
| A.6.3 | 32K 后继续对话 | PENDING | |
| A.6.4 | 32K decode TPS ≥ 20 | PENDING | 记录 wall-clock TPS |
| A.6.5 | parallel=1 32K 确认 | PENDING | 参考 `reports/qwen35_9b/q4km_long32k_parallel1_20260519_0128.md` |

### A.7 压力边界

| # | 检查项 | 状态 | 证据 / 备注 |
|---|--------|------|-------------|
| A.7.1 | 并发 8 请求不崩溃 | PENDING | |
| A.7.2 | max_tokens=0 不崩溃 | PENDING | |
| A.7.3 | 超长 max_tokens 截断 | PENDING | |

**Track A 小计:** 0 PASS / 0 FAIL / 22 PENDING

---

## Track B: NVIDIA Lynn NVFP4 W4A16

**默认端口:** `18191`
**模型路径:** `~/Models/Lynn/Qwen3.5-9B/nvfp4-w4a16/`
**QA 环境:** Linux + Blackwell GPU (R6000)

### B.1 模型包存在性

| # | 检查项 | 状态 | 证据 / 备注 |
|---|--------|------|-------------|
| B.1.1 | 模型目录存在 | PENDING | |
| B.1.2 | config.json 存在 | PENDING | |
| B.1.3 | tokenizer.json 存在 | PENDING | |
| B.1.4 | tokenizer_config.json 存在 | PENDING | |
| B.1.5 | model.safetensors.index.json 存在 | PENDING | |
| B.1.6 | safetensors shard 存在 | PENDING | |
| B.1.7 | 总大小 ~8.3 GiB | PENDING | |

### B.2 Manifest 校验

| # | 检查项 | 状态 | 证据 / 备注 |
|---|--------|------|-------------|
| B.2.1 | lynn_quant_manifest.json 存在 | PENDING | |
| B.2.2 | manifest `quantized_count` | PENDING | |
| B.2.3 | manifest `output_shards` | PENDING | |
| B.2.4 | shard count 匹配 | PENDING | |
| B.2.5 | manifest `pack_elapsed_seconds` | PENDING | |
| B.2.6 | SHA256 checksum 验证 | PENDING | `checksums.sha256` 尚未发布（TODO） |

### B.3 Loader Smoke

| # | 检查项 | 状态 | 证据 / 备注 |
|---|--------|------|-------------|
| B.3.1 | Server 启动 (W4A16, port 18191) | PENDING | |
| B.3.2 | /health 返回 200 | PENDING | |
| B.3.3 | /v1/models 返回 200 | PENDING | |
| B.3.4 | load_seconds < 60s | PENDING | |
| B.3.5 | 无 CUDA OOM | PENDING | |
| B.3.6 | GPU 温度正常 | PENDING | |

### B.4 Chat Smoke

| # | 检查项 | 状态 | 证据 / 备注 |
|---|--------|------|-------------|
| B.4.1 | 单轮 chat 返回非空 | PENDING | |
| B.4.2 | 单轮中文 chat | PENDING | |
| B.4.3 | 多轮上下文保持 | PENDING | |

### B.5 Structured Gate (P196)

| # | 检查项 | 状态 | 证据 / 备注 |
|---|--------|------|-------------|
| B.5.1 | P196 report 存在 | PENDING | `reports/qwen35_9b/P196_W4A8_STRUCTURED_CONTENT_GATE_20260519.md` |
| B.5.2 | W4A16 exact-match all-pass | PENDING | |
| B.5.3 | W4A16 codex prompt 格式合法 | PENDING | |
| B.5.4 | W4A16 JSON structured gate 通过 | PENDING | |

### B.6 速度基线 (P25 Probe)

| # | 检查项 | 状态 | 证据 / 备注 |
|---|--------|------|-------------|
| B.6.1 | P25 TPS (128 tokens) | PENDING | |
| B.6.2 | P25 TPS (512 tokens) | PENDING | |
| B.6.3 | TPS ≥ baseline [55, 70] | PENDING | |
| B.6.4 | TPS 波动 ≤ 5% | PENDING | |

### B.7 失败回滚: W4A16 Safe Default

| # | 检查项 | 状态 | 证据 / 备注 |
|---|--------|------|-------------|
| B.7.1 | W4A16 永远是 safe default | PENDING | |
| B.7.2 | W4A16 → W4A8 不可自动升级 | PENDING | |
| B.7.3 | W4A16 回滚路径验证 | PENDING | |
| B.7.4 | W4A16 无 structured 退化 | PENDING | |

**Track B 小计:** 0 PASS / 0 FAIL / 23 PENDING

---

## Track C: Experimental — W4A8 / FP4×FP8 Resident

**状态:** **experimental**。不作为首发 stable 轨道。

### C.1 门控要求

| # | 检查项 | 状态 | 证据 / 备注 |
|---|--------|------|-------------|
| C.1.1 | P197 drift probe 通过 | PENDING | `reports/qwen35_9b/p197_*_drift_*.json` |
| C.1.2 | P190 true FP8 resident gate 通过 | PENDING | `reports/qwen35_9b/p190_qwen35_9b_true_fp8_resident_gate_*.json` → `TRUE_FP8_RESIDENT_EXACT` |
| C.1.3 | P196 W4A8 column 通过 | PENDING | |
| C.1.4 | P193 boundary admission GREEN | PENDING | `python3 benchmarks/p193_qwen35_9b_native_boundary_admission.py` |

### C.2 Experimental 定位声明

| # | 检查项 | 状态 | 证据 / 备注 |
|---|--------|------|-------------|
| C.2.1 | 本文件标记 experimental | PASS | Track C 标题含 **experimental** |
| C.2.2 | Release Matrix 标记 experimental | PENDING | `docs/QWEN35_9B_RELEASE_MATRIX_20260519.md` |
| C.2.3 | Model Card 区分 stable vs experimental | PENDING | |
| C.2.4 | 用户文档不含 W4A8 安装步骤 | PENDING | |

### C.3 实验性运行时

| # | 检查项 | 状态 | 证据 / 备注 |
|---|--------|------|-------------|
| C.3.1 | W4A8 flag 存在但需显式开启 | PENDING | |
| C.3.2 | W4A8 加载不 crash (dev) | PENDING | |
| C.3.3 | W4A8 decode TPS speedup | PENDING | |

**Track C 小计:** 1 PASS / 0 FAIL / 10 PENDING

---

## Overall Summary

| Track | Items | PASS | FAIL | PENDING | Decision |
|-------|-------|------|------|---------|----------|
| A: Mac Q4_K_M | 22 | 0 | 0 | 22 | PENDING_QA |
| B: NVIDIA NVFP4 W4A16 | 23 | 0 | 0 | 23 | PENDING_QA |
| C: Experimental W4A8 | 11 | 1 | 0 | 10 | PENDING_GATE (P197/P190) |
| **Total** | **56** | **1** | **0** | **55** | **PENDING_QA** |

---

## Blockers

- **Checksum**: `checksums.sha256` 尚未发布。A.1.3 和 B.2.6 依赖此文件。
- **P197/P190 gate**: W4A8 / FP4×FP8 resident 不是首发 stable 轨道。Track C 在不影响 A/B 的前提下可延后执行。
- **NVFP4 模型包**: 需确认 `lynn_quant_manifest.json` 和 safetensors shard 已就位。

---

## Related Reports

| Report | Path | Status |
|--------|------|--------|
| Release Matrix | `docs/QWEN35_9B_RELEASE_MATRIX_20260519.md` | 已存在 |
| Release Status (gate-level) | `docs/QWEN35_9B_RELEASE_STATUS_20260519.md` | 已存在 |
| Q4_K_M llama.cpp Baseline | `docs/QWEN35_9B_Q4KM_LLAMA_BASELINE_20260519.md` | 已存在 |
| NVFP4 R6000 Pipeline | `docs/QWEN35_9B_R6000_NVFP4_PIPELINE_20260518.md` | 已存在 |
| Install Quickstart | `docs/QWEN35_9B_INSTALL_QUICKSTART_20260519.md` | 已存在 |
| P196 W4A8 Structured Gate | `reports/qwen35_9b/P196_W4A8_STRUCTURED_CONTENT_GATE_20260519.md` | 已存在 |
| P190 True FP8 Resident Gate | `reports/qwen35_9b/p190_qwen35_9b_true_fp8_resident_gate_*.json` | PENDING: true resident build/preflight still blocked; do not promote W4A8 |
| P197 Drift Probe | `reports/qwen35_9b/p197_*_drift_*.json` | AMBER seen on fake-W4A8 path; true resident still requires P190/P198 |
| Native Boundary Admission | `docs/QWEN35_9B_NATIVE_BOUNDARY_ADMISSION_20260519.md` | 已存在 |

---

*QA 状态文件，随每次执行更新。最后更新: 2026-05-19 (initial).*
