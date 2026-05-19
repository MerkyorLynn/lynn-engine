# Qwen3.5-9B 首发 QA Checklist · 2026-05-19

**Branch:** `deepseek/qwen35-9b-release-qa-checklist-20260519`
**Scope:** 用户可验收的 QA 检查清单。覆盖 Mac Q4_K_M 和 NVIDIA NVFP4 W4A16 两条稳定轨。
**不覆盖:** CUDA kernel 修改、`engine/csrc/`、`server/`、`benchmarks/` 内的代码变更。

---

## Track Overview

| Track | 硬件 | 运行时 | 制品 | 状态 |
|-------|------|--------|------|------|
| Mac Q4_K_M | Apple Silicon / CPU / 通用 CUDA | llama.cpp | Q4_K_M GGUF | stable |
| NVIDIA NVFP4 W4A16 | Blackwell GPU | Lynn Engine | Lynn-native NVFP4 W4A16 | safe default |
| NVIDIA W4A8 / FP4×FP8 | Blackwell GPU (research) | Lynn Engine | Lynn-native FP4×FP8 resident | **experimental** |

---

## Track A: Mac Q4_K_M (llama.cpp)

**目标环境:** macOS Apple Silicon。llama.cpp 通过 Homebrew 或源码编译。
**默认端口:** `18099`
**模型路径:** `~/Models/Lynn/Qwen3.5-9B/q4_k_m/Qwen3.5-9B-Q4_K_M.gguf`

### A.1 下载 & Checksum

| # | 检查项 | 命令 | PASS/FAIL/PENDING |
|---|--------|------|-------------------|
| A.1.1 | GGUF 文件存在 | `ls -lh ~/Models/Lynn/Qwen3.5-9B/q4_k_m/Qwen3.5-9B-Q4_K_M.gguf` | PENDING |
| A.1.2 | GGUF 文件大小 (预期 ~5.5 GiB) | `stat -f%z ~/Models/Lynn/Qwen3.5-9B/q4_k_m/Qwen3.5-9B-Q4_K_M.gguf` | PENDING |
| A.1.3 | SHA256 checksum 验证 | `shasum -a 256 -c ~/Models/Lynn/Qwen3.5-9B/checksums.sha256` | PENDING |
| A.1.4 | GGUF 元信息可读 (model name / quant) | `llama-export --help 2>/dev/null; head -c 1024 ~/Models/Lynn/Qwen3.5-9B/q4_k_m/Qwen3.5-9B-Q4_K_M.gguf \| strings \| grep -i qwen` | PENDING |

### A.2 llama.cpp Server 启动

| # | 检查项 | 命令 | PASS/FAIL/PENDING |
|---|--------|------|-------------------|
| A.2.1 | llama-server binary 存在 | `which llama-server \|\| ls ~/llama.cpp/build/bin/llama-server` | PENDING |
| A.2.2 | Server 启动 (port 18099, model qwen35-9b-q4km) | `bash scripts/local_qwen35_9b_q4km_llamacpp_server.sh --host 127.0.0.1 --port 18099 --model-name qwen35-9b-q4km` | PENDING |
| A.2.3 | Server 进程存活 | `curl -fsS -o /dev/null -w '%{http_code}' http://127.0.0.1:18099/health` 返回 `200` | PENDING |
| A.2.4 | 启动无 OOM / 无 crash 报错 | 检查 server stderr，无 `CUDA error` / `ggml_metal_*` fatal | PENDING |

### A.3 /v1/models 端点

| # | 检查项 | 命令 | PASS/FAIL/PENDING |
|---|--------|------|-------------------|
| A.3.1 | /v1/models 返回 200 | `curl -fsS -o /dev/null -w '%{http_code}' http://127.0.0.1:18099/v1/models` | PENDING |
| A.3.2 | 响应体包含 `qwen35-9b-q4km` | `curl -fsS http://127.0.0.1:18099/v1/models \| python3 -c "import sys,json; d=json.load(sys.stdin); print([m['id'] for m in d['data']])"` | PENDING |
| A.3.3 | 响应体包含 `object: list` | `curl -fsS http://127.0.0.1:18099/v1/models \| python3 -c "import sys,json; d=json.load(sys.stdin); print(d['object'])"` | PENDING |

### A.4 普通 Chat 补全

| # | 检查项 | 命令 | PASS/FAIL/PENDING |
|---|--------|------|-------------------|
| A.4.1 | 单轮 chat 返回非空文本 | `curl -fsS http://127.0.0.1:18099/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"qwen35-9b-q4km","messages":[{"role":"user","content":"Say OK in one short sentence."}],"temperature":0,"max_tokens":32}'` | PENDING |
| A.4.2 | 单轮中文 chat 返回中文 | `curl -fsS http://127.0.0.1:18099/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"qwen35-9b-q4km","messages":[{"role":"user","content":"用一句中文说你好。"}],"temperature":0,"max_tokens":32}'` | PENDING |
| A.4.3 | 多轮对话保持上下文 | 两轮 curl：第一轮说 "My name is Bob"，第二轮问 "What is my name?"，回答含 "Bob" | PENDING |
| A.4.4 | 空消息不崩溃 | `curl -fsS http://127.0.0.1:18099/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"qwen35-9b-q4km","messages":[{"role":"user","content":""}],"temperature":0,"max_tokens":16}'` | PENDING |

### A.5 JSON 输出

| # | 检查项 | 命令 | PASS/FAIL/PENDING |
|---|--------|------|-------------------|
| A.5.1 | JSON response_format 返回合法 JSON | `curl -fsS http://127.0.0.1:18099/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"qwen35-9b-q4km","messages":[{"role":"user","content":"Return a JSON object with keys: name, age, city. Use your best guess."}],"response_format":{"type":"json_object"},"temperature":0,"max_tokens":128}'` | PENDING |
| A.5.2 | JSON 可被 python json.loads 解析 | 对 A.5.1 输出 pipe 到 `python3 -c "import sys,json; d=json.load(sys.stdin); json.loads(d['choices'][0]['message']['content'])"` | PENDING |
| A.5.3 | 代码块 JSON 输出 (无 response_format) | 提示词 "输出一个 JSON 对象，包含键 name 和 version"，检查是否可解析 | PENDING |
| A.5.4 | Structured output smoke (via smoke script) | `bash scripts/local_qwen35_9b_q4km_smoke.sh --base-url http://127.0.0.1:18099/v1 --model qwen35-9b-q4km` | PENDING |

### A.6 32K Context Smoke

| # | 检查项 | 命令 | PASS/FAIL/PENDING |
|---|--------|------|-------------------|
| A.6.1 | 32K 长提示不崩溃 | 发送 ~32000 chars 的重复文本作为 prompt，请求 128 token 补全 | PENDING |
| A.6.2 | 32K 补全内容不空洞 | A.6.1 的补全内容应包含至少一个完整句子（非纯空白/乱码） | PENDING |
| A.6.3 | 32K 后继续对话正常 | 在 A.6.1 对话中追加一轮短消息，返回连贯回答 | PENDING |
| A.6.4 | 32K decode TPS 不低于 20 (Mac Silicon) | 记录 wall-clock TPS，不低于阈值 | PENDING |
| A.6.5 | parallel=1 32K 确认 (非 slot partition 问题) | 使用 `PARALLEL=1` 重新测试 32K，参考 `reports/qwen35_9b/q4km_long32k_parallel1_20260519_0128.md` | PENDING |

### A.7 压力边界

| # | 检查项 | 命令 | PASS/FAIL/PENDING |
|---|--------|------|-------------------|
| A.7.1 | 并发 8 请求不崩溃 | 同时发送 8 个短 chat 请求 (parallel=8)，全部返回 200 | PENDING |
| A.7.2 | max_tokens=0 不崩溃 | `max_tokens: 0` 应返回空 completion 而非 crash | PENDING |
| A.7.3 | 超长 max_tokens 截断行为 | `max_tokens: 999999` 应正常截断而非 OOM | PENDING |

---

## Track B: NVIDIA Lynn NVFP4 W4A16

**目标环境:** Linux + NVIDIA Blackwell GPU (R6000 / B200)。Lynn Engine repo 已 clone，CUDA/PyTorch 已安装。
**默认端口:** `18191`
**模型路径:** `~/Models/Lynn/Qwen3.5-9B/nvfp4-w4a16/`

### B.1 模型包存在性

| # | 检查项 | 命令 | PASS/FAIL/PENDING |
|---|--------|------|-------------------|
| B.1.1 | 模型目录存在 | `ls -d ~/Models/Lynn/Qwen3.5-9B/nvfp4-w4a16/` | PENDING |
| B.1.2 | config.json 存在 | `ls -lh ~/Models/Lynn/Qwen3.5-9B/nvfp4-w4a16/config.json` | PENDING |
| B.1.3 | tokenizer.json 存在 | `ls -lh ~/Models/Lynn/Qwen3.5-9B/nvfp4-w4a16/tokenizer.json` | PENDING |
| B.1.4 | tokenizer_config.json 存在 | `ls -lh ~/Models/Lynn/Qwen3.5-9B/nvfp4-w4a16/tokenizer_config.json` | PENDING |
| B.1.5 | model.safetensors.index.json 存在 | `ls -lh ~/Models/Lynn/Qwen3.5-9B/nvfp4-w4a16/model.safetensors.index.json` | PENDING |
| B.1.6 | 至少一个 safetensors shard 存在 | `ls ~/Models/Lynn/Qwen3.5-9B/nvfp4-w4a16/model-00001-of-*.safetensors` | PENDING |
| B.1.7 | 总大小 ~8.3 GiB (W4A16 NVFP4) | `du -sh ~/Models/Lynn/Qwen3.5-9B/nvfp4-w4a16/` | PENDING |

### B.2 Manifest 校验

| # | 检查项 | 命令 | PASS/FAIL/PENDING |
|---|--------|------|-------------------|
| B.2.1 | lynn_quant_manifest.json 存在 | `ls -lh ~/Models/Lynn/Qwen3.5-9B/nvfp4-w4a16/lynn_quant_manifest.json` | PENDING |
| B.2.2 | manifest 包含 `quantized_count` | `python3 -c "import json; d=json.load(open('$HOME/Models/Lynn/Qwen3.5-9B/nvfp4-w4a16/lynn_quant_manifest.json')); print('quantized_count:', d.get('quantized_count','MISSING'))"` | PENDING |
| B.2.3 | manifest 包含 `output_shards` | `python3 -c "import json; d=json.load(open('$HOME/Models/Lynn/Qwen3.5-9B/nvfp4-w4a16/lynn_quant_manifest.json')); print('output_shards:', len(d.get('output_shards',[])))"` | PENDING |
| B.2.4 | manifest shard count 匹配实际文件数 | 对比 manifest `output_shards` 数量与 `model-*.safetensors` 实际文件数 | PENDING |
| B.2.5 | manifest 包含 `pack_elapsed_seconds` | `python3 -c "import json; d=json.load(open('$HOME/Models/Lynn/Qwen3.5-9B/nvfp4-w4a16/lynn_quant_manifest.json')); print('pack_elapsed:', d.get('pack_elapsed_seconds','MISSING'))"` | PENDING |
| B.2.6 | SHA256 checksum 验证 | `shasum -a 256 -c ~/Models/Lynn/Qwen3.5-9B/checksums.sha256` | PENDING |

### B.3 Loader Smoke

| # | 检查项 | 命令 | PASS/FAIL/PENDING |
|---|--------|------|-------------------|
| B.3.1 | Server 启动 (W4A16 safe profile, port 18191) | `cd /path/to/lynn-engine && export MODEL_DIR="$HOME/Models/Lynn/Qwen3.5-9B/nvfp4-w4a16" && python -m server.openai_http --model "$MODEL_DIR" --served-name qwen35-9b-nvfp4-w4a16 --host 127.0.0.1 --port 18191 --dtype bfloat16` | PENDING |
| B.3.2 | /health 返回 200 | `curl -fsS -o /dev/null -w '%{http_code}' http://127.0.0.1:18191/health` | PENDING |
| B.3.3 | /v1/models 返回 200 且含模型名 | `curl -fsS http://127.0.0.1:18191/v1/models \| python3 -c "import sys,json; d=json.load(sys.stdin); print([m['id'] for m in d['data']])"` | PENDING |
| B.3.4 | 模型加载时间记录 (load_seconds) | server stdout/stderr 中记录 load_seconds，参考值 < 60s | PENDING |
| B.3.5 | 加载无 CUDA OOM | `nvidia-smi` 检查显存占用合理 (~8.3 GiB 模型 + KV cache) | PENDING |
| B.3.6 | 加载后 GPU 温度正常 | `nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader` | PENDING |

### B.4 Chat Smoke

| # | 检查项 | 命令 | PASS/FAIL/PENDING |
|---|--------|------|-------------------|
| B.4.1 | 单轮 chat 返回非空 | `curl -fsS http://127.0.0.1:18191/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"qwen35-9b-nvfp4-w4a16","messages":[{"role":"user","content":"Say OK in one short sentence."}],"temperature":0,"max_tokens":32}'` | PENDING |
| B.4.2 | 单轮中文 chat | `curl -fsS http://127.0.0.1:18191/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"qwen35-9b-nvfp4-w4a16","messages":[{"role":"user","content":"用一句中文说你好。"}],"temperature":0,"max_tokens":32}'` | PENDING |
| B.4.3 | 多轮上下文保持 | 两轮对话，验证能否记住用户名 | PENDING |

### B.5 Structured Gate (P196)

| # | 检查项 | 命令 | PASS/FAIL/PENDING |
|---|--------|------|-------------------|
| B.5.1 | P196 structured content gate 已运行 | `ls reports/qwen35_9b/P196_W4A8_STRUCTURED_CONTENT_GATE_20260519.md` | PENDING |
| B.5.2 | W4A16 参考线 exact-match 全部通过 | P196 report 中 W4A16 column exact_count == total | PENDING |
| B.5.3 | W4A16 codex prompt 输出格式合法 | 对 70-prompt structured hard set，W4A16 输出全部符合预期格式 | PENDING |
| B.5.4 | W4A16 JSON 结构化 gate 通过 | JSON response_format 测试的输出全部可解析 | PENDING |

### B.6 速度基线 (P25 Probe)

| # | 检查项 | 命令 | PASS/FAIL/PENDING |
|---|--------|------|-------------------|
| B.6.1 | P25 single decode TPS (128 tokens) | `python3 benchmarks/p25_server_decode_tps_probe.py --url http://127.0.0.1:18191/v1 --model qwen35-9b-nvfp4-w4a16 --chat --max-tokens 128 --runs 3 --out reports/qwen35_9b/local_nvfp4_w4a16_p25_smoke.json` | PENDING |
| B.6.2 | P25 single decode TPS (512 tokens) | 同 B.6.1，max-tokens 512 | PENDING |
| B.6.3 | P25 TPS ≥ baseline (58-62 TPS range) | decode TPS 落在 [55, 70] 区间（W4A16 safe profile 预期） | PENDING |
| B.6.4 | P25 TPS 波动 ≤ 5% (3 runs) | 3 次运行 TPS std / mean ≤ 0.05 | PENDING |

### B.7 失败回滚: W4A16 Safe Default

| # | 检查项 | 命令 | PASS/FAIL/PENDING |
|---|--------|------|-------------------|
| B.7.1 | W4A16 永远是 safe default | 无论 W4A8 门控是否通过，W4A16 路径必须可用 | PENDING |
| B.7.2 | W4A16 → W4A8 不可自动升级 | `--dtype bfloat16` 默认加载 W4A16；W4A8 需显式 flag | PENDING |
| B.7.3 | W4A16 回滚路径验证 | 当 W4A8 flag 失败时，fallback 到 W4A16 不 crash | PENDING |
| B.7.4 | W4A16 无 structured 退化 | W4A16 在所有 structured prompt 测试中通过（参考 P196） | PENDING |

---

## Track C: Experimental — W4A8 / FP4×FP8 Resident

**状态:** **experimental**。不作为首发 stable 轨道。必须通过 P197 (per-step token drift) 和 P190 (true FP8 resident gate) 后才能考虑升级。

### C.1 门控要求

| # | 检查项 | 命令 | PASS/FAIL/PENDING |
|---|--------|------|-------------------|
| C.1.1 | P197 per-step token drift probe 通过 | `ls reports/qwen35_9b/p197_*_drift_*.json` → 报告中 max_drift 在容差内 | PENDING |
| C.1.2 | P190 true FP8 resident gate 通过 | `ls reports/qwen35_9b/p190_qwen35_9b_true_fp8_resident_gate_*.json` → `verdict: TRUE_FP8_RESIDENT_EXACT` | PENDING |
| C.1.3 | P196 structured content gate W4A8 column 通过 | P196 report 中 W4A8 exact_count == total | PENDING |
| C.1.4 | P193 native boundary admission 通过 (GREEN) | `python3 benchmarks/p193_qwen35_9b_native_boundary_admission.py --report-dir reports/qwen35_9b/` → `GREEN_FIXTURE` | PENDING |

### C.2 Experimental 定位声明

| # | 检查项 | 命令 | PASS/FAIL/PENDING |
|---|--------|------|-------------------|
| C.2.1 | 文档标记 experimental | 本清单 Track C 标题含 **experimental** | PASS (by construction) |
| C.2.2 | Release Matrix 标记 experimental | `docs/QWEN35_9B_RELEASE_MATRIX_20260519.md` 中 W4A8 行状态为 `experimental` | PENDING |
| C.2.3 | Model Card 区分 stable vs experimental | Model Card 明确列出 stable tracks (Q4_K_M, NVFP4 W4A16) 和 experimental (W4A8) | PENDING |
| C.2.4 | 用户文档不含 W4A8 安装步骤 | Quickstart / Install 文档不将 W4A8 列为首发路径 | PENDING |

### C.3 实验性运行时 (仅供开发)

| # | 检查项 | 命令 | PASS/FAIL/PENDING |
|---|--------|------|-------------------|
| C.3.1 | W4A8 flag 存在但需显式开启 | `python -m server.openai_http --help \| grep -i w4a8` 确认 W4A8 不默认启用 | PENDING |
| C.3.2 | W4A8 加载不 crash (dev only) | 显式 W4A8 flag 加载模型，返回 /health 200 | PENDING |
| C.3.3 | W4A8 decode TPS 相对 W4A16 speedup | `python3 benchmarks/p25_server_decode_tps_probe.py` W4A8 vs W4A16 TPS 对比 | PENDING |

---

## Summary Matrix

| Track | Items Total | PASS | FAIL | PENDING | Blocker? |
|-------|-------------|------|------|---------|----------|
| A: Mac Q4_K_M | 22 | 0 | 0 | 22 | — |
| B: NVIDIA NVFP4 W4A16 | 23 | 0 | 0 | 23 | — |
| C: Experimental W4A8 | 11 | 1 | 0 | 10 | P197/P190 gate required |
| **Total** | **56** | **1** | **0** | **55** | — |

---

## How to Use

1. **QA 执行者** 按顺序执行每个 Track 的检查项。
2. 每完成一项，将 PASS/FAIL/PENDING 列更新为实际结果，并追加备注。
3. FAIL 项必须附带失败日志 / curl 输出。
4. 所有 PENDING 项在执行前保持 PENDING。
5. Track C 的 PENDING 项在 P197/P190 gate 通过前不需要执行。

**状态文件:** `reports/qwen35_9b/QWEN35_9B_RELEASE_QA_STATUS_20260519.md`
（该文件跟踪实际执行结果，本清单为模板。）

---

*本清单首版生成于 2026-05-19。随 QA 执行持续更新。*
