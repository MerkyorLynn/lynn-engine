# Spark sm_121 平台移植 / 验证笔记

> **Branch**: `spark/sm121-port`(this branch).  
> **Scope**: DGX Spark GB10 sm_121 平台兼容、启动脚本、eval harness、scalar_bridge ↔ native_prepared path 开关验证。  
> **所有性能数字 = Spark-only**,不代表 Lynn engine 真实性能(那是 R6000 主线 `feat/phase3.2-opt-in-paths` 的事)。

---

## 硬件 canonical

| 项 | DGX Spark |
|---|---|
| GPU | NVIDIA GB10(Grace ARM + Blackwell)|
| Compute capability | **sm_121** (12.1) — 跟 RTX PRO 6000 sm_120 binary 不兼容 |
| Unified memory | **119 GiB** CPU + GPU 共享(无独立显存)|
| Disk | 3.6T `/dev/nvme0n1p2` |
| CUDA(docker)| 13.0(`sglang dev-cu13`)/ 13.0.1(`llama.cpp full-cuda`)|
| PyTorch(docker)| 2.9.1+cu130 |
| Triton | 3.5.1(sm_121 JIT 已 verified)|
| vLLM 兼容 | ❌(strict cap check 8.0-12.0,sm_121 reject)|
| SGLang 兼容 | ✅(`sglang dev-cu13`,广 arch + PTX fallback)|
| llama.cpp 兼容 | ✅(`ghcr.io/ggml-org/llama.cpp:full-cuda`,含 `llama-imatrix` / `llama-quantize` / `llama-perplexity`)|

---

## 27B NVFP4 ground truth on Spark

**Model**: `Lynn-V4-Distill-Qwen-27B-A3B-NVFP4`(Lynn-native format `nvfp4_e2m1_rowwise_per_16`)  
**Location on Spark**: `/home/merkyor/models/lynn-27b-variable-recovery-step5000-nvfp4-final/`  
**File layout**: per-tensor `model-NNNNNN-{nvfp4,keep}.safetensors`(1026 files,~20G total)+ `lynn_quant_manifest.json`(per-tensor 量化决策)

### Reader 兼容性

| Format | Reader |
|---|---|
| Lynn-native `nvfp4_e2m1_rowwise_per_16`(this) | **只 Lynn engine** ✓(其他全 ❌)|
| `v8-RTN compressed-tensors`(V Flash 用过的格式) | SGLang dev-cu13 ✓ / llama.cpp ✗ |

### Server 启动

参考 `scripts/spark/run_27b_nvfp4_server.sh`,关键 env vars:
```
LYNN_PREFILL_WARMUP=1
LYNN_LINEAR_ATTN_RECURRENT_BACKEND=triton_fused_prepare
LYNN_MOE_IMPL=triton
LYNN_QK_NORM_ROPE_BACKEND=triton_pair
LYNN_RMSNORM_GATED_BACKEND=triton
LYNN_LINEAR_ATTN_INPROJ_FUSED=1
LYNN_LINEAR_BLOCK_GRAPH=1
LYNN_LINEAR_BLOCK_GRAPH_REUSE=1
LYNN_LINEAR_BLOCK_GRAPH_PREWARM=1
LYNN_LINEAR_STATE_UPDATE=inplace
# Native FP4 packed decode opt-in(当前 Spark sm_121 没触发,见下)
LYNN_PACKED_DECODE=1
LYNN_PACKED_DECODE_FULL_ATTN=1
LYNN_PACKED_DECODE_LINEAR_ATTN=1
LYNN_PACKED_DECODE_PREPARE_NATIVE=1
LYNN_PACKED_SHARED_EXPERT=1
LYNN_LINEAR_ATTN_INPROJ_FUSED_NATIVE_FP4=1
```

**加载时间**: ~152s on Spark sm_121(40 层 NVFP4 slow dequant + outside weights + prefill warmup + linear block graph prewarm)
**Memory footprint**: ~50-60G BF16 resident + ~15-20G workspace ≈ **75-80G total**(headroom 需 ≥ 30G,即 free ≥ 100G 才启)
**OOM history**: 2026-05-15 第一次启动 ~88G used baseline + 27B slow dequant 撞 119G 上限,**OOM 屠杀 sshd 卡 ~2h**,见 `feedback_mem_budget_before_big_process_20260515.md`(memory)

---

## 启动前 mem budget 铁律

```
启动前 free -h
  current_used + 75G(27B footprint) + 10G(safety) < 119G
即 free + buff/cache available 必 ≥ 85G
```

不满足:先 stop 占内存的 dockers(V Flash production / TTS / ASR / emotion2vec etc. 按需)。

---

## Round 1 ship-gate evidence(2026-05-15)

**Backend**: `scalar_bridge`(`packed decode aliases attached=190 skipped=0 backend=scalar_bridge native_prepared=0`)

| 项 | 结果 |
|---|---|
| 6/6 coherent smoke | ✅ PASS |
| Tool-call strict (1 prompt) | ✅ `get_weather(city="Tokyo")` correct |
| Long-ctx 5/5 (16K 中文输入) | ✅ Chinese coherent in 6.5-7.6s each |
| V9 holdout 8/8 | ✅ answered(数学/化学/生物/代码/物理 — quality 看上去合理) |
| **Single-stream TPS** | **23.88 TPS** mean / 23.89 median |
| **TTFT** | 0.52-0.68s |
| Steady decode | 42ms/token |
| Prefill | ~0.52s |

⚠️ **数字标识 reminder**:这些都是 **Spark sm_121 scalar_bridge** 数字,**不是** Lynn engine production stable(那是 R6000 sm_120 = 68-69 TPS / breakthrough = 103.44 TPS)。

---

## 已知 ship-blocker(不是 engine bug)

### `<think></think>` thinking-loop with greedy

**症状**: temperature=0 greedy + Qwen3 thinking 模型 → `<think>\n\n</think>\n\n<think>...` 死循环到 max_tokens,V8/V9 全 fail。  
**根因**: chat template 没默认 `enable_thinking=false` + stop_token 没列全。**跟 Lynn engine packed NVFP4 / native FP4 / scalar_bridge path 无关**。  
**Spark workaround**: eval harness 在 user message 前缀加 `/no_think `(参考 `v8v9_full_eval.py` chat() 函数)。  
**Production fix**(R6000 主线 / 模型卡侧): full no-think template + stop sequence 完整列表 + greedy 死循环 smoke gate。  
**memory**: `feedback_thinking_loop_is_template_not_engine_20260515.md`

### Temperature > 0 ValueError

**症状**: temperature=0.7 调用 → HTTP 500 `ValueError: Lynn engine MVP server currently supports greedy temperature=0 only`。  
**Spark workaround**: eval harness 全部用 `temperature=0.0`。  
**Production path**: R6000 主线决定是否给 Lynn engine 加 sampling support。

---

## Native FP4 packed decode path 未触发

Current server 状态 `backend=scalar_bridge native_prepared=0`,所有 `LYNN_PACKED_DECODE_*` 只挂了 alias `attached=190 skipped=0` 但实际 inference 仍走 scalar_bridge。  
**TODO**: 实验 `LYNN_PACKED_DECODE_BACKEND=...` value sweep(待查代码确定正确开关),让 Spark sm_121 真走 native packed decode path 出 portability 数字。

---

## Scripts in this branch

| File | Purpose |
|---|---|
| `scripts/spark/run_27b_nvfp4_server.sh` | docker run Lynn engine OpenAI HTTP server (port 18099) with sm_121 env vars |
| `scripts/spark/nvfp4_landing_sanity.py` | 1026-tensor count + manifest + tokenizer + safetensors load smoke |
| `scripts/spark/master_27b_eval.py` | Round-1 ship-gate: 6-smoke + tool-call + V9-8 + longctx + TPS |
| `scripts/spark/v8v9_full_eval.py` | **Round-2 deep**: V8(35) + V9(60) + TPS, `/no_think` patched + temp=0 |
| `scripts/spark/landing_pipeline.sh` | Auto: wait transfer → rename → sanity → start server → master eval |

Eval result outputs: `/home/merkyor/reports/27b_nvfp4_*` on Spark — **not committed to repo**(runtime data only).

---

## 不同步主线 of 内容

本分支**不**含:
- native FP4 fused decode / full-token graph / packed nvfp4 moe / native FP4 lm_head 等 R6000 主线工作(那是 `feat/phase3.2-opt-in-paths`)
- 103.44 TPS / 107.23 TPS / 99.86 TPS breakthrough 数字归属(也是主线)

需要同步主线的引擎代码更新时:
```
cd lynn-engine
git fetch origin
git merge origin/main  # 或 cherry-pick 需要的 commits
```
