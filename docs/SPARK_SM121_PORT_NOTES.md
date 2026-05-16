# Spark sm_121 平台移植 / 验证笔记

> **Branch**: `spark/sm121-port`(this branch).  
> **Scope**: DGX Spark GB10 sm_121 平台兼容、启动脚本、eval harness、scalar_bridge ↔ native_prepared path 开关验证。  
> **所有性能数字 = Spark-only**,不代表 Lynn engine 真实性能(那是 R6000 主线 `feat/phase3.2-opt-in-paths` 的事)。

---

## 🏁 最新战报 — SP-01...SP-08 Triton autotune 完成(2026-05-16)

**Lynn 27B NVFP4 vs SGLang FP8+MTP @ Spark sm_121,一个模型一个设备**

| Metric | Lynn 27B NVFP4 (SP-08) | SGLang FP8+MTP 35B | Verdict |
|---|---:|---:|---|
| single mean | **49.37** | 43.44 | **Lynn +13.7%** ⭐ |
| single peak | **49.38** | 47.30 | **Lynn +4.4%** ⭐ |
| mixed mean | 49.11 | 49.97 | ≈ TIED(0.86 TPS within SGLang 1.4 SE)|
| mixed peak | 49.39 | 62.51 | SGLang +27%(架构红利:NEXTN MTP head)|
| stddev | **0.17** | 6.22 | **Lynn 37× steadier** ⭐ |

**3 胜 / 1 平 / 1 负** — autotune-only 路线天花板。8 步累计 **+13.9% / +13.5%**(43.33 → 49.37 single,43.26 → 49.11 mixed),**kernel 数学全程不变**,纯 `(BLOCK_INTER, BLOCK_HIDDEN, num_warps, num_stages)` launch-config sweep。

**完整方法论 + 8 步 trajectory + 剩余 3 瓶颈 + llama.cpp 70 TPS 解释 + production promotion gate**:[`reports/sp01_autotune/SPARK_VS_SGLANG_FINAL_20260516.md`](../reports/sp01_autotune/SPARK_VS_SGLANG_FINAL_20260516.md)

**Production 启用方法**:`scripts/spark/run_27b_nvfp4_server.sh` 加 `LYNN_SP_TRITON_AUTOTUNE=1` env(其他 env 全部保持 production canonical 不动)。完全 reversible,kernel 数学等价。

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

| Format | Reader 与运行路径 |
|---|---|
| Lynn-native `nvfp4_e2m1_rowwise_per_16`(this) | **只 Lynn engine fail-loud loader** ✓(其他全 ❌)。文件层面可保 packed ~20G,但**当前 resident_runner 仍有 BF16 shadow**(加载时 slow dequant 到 BF16 resident),运行显存还**没完全吃到 20G 红利** — 减 shadow 是 P9+ resident_runner 待做项。 |
| `v8-RTN compressed-tensors`(V Flash 用过的格式) | SGLang dev-cu13 ✓ — **可走 `CompressedTensorsW4A4Nvfp4MoE` native-ish path**,不必是"全 dequant 到 BF16 再 GEMM";但**它是通用框架 loader 格式,不是 Lynn 项目可控的 variable-expert runtime 格式**(拿不到 per-layer 不同 expert 数 / per-tensor 量化决策 fine-grained control)。llama.cpp ✗。|

### Native FP4 算力潜力 — 定性(不是已实现数字)

- **2× BF16 算力**(Blackwell sm_120/121 FP4 tensor core 设计上限)= **目标上限 / 硬件路径**,**不是 Lynn engine 当前全链路已经达到的数字**
- 当前 Spark sm_121 上 27B NVFP4 跑 `backend=scalar_bridge`,**不走 native FP4 GEMM**;native_prepared path 切换实验是本分支 portability 验证项之一
- 任何 model card / 知乎稿 / 宣传材料提及 "2× BF16 算力" 必带 framing:**"硬件理论上限 / Lynn engine 正向其推进"**,不能写 "已达成"

### Server 启动

参考 `scripts/spark/run_27b_nvfp4_server.sh`,**完整 P10 production env vars**(与 R6000 88-89 TPS stable 配置对齐):
```
LYNN_PREFILL_WARMUP=1
LYNN_MOE_IMPL=packed_nvfp4                     ← MUST(默认 triton 不走 packed NVFP4 path)
LYNN_LINEAR_ATTN_RECURRENT_BACKEND=triton_fused_prepare
LYNN_LINEAR_ATTN_RECURRENT_INPLACE=1           ← MUST(P10 新增 inplace)
LYNN_LINEAR_STATE_UPDATE=inplace
LYNN_QK_NORM_ROPE_BACKEND=triton_pair
LYNN_RMSNORM_GATED_BACKEND=triton
LYNN_LINEAR_ATTN_INPROJ_FUSED=1
LYNN_LINEAR_ATTN_INPROJ_FUSED_NATIVE_FP4=1     ← MUST(P8 native FP4 fused inproj)
LYNN_NATIVE_FP4_LM_HEAD=1                      ← MUST(P10 native FP4 lm_head,FP4 lm_head 103.49 TPS strict path)
LYNN_LINEAR_BLOCK_GRAPH=1
LYNN_LINEAR_BLOCK_GRAPH_REUSE=1
LYNN_LINEAR_BLOCK_GRAPH_PREWARM=1
LYNN_PACKED_DECODE=1
LYNN_PACKED_DECODE_BACKEND=native_fast_2d      ← MUST(default scalar_bridge,如不设 native_prepared=0)
LYNN_PACKED_DECODE_FULL_ATTN=1
LYNN_PACKED_DECODE_LINEAR_ATTN=1
LYNN_PACKED_DECODE_PREPARE_NATIVE=1
LYNN_PACKED_SHARED_EXPERT=1
```

⚠️ **Round-1 evidence(23.88 TPS)是错配置数字**:之前漏掉 `LYNN_MOE_IMPL=packed_nvfp4` / `LYNN_LINEAR_ATTN_RECURRENT_INPLACE=1` / `LYNN_NATIVE_FP4_LM_HEAD=1` 三个 MUST env,导致 server fallback 到 scalar_bridge 老 path。修对配置后 Spark 应该向 30-50+ TPS 推进(参考 R6000 P10 doc `LYNN_ENGINE_P10S_GRAPH_BOUNDARY_20260515.md`)。

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
