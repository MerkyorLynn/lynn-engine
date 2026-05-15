# Lynn Engine

> **为 NVIDIA Blackwell 写的 Lynn 27B-A3B NVFP4 单模型推理引擎。**
> 从零写,锁定 Lynn 自家的 variable-pruned MoE + NVFP4 格式,目标很窄也很硬:在 R6000 / Spark 这类 Blackwell 机器上,把 Lynn 27B 基座跑成可生产、可优化、可长期接管的推理内核。

[Read in English](README_EN.md) · [📝 知乎工程复盘(2026-05-11)](https://zhuanlan.zhihu.com/p/2036443846322680848) · [战略文档](docs/STRATEGY.md) · [架构设计](docs/DESIGN.md)

[![commits](https://img.shields.io/github/commit-activity/m/MerkyorLynn/lynn-engine)](https://github.com/MerkyorLynn/lynn-engine/commits/main)
[![license](https://img.shields.io/badge/license-TBD-orange)](.)

## 当前状态(2026-05-15)

Lynn engine 已经从“Qwen 35B 架构复刻”推进到 **Lynn 27B final 基座的独立 NVFP4 runtime**:

| 项目 | 状态 |
|---|---|
| **27B final BF16** | ✅ Recovery step5000 final 已 merge,structural validation PASS,greedy sanity PASS |
| **27B Lynn-native NVFP4** | ✅ 20G artifact 已生成并传到 R6000,manifest integrity PASS |
| **独立加载** | ✅ 不依赖 vLLM / SGLang / TRT-LLM / llama.cpp,直接读 safetensors + Lynn quant manifest |
| **6-prompt coherent smoke** | ✅ 中文解释 / Python / RoPE-ALiBi / 英文算术 / tool JSON / longctx 全通过 |
| **当前 R6000 steady decode** | ✅ **66-68 tok/s**(真实 generate path,32-token smoke) |
| **稳定 CUDA graph ceiling** | ✅ **78.8 tok/s**(full-token graph probe,可复现) |
| **下一目标** | 100 tok/s:packed-resident NVFP4 + hot path kernel fusion |

当前主力 artifact:

```text
Lynn 27B variable-pruned Recovery step5000
├── BF16 final      ~60G  (reference / eval / fallback)
└── NVFP4 final     ~20G  (Lynn-native runtime artifact)
```

> 注意:这里的 NVFP4 是 **Lynn-native variable-expert NVFP4**。它不是公开发布用的 compressed-tensors v8-RTN,也不是 GGUF Q4_K_M。通用框架通常不能直接加载这个 variable-pruned artifact,这正是 Lynn engine 要存在的原因。

## 性能进度

| 阶段 | 单 token 延迟 | t/s | 状态 |
|---|---|---|---|
| Phase 2 brute-force | ~300 ms | 2-3 | 历史基线 |
| Phase 3.1 incremental decode | ~200 ms | 5 | 历史基线 |
| P5/P6 eager Triton path | ~30-33 ms | 30-33 | P6-K/N/O |
| **P6-S resident graph smoke** | **~15-16 ms** | **63-66** | ✅ 50TPS 目标突破 |
| **P7 current serving env** | **~14.6-15.0 ms** | **66-68** | ✅ 6-prompt generate PASS |
| **P7/P8 CUDA graph ceiling** | **12.68 ms** | **78.8** | ✅ 稳定可复现 ceiling |
| P8 torch.compile spike | 12.33 ms | 81.1 | 实验信号,非产品路径 |
| **P9 target** | **~10 ms** | **100** | packed-resident + kernel fusion |
| Long target | <5 ms | >200 | native FP4 / larger fused blocks |

当前 R6000 推荐环境:

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LYNN_PREFILL_WARMUP=1
export LYNN_LINEAR_ATTN_RECURRENT_BACKEND=triton_fused_prepare
export LYNN_MOE_IMPL=triton
export LYNN_QK_NORM_ROPE_BACKEND=triton
export LYNN_RMSNORM_GATED_BACKEND=triton
export LYNN_LINEAR_ATTN_INPROJ_FUSED=1
export LYNN_LINEAR_BLOCK_GRAPH=1
export LYNN_LINEAR_BLOCK_GRAPH_REUSE=1
export LYNN_LINEAR_BLOCK_GRAPH_PREWARM=1
export LYNN_LINEAR_STATE_UPDATE=inplace
```

实测 final step5000 NVFP4 32-token smoke:

```text
decode TPS per prompt: 66.62 / 68.27 / 68.28 / 68.24 / 68.44 / 67.89
load time:             10.7s
graph capture:         0.0s per request after prewarm
output:                coherent 6/6
```

## 27B 质量与基座状态

27B 来自 Qwen 3.6 35B-A3B BASE 的 variable-expert 剪枝路线:

```text
BASE 35B-A3B
  → activation profile
  → variable-target expert pruning(1010 experts cut,front layers protected)
  → router fine-tune
  → Recovery LoRA
  → step5000 selected as final
  → BF16 merge
  → Lynn-native NVFP4 quantization
```

已知质量结论:

| Variant | 状态 | 结果 |
|---|---|---|
| 27B BF16 step5000 | ✅ full eval | V8 strict 33/34 = 97.06%, V9 adjusted 37/59 = 62.71% |
| 27B NVFP4 step5000 | ✅ runtime smoke | 6-prompt resident smoke PASS,2-token greedy sanity PASS |
| 27B Q4_K_M | ⏳ not primary | variable-expert GGUF 需要额外 padding/format work,不是当前 Lynn-native 主线 |

Recovery v1.1 targeted longctx/chem/sql 已试过,但未取代 step5000:它没有改善 longctx,并拉低总分。因此当前基座选择 **step5000 final**。

## 路线 — R6000 first,Spark second

详见 [`docs/STRATEGY.md`](docs/STRATEGY.md)。简版:

| 阶段 | 状态 | 目标 |
|---|---|---|
| **P6** | done | 50TPS 突破:resident graph smoke 63-66TPS |
| **P7** | done | graph reuse / prewarm / RMSNormGated / serving env,66-68TPS |
| **P8** | partial | 78.8TPS CUDA graph ceiling + 81TPS compile spike |
| **P9** | current | packed-resident NVFP4,释放 35-40G resident memory,奔 100TPS |
| **P10** | next | native FP4 / larger fused kernels,奔 200TPS |

**已锁定决策**:
- 推理硬件:**Blackwell sm_12x**(DGX Spark / 5090 / RTX PRO 6000)
- 推理主格式:**Lynn-native NVFP4**(BF16 仅 reference;GGUF/FP8 是兼容/对外测试资产)
- 推理范围:**单 prompt + batch=1**(不做 PagedAttention)
- 模型锁定:**Lynn-27B-A3B variable-pruned family**
- 定位:**vertical companion** 给 Lynn LoRA + 剪枝训练流水线,**不替 vLLM 当通用引擎**

## 教程 — 即使你不写自己的引擎也值得读

写 Lynn engine 时,我们挖出了 Qwen 3.6 35B-A3B **跟 Llama / Qwen 2 不一样、文档没写明的怪癖**。共 7 篇深度文章在 [`tutorials/`](tutorials/):

| # | 主题 | 一句话 |
|---|---|---|
| [01](tutorials/01_rmsnorm_one_plus_weight.md) | RMSNorm `(1.0 + w) × x` 不是 `w × x` | Qwen 3 系 RMSNorm 是 +1 偏移。照 Llama 抄数值偏 ~10x |
| [02](tutorials/02_rope_three_gotchas.md) | RoPE 三个连环坑 | theta 在 `rope_parameters`(不是 `rope_theta`)+ `partial_rotary_factor=0.25` + GPT-NeoX 半切(不是 Qwen 2 even/odd)|
| [03](tutorials/03_attn_output_gate.md) | q_proj 是 2× per-head 切分 | 必须先 view 成 (..., H_Q, 2*head_dim) 再 chunk,否则 head_i_gate 混进 head_i_q |
| [04](tutorials/04_gated_delta_net.md) | linear_attention = GatedDeltaNet | Mamba 风格 chunk 递推 + delta rule + l2norm Q/K |
| [05](tutorials/05_three_invisible_bugs.md) | 三个 self-consistent bug 复盘 | reference + lynn 同源同错 = 自一致测试假阳的教训 |
| [06](tutorials/06_moe_router_softmax_topk_order.md) | MoE router order + shared expert | Qwen 用 softmax-all → topK → renormalize,跟 naive 数学等价但精度路径不同 |
| [07](tutorials/07_lora_on_gated_delta_net.md) | 给 GatedDeltaNet 加 LoRA | 哪些线性层可加 / 哪些不能 / r=384 用于 Recovery 的理由 |

[`tutorials/posts/zhihu_qwen36_engine_postmortem.md`](tutorials/posts/zhihu_qwen36_engine_postmortem.md) 是知乎博客风格的合集长文。

## 快速上手(R6000 / Blackwell)

```bash
# 1. 准备 Lynn-native NVFP4 artifact
MODEL=/root/autodl-tmp/models/lynn-27b-variable-recovery-step5000-nvfp4-final

# 2. 启用当前 R6000 best env
export PYTHONPATH=/root/autodl-tmp/lynn-engine
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LYNN_PREFILL_WARMUP=1
export LYNN_LINEAR_ATTN_RECURRENT_BACKEND=triton_fused_prepare
export LYNN_MOE_IMPL=triton
export LYNN_QK_NORM_ROPE_BACKEND=triton
export LYNN_RMSNORM_GATED_BACKEND=triton
export LYNN_LINEAR_ATTN_INPROJ_FUSED=1
export LYNN_LINEAR_BLOCK_GRAPH=1
export LYNN_LINEAR_BLOCK_GRAPH_REUSE=1
export LYNN_LINEAR_BLOCK_GRAPH_PREWARM=1
export LYNN_LINEAR_STATE_UPDATE=inplace

# 3. 运行 resident smoke
python benchmarks/resident_cli.py \
  --model "$MODEL" \
  --prompts-jsonl /root/autodl-tmp/reports/lynn-engine-p5/p7i_6prompt.jsonl \
  --max-new 32 \
  --chat-template \
  --out /tmp/lynn_27b_nvfp4_smoke.json

# 4. 可选:OpenAI-compatible server
python -m server.openai_http \
  --model "$MODEL" \
  --host 0.0.0.0 \
  --port 18099
```

## 仓库结构

```
engine/
  loader.py                       FP8 e4m3 反量化 + 单层 safetensors 加载器
  qwen36_block.py                 早期完整 transformer block(P1.1,带 bug 的 reference)
  qwen36_linear_attn_block.py     GatedDeltaNet 移植 — 跟 HF bit-exact
  full_forward.py                 40 层端到端 forward(已修正)
  inference_state.py              单 request 的 KV cache + recurrent state
  incremental_decode.py           prefill / decode 原语(Phase 3.1)
  moe_optimized.py                MoE 三档优化(active / bmm / indexed bmm)
  convert_fp8_to_bf16.py          离线 FP8 → BF16 转换器(CPU)
  test_*.py                       逐层 alignment 测试 + 多 prompt 验证

triton_kernels/
  attention.py / rope.py / rmsnorm.py / moe.py   P1 spike 内核
  moe_expert_ffn.py               Phase 3.2.3 融合 MoE 内核(scaffold)

server/
  openai_http.py                  FastAPI OpenAI 兼容 server
  README.md                       brain 接入指引

docs/
  STRATEGY.md                     ⭐ 战略 + 长期路线图
  DESIGN.md                       架构 + 实施日志(长)
  PHASE3_KV_CACHE_DESIGN.md       Phase 3.1 设计文档
  NVFP4_QUANTIZATION.md           NVFP4 量化 pipeline(C 阶段末 + B 阶段输入)

pruning/
  calibration/
    seeds.jsonl                   102 手工策划 seed prompt(19 类别)
    expand.py                     DeepSeek API 扩到 1440 prompts
    calibration_set_v1.1.jsonl    生成的完整集
  profile_activations.py          Phase 1 W1: 激活画像脚本
  decide_pruning.py               Phase 1 W2: drop 30 expert 决策算法
  training/
    recovery_lora_qwen36.yaml     Recovery LoRA r=384 配置(LLaMA-Factory)

benchmarks/
  lynn_27b_vs_35b.py              Phase 1 W4 gate test(自动跑 V9 比较)

tutorials/
  README.md + 7 篇 tutorial + Zhihu 长文
```

## 部署目标硬件

| 硬件 | VRAM | 状态 |
|---|---|---|
| **DGX Spark**(GB10 sm_121,unified 119GB) | 119 GB | C 阶段开发主力 |
| **RTX 5090 笔记本**(sm_120,24GB) | 24 GB | **Lynn-27B-A3B-NVFP4 占 ~20 GB,4 GB 余量给 16K context** ✅ |
| **RTX 5090 台式机**(sm_120,32GB) | 32 GB | 预期 180-250 t/s,32K context 宽裕 |
| **RTX PRO 6000 Blackwell**(sm_120,96GB) | 96 GB | 多 LoRA 切换 + 长 context 扩展 |
| ~~4090 / Ada~~ | — | 不支持(没 FP4 tensor cores)|
| ~~A100 / H100 / Hopper~~ | — | 不支持(同上,FP4 emulation 不值)|
| ~~Ampere / Volta~~ | — | 不支持(老)|

## 为什么写这个

Lynn brain 用 Qwen 3.6 35B-A3B 当主链,每天处理几千个 agent 请求。vLLM(目前生产)能吃掉 Spark 内存带宽 ~80%,Lynn engine 的目标是**单模型锁定 + Blackwell sm_12x 特化** + agent prompt 缓存(99% 命中)拿到剩下 20% + 更低 tail latency。

但更重要的产出是**通过自己写引擎获得的理解** — 上面的 tutorials 就是这个理解的固化。

## 老实讲的取舍

```
❌ 锁定 Qwen 3.6 35B-A3B(及其剪枝衍生)+ Blackwell sm_12x
❌ Qwen 升 3.7/4 不兼容时,需要 4-6 周重写
❌ 不做批处理 / 多并发(单 prompt 焦点)
❌ 不做采样(只 greedy);不做 beam / speculative
❌ 不支持 FP8/INT4/AWQ/GPTQ 等(NVFP4 唯一生产格式)
✅ 完全契合 Lynn brain 的部署模式
✅ 跟 LoRA + 剪枝训练流水线 vertical 整合
✅ 所有架构细节都给后人写下来了
```

## 不打算做的事(也写明防止 scope creep)

```
❌ Continuous batching / PagedAttention(vLLM 主场)
❌ 多模型 loader(锁 Lynn-27B-A3B 家族)
❌ 多量化格式(NVFP4 唯一)
❌ Hopper / Ada / Ampere 支持(Blackwell 唯一)
❌ TP / PP / 多机分布式(单机单 GPU)
❌ Vision encoder 整合(Lynn 不接图像;后续看)
❌ 与 vLLM 通用 API 完全兼容(只兼容 brain 用的子集)
❌ 投 paper(Lynn 是产品,不是论文)
```

## License

TBD(大概率 MIT,B 阶段生产切换前定下来)。

---

## 相关链接

- [战略 / 路线图(STRATEGY.md)](docs/STRATEGY.md)
- [架构设计(DESIGN.md)](docs/DESIGN.md)
- [Phase 3 incremental decode 设计](docs/PHASE3_KV_CACHE_DESIGN.md)
- [NVFP4 量化 pipeline](docs/NVFP4_QUANTIZATION.md)
- [剪枝 pipeline(calibration / profile / decide / LoRA)](pruning/README.md)
- [HTTP server(brain 接入指引)](server/README.md)
- [English README](README_EN.md)
- [📝 知乎工程复盘:从零开始 Qwen 3.6 35B-A3B 写专用推理引擎(2026-05-11 mega-post)](https://zhuanlan.zhihu.com/p/2036443846322680848) — Phase 2 + Phase 3.2 + NVFP4 路线决策三段合并,~15-20k 中文字
- [Toolabstain 论文(姊妹项目)](https://github.com/MerkyorLynn/toolabstain-paper)
