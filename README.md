# Lynn Engine

> **为 NVIDIA Blackwell 写的 Qwen 3.6 35B-A3B 单模型推理引擎。**
> 从零写,锁定 NVFP4 + 单 prompt 场景,目的是 (a) 把模型每一层搞懂 (b) 在 Lynn 自家 27B-A3B 剪枝模型上单流速度超过 vLLM/SGLang 这种通用框架。

[Read in English](README_EN.md) · [战略文档](docs/STRATEGY.md) · [架构设计](docs/DESIGN.md)

[![commits](https://img.shields.io/github/commit-activity/m/MerkyorLynn/lynn-engine)](https://github.com/MerkyorLynn/lynn-engine/commits/main)
[![license](https://img.shields.io/badge/license-TBD-orange)](.)

## 🆕 2026-05-14 进度更新 — P1-P2 通过,P3 native FP4 GEMM 启动

最新工作都在 [`phase4/reference-workload`](https://github.com/MerkyorLynn/lynn-engine/tree/phase4/reference-workload) 分支(commit range [`e4bb9d5`](https://github.com/MerkyorLynn/lynn-engine/commit/e4bb9d5) → [`7c7f735`](https://github.com/MerkyorLynn/lynn-engine/commit/7c7f735))。

**✅ P1 — 独立 loader**:直接读 safetensors blob + `quantization_config`,**绕开 `transformers.AutoModelForX`**。免疫 modelopt 在 4 大主流 serving 引擎(vLLM / SGLang dev-cu13 / SGLang stable / TRT-LLM 1.2)silent random-init experts 的死结。

**✅ P2 — Reference parity + serving loop 完整闭合**:
- 40 层 BF16 + NVFP4 v8-RTN 跟 reference logits cosine **0.99591**,top-10 overlap **90%**
- Incremental decode parity(KV cache + linear-attention recurrent state)
- **Resident runner** load 一次跑多 prompt → **13.1 tok/s** 慢路径基线
- **CLI 入口**([`e1aa5b4`](https://github.com/MerkyorLynn/lynn-engine/commit/e1aa5b4)):BF16 12.80 / NVFP4 12.62 tok/s
- **OpenAI-compatible HTTP server**([`a593795`](https://github.com/MerkyorLynn/lynn-engine/commit/a593795) NVFP4 + [`7c7f735`](https://github.com/MerkyorLynn/lynn-engine/commit/7c7f735) BF16):`/health` + `/v1/chat/completions` + `/v1/completions`,**两个量化档共用一份 `LynnIncrementalRunner` decode 代码**

**🔜 P3 — native FP4 GEMM 启动**(R6000 lease 截止 2026-05-17,3 天硬窗口):

**P3-A 到 P3-H 全部 PASS**(最新 commit [`cd9f9c0`](https://github.com/MerkyorLynn/lynn-engine/commit/cd9f9c0)):

| 阶段 | 验证 | cosine | rel_l2 | commit |
|---|---|---|---|---|
| P3-A | Packed NVFP4 matvec kernel | **1.0000** | 1.8e-7 | — |
| P3-B | `PackedNVFP4Linear` runtime wrapper | **1.0** | 8.12e-9 | — |
| P3-C | `decode_linear_attn` QKV packed projection | 0.999987 | 5.07e-3 | — |
| P3-D | `decode_linear_attn` 全部 5 个 projection(QKV + Z + B + A + out) | 0.999981 | 6.2e-3 | — |
| P3-E | dual `a+b` projection fusion(launch-overhead **2.02×**) | — | — | — |
| P3-F | packed single expert FFN(gate / up / down) | 0.999995 | 3.28e-3 | [`819d37f`](https://github.com/MerkyorLynn/lynn-engine/commit/819d37f) |
| P3-G | layer0 完整 decode bridge(linear-attn + top-k 8 active expert) | 0.9999976 | 2.18e-3 | [`57df76f`](https://github.com/MerkyorLynn/lynn-engine/commit/57df76f) |
| **P3-H** | layer0 bridge × **8 deterministic seeds + set/order gate**(worst cosine 0.99999696) | — | 最大 2.46e-3 | [**`cd9f9c0`**](https://github.com/MerkyorLynn/lynn-engine/commit/cd9f9c0) |

**Packed NVFP4 已 wire 进完整 decode 子图**:linear-attention 5 projection + MoE top-k 8 active expert + 双 projection fusion 全部 PASS,**8 seed 跑下来 8/8 通过 set-level gate**(1/8 order swap WARN,不算 fail)。

### 验证基线双层 framing(P2 + P3 累计建立)

Lynn engine 的 NVFP4 量化引擎验证规则,**区分真发散 vs 可接受边界噪声**两层:

| 层 | 严格 FAIL 条件 | 算 WARN 不卡 ship | 起源 |
|---|---|---|---|
| **Token 层**(generation) | 不在 reference top-k 之内 | margin < 0.5 close-margin tiebreaker | P2-F |
| **Router 层**(MoE top-k 8 expert dispatch) | top-k **set** 不同 | top-k **set** 同 + 顺序不同 | **P3-H** |

两条都过才严格 PASS。业界对 MoE 量化引擎的验证通常没这层区分 — 要么 strict exact match 抓 false alarm,要么只看 cosine 漏 router 选错。

### 重要 framing

P3-A→H 完成的是 **correctness + runtime plumbing 的验证**,不是 production TPS。2.02× fusion speedup 是 kernel-launch overhead 收益,**packed scalar bridge 当前仍慢于 resident BF16 baseline**,真正 production TPS 起飞要等 Blackwell FP4 tensor core(P4/P5)替掉当前 scalar bridge kernels。

**下一站**:P3-I(多 layer 类型覆盖,验证不是 layer0 特例)→ P4/P5(true Blackwell FP4 GEMM)→ 27B 剪枝模型 engine 端验证(V Flash ship 之后)。

**验证基线锁定**(P2 期间确立,P3 起作 ship gate):

| 指标 | 阈值 | Lynn engine 实测 |
|---|---|---|
| logits cosine | ≥ 0.995 | **0.99591** ✓ |
| top-10 overlap | ≥ 90% | **90%** ✓ |
| greedy parity, margin > 0.5 tokens | 100% match | 3/4 exact(1/4 close-margin tiebreaker 算量化噪声) |
| Resident throughput(慢路径) | — | 13.1 tok/s |

**连载复盘**:[Zhihu 三合一长文](https://zhuanlan.zhihu.com/p/2036443846322680848)(2026-05-11 首发 Phase 2+3.2+NVFP4,持续按里程碑追加)。

**配套生态**:
- [`MerkyorLynn/qwen3.6-nvfp4-toolkit`](https://github.com/MerkyorLynn/qwen3.6-nvfp4-toolkit) — NVFP4 量化配方,产出本 engine 跑的 v8-RTN ckpt
- [`MerkyorLynn/lynn-distill-toolkit`](https://github.com/MerkyorLynn/lynn-distill-toolkit) — V4-Pro Distill 蒸馏 pipeline + 4-gate eval + sanity + ship pipeline,模型 ckpt 在 [HuggingFace](https://huggingface.co/nerkyor/Lynn-V4-Pro-Distill-Qwen-35B-A3B) / [ModelScope](https://modelscope.cn/models/Merkyor/Lynn-V4-Pro-Distill-Qwen-35B-A3B)

---

## 当前能力

✅ **端到端数值正确性已验证** — Lynn engine greedy decode 跟生产 vLLM **逐 token 完全一致**。

```
prompt: "The capital of France is"
vLLM:   ' Paris, a city renowned'
Lynn:   ' Paris, a city renowned'   ← 5/5 token 完全一致
```

✅ **40 层全部数值通过**:
- 30 层 linear_attention(GatedDeltaNet)— 跟 HF 逐 bit 一致
- 10 层 full_attention — 端到端 logits 比对通过
- 多 prompt 验证:8 个不同 prompt,top-K(10)平均跟 vLLM 9.8/10 重合

⚙️ **Phase 3.1 incremental decode 已 ship** — full_attention 走 KV cache,linear_attention 走 recurrent state cache。

⚙️ **OpenAI 兼容 HTTP server** — Lynn brain 改一个 URL 就能 A/B 测试。

## 性能进度

| 阶段 | 单 token 延迟 | t/s | 状态 |
|---|---|---|---|
| Phase 2 brute-force | ~300 ms | 2-3 | shipped |
| **Phase 3.1 incremental decode** | **~200 ms** | **5** | **shipped(commit `1e2980b`)** |
| Phase 3.2(active-experts + bmm + indexed)| 目标 ~100-130 ms | 8-10 | 代码已 commit,未实测 |
| Phase 3.3(Triton-fused MoE FFN)| 目标 ~50 ms | 20 | scaffold + 设计已写 |
| Phase 3.4(CUTLASS NVFP4 grouped)| 目标 ~10-15 ms | 60-100 | B 阶段(未来)|
| vLLM SGLang+MTP 基线 | ~14 ms | 60-70 | 参考对照 |

## 长期路线 — C(36h)→ 验证 → B(6 月)

详见 [`docs/STRATEGY.md`](docs/STRATEGY.md)。简版:

| 阶段 | 时间 | 目标 |
|---|---|---|
| **C 阶段(36h 墙钟)** | 2026-05-09 23:30 → 2026-05-15 | A/B ablation:V4 Pro 蒸 vs V4 Flash 蒸 → winner → 剪 30 expert → Recovery LoRA → V9 gate |
| **验证阶段(2-4 周,B 准备并行)** | 2026-05-15 → 2026-06-15 | brain 接 Lynn-V4-Distill-Qwen-27B-A3B-NVFP4 走 SGLang 生产 + 用户实测 |
| **过渡决策点** | 2026-06-15 | 用户实测 ≥ 2 周稳定 + ROI 够 = 进 B |
| **B 阶段** | 2026-06-15 → 2026-12 末(~6 月) | Lynn engine + NVFP4 + agent prefix cache 替 SGLang 当 brain primary |

**⚡ 蒸馏窗口期 — V4 Pro 75% off 截至 2026-05-31**(`$0.435/M in + $0.87/M out`,原价 4x),5/12 前完成所有 V4 Pro 蒸馏踩在促销内。

**A/B Ablation 设计**:
- **A · Lynn-V4-Pro-Distill**:reasoning / 长文调研强,~$57(promo)
- **B · Lynn-V4-Flash-Distill**:风格匹配 brain / 速度直接,~$8

总预算 ~$95-100(双 A100 不能并行,顺序训 24h,~36h 墙钟出 35B winner → 5/15 出 27B)。

**已锁定决策**:
- 推理硬件:**Blackwell sm_12x**(DGX Spark / 5090 / RTX PRO 6000)
- 推理量化:**NVFP4 唯一**(BF16 仅 reference)— 不做 FP8/INT4/AWQ/GGUF
- 推理范围:**单 prompt + batch=1**(不做 PagedAttention)
- 模型锁定:**Lynn-27B-A3B**(Qwen 3.6 35B-A3B 剪 30 expert + Recovery LoRA + V4-Pro 蒸馏)
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

## 快速上手(DGX Spark)

```bash
# 1. 一次性把 FP8 转 BF16(避开 HF FP8 deep-gemm 元数据问题)
docker run --rm --user 1000:1000 \
  -v /home/merkyor/models:/models \
  -v /tmp/lynn-engine:/work -w /work \
  nvcr.io/nvidia/vllm:26.03.post1-py3 \
  python3 engine/convert_fp8_to_bf16.py \
    --src /models/Qwen3.6-35B-A3B-FP8 \
    --dst /models/Qwen3.6-35B-A3B-BF16

# 2. 停 vLLM(Lynn 需要 ~67 GB 内存放权重)
docker stop vllm-qwen35a3b

# 3. 跑 incremental decode demo
docker run --rm --gpus all --ipc=host --user 1000:1000 \
  -v /home/merkyor/models:/models \
  -v /tmp/lynn-engine:/work -w /work \
  -e PYTHONPATH=/work \
  nvcr.io/nvidia/vllm:26.03.post1-py3 \
  bash -c "pip install -q --user transformers==5.8.0 && \
           python3 engine/full_forward.py \
             --prompt 'The capital of France is' \
             --max-new 5 --mode incremental"

# 4.(可选)启动 OpenAI 兼容 HTTP server
docker run -d --rm --gpus all --ipc=host --user 1000:1000 \
  -v /home/merkyor/models:/models \
  -v /tmp/lynn-engine:/work -w /work \
  -p 127.0.0.1:18099:18099 \
  -e PYTHONPATH=/work \
  nvcr.io/nvidia/vllm:26.03.post1-py3 \
  bash -c "pip install -q --user transformers==5.8.0 fastapi uvicorn && \
           python3 -m server.openai_http \
             --model /models/Qwen3.6-35B-A3B-FP8 \
             --host 0.0.0.0 --port 18099"
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
- [Toolabstain 论文(姊妹项目)](https://github.com/MerkyorLynn/toolabstain-paper)
