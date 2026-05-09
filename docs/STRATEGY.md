# Lynn Engine — Strategy & Long-Term Roadmap

> **Locked decisions**(2026-05-09):
>  - Inference hardware: **Blackwell sm_12x only**(DGX Spark + 5090 / RTX PRO 6000)
>  - Inference quantization: **NVFP4 only**(BF16 reference)— no FP8/FP16/INT4/AWQ/GGUF
>  - Inference scope: **single-prompt, single-batch**(no PagedAttention, no continuous batching)
>  - Model lock: **Lynn-27B-A3B**(剪枝 from Qwen 3.6 35B-A3B + Recovery LoRA)
>  - Position: **vertical companion to Lynn LoRA + pruning pipeline**, not a vLLM replacement

This document is the **north star**. When in doubt about scope creep, refer here.

---

## TL;DR — Two phases C → B over 12 months

| Phase | Window | Goal | Production-grade? |
|---|---|---|---|
| **C** | 2026-05 → 2026-08 | Lynn-27B-A3B model出炉(剪枝 + LoRA recovery)+ Lynn engine 当 reference / training validator | ❌ No, SGLang 跑生产 |
| **Transition gate** | 2026-08 末 | Decide: Lynn-27B-A3B 用户验证 ≥ 2 周稳定 + B 阶段 ROI 够 = enter B | — |
| **B** | 2026-09 → 2027-02 | Lynn engine + NVFP4 替 SGLang 当 brain primary | ✅ Yes |

---

## Why C → B(not direct B)

Direct B(立即追求生产引擎):
- **6-9 周 brutal engineering on untested NVFP4 kernels**
- Lynn-27B-A3B 还没 ship(没模型 = 没东西 ship)
- 高失败概率(NVFP4 ecosystem 还不稳)

C → B(分两阶段):
1. **C 阶段**先把 27B 剪枝模型 ship + 用 SGLang 跑生产(用户先用上)
2. **C 阶段**用 Lynn engine 做 reference 验证 / LoRA gate / activation 分析(vertical 价值)
3. **C 阶段**积累 NVFP4 量化经验(modelopt v7 路径,memory 记)
4. **C 阶段**结束有 ship 的模型 + 实战数据 + 用户反馈 → **B 阶段值不值得做有依据**

---

## Why NOT compete with vLLM/SGLang on generality

| 维度 | vLLM/SGLang | Lynn engine 做这个 = 死路 |
|---|---|---|
| Contributor 数 | 70+ / 30+ | 1 |
| 优化人月 | 几年 | 一个月 |
| 多模型/多硬件 | 全套支持 | 锁 1 模型 1 架构 |
| 多量化 | FP8/INT4/AWQ/GPTQ/MXFP4/NVFP4 | 只 BF16 + NVFP4 |
| 多 batch | PagedAttention | 单 prompt |

通用引擎 Lynn 不可能赢。**专做单 prompt + Lynn 私有模型 + Blackwell + NVFP4** 才有 unique value。

## What Lynn engine WILL be(B 阶段 4 个 unique 优势)

vLLM/SGLang 不会做的 4 个特性,这是 Lynn engine 的 MOAT:

### B1. NVFP4 grouped expert FFN(单 prompt 特化)
- 通用引擎要支持 batch=1..N、多 expert routing 模式
- Lynn 锁 batch=1 + Lynn-27B-A3B 剪枝后路由可预测 → CUTLASS kernel 可极致优化
- 预期 vs SGLang NVFP4 同硬件:**单 prompt 快 20-40%**

### B2. Disk KV cache + SHA1 prefix matching(antirez ds4 思路)
- vLLM/SGLang 的 PagedAttention 是**显存级**分页,跨会话不持久化
- Lynn brain agent prompt **99% 重复**(同一段 system prompt + tool schema + few-shot 反复用)
- Lynn engine SHA1 prefix cache:**重复 prompt prefill ≈ 0 ms**
- 用户感知:第一 token 延迟 vLLM ~0.5s → Lynn ~50ms

### B3. Lynn-private tool_call parser
- vLLM 通用 qwen3_coder parser 不知道 Lynn 工具的具体 schema
- Lynn engine 内置 Lynn 全部 tool 的 schema → JSON validate / 错误恢复更精准

### B4. LoRA hot-swap(< 50ms)
- vLLM 支持 LoRA 但加载冷启延迟高
- Lynn 锁单模型,LoRA in-memory cache + 2-3 个 adapter 并存 swap

---

## Quantization 最终决策

```
✅ BF16            永久 reference(数值真值 / 训练 verify / debug)
                    Lynn-27B-A3B BF16 留磁盘 + lynn-engine 跑 reference
✅ NVFP4(E2M1)    唯一生产格式,Blackwell tensor cores 直 acc
                    Lynn-27B-A3B NVFP4 ~14GB = 4090/5090/Spark 全跑

⚠️ FP8(过渡)     C 阶段用(Lynn 35B baseline 是 FP8)
                    B 阶段后 deprecated,生产不再用
                    出问题 fallback 用 SGLang 35B-FP8 不在 Lynn engine 内回退

❌ FP16            BF16 替代,精度无损 + 数值更稳
❌ INT4 (AWQ/GPTQ) 已被 NVFP4 替代(同 4-bit,但 Blackwell 支持 + 精度更好)
❌ INT8            同
❌ GGUF / Q4_K_M / Q5 / Q6   llama.cpp 主场,Lynn 4090 fallback 直接用 llama-server 跑
```

## Hardware target

```
✅ DGX Spark (sm_121)         GB10 Blackwell 119GB unified, 273 GB/s
✅ NVIDIA 5090 (sm_120)        Blackwell, 32GB GDDR7, 1.79 TB/s
✅ RTX PRO 6000 Blackwell      96GB GDDR7, 2 TB/s

❌ A100 / H100                  Hopper,FP4 tensor cores 没有(emulation 不值)
❌ 4090 / Ada                  Ada,无 FP4 tensor cores
❌ Ampere / Volta              老
❌ AMD / Apple Silicon         没 NVFP4
```

**4090 fallback 路径**:Lynn engine 不支持。Lynn-27B-A3B-IQ4_XS 跑 llama.cpp 走 llama-server,Lynn engine 不参与。这是合理的分工。

---

## C 阶段产出清单(2026-05 → 2026-08)

### C1 · Phase 3.1/3.2 验证 + Lynn 35B baseline 数据 ✅(代码 ready,等 DGX)
- `engine/test_incremental_decode.py` 跑过
- `engine/test_moe_optimized.py` 跑过 + 选最佳 MoE 路径
- 5/8 prompt 跟 vLLM 重测一致
- 50 token 稳态 t/s 测量

### C2 · Lynn-35B-A3B activation profile ✅(代码 ready,等 DGX)
- `pruning/profile_activations.py` 跑 calibration_set v1.1(1436 prompts)
- 输出 `activation_profile_35B.jsonl`(~50MB)

### C3 · Drop 30 expert 决策 ✅(代码 ready,等 profile 跑完)
- `pruning/decide_pruning.py` 输出 `drop_candidates_30.json` + rationale.md
- 用户 review + ✅ 才进 C4

### C4 · Recovery LoRA 训练(本文件 deliverable + A100 跑)
- `pruning/training/recovery_lora_qwen36.yaml` LLaMA-Factory 配置
- A100 双卡 ZeRO-3,r=384,Stage 1+5+4+6'+rehearsal
- ~5h × 2-3 轮迭代

### C5 · Lynn-27B-A3B 数值校验
- Lynn engine BF16 跑 27B,跟 HF 单层 alignment(类似 Phase 3a 测试)
- V8/V9 benchmark vs 35B baseline,退化 ≤ 3% = ship

### C6 · NVFP4 量化(为 B 阶段铺路)
- `docs/NVFP4_QUANTIZATION.md` pipeline 文档
- modelopt 0.43 v7-RTN 路径(memory `reference_qwen36_nvfp4_v8_rtn.md`)
- 输出 `Lynn-27B-A3B-NVFP4-v8-RTN`(BF16 + NVFP4 双套权重)

### C7 · 教学 + 公开 ✅(已完成,继续维护)
- 7 篇 tutorials + Zhihu 长文 + B 站脚本(待发)+ Twitter thread
- HuggingFace blog post(可选)

---

## Transition gate(2026-08 末)

进 B 的硬条件,**全满足才进**:

- [ ] Lynn-27B-A3B 在 SGLang 上跑,brain 接入用 ≥ 2 周
- [ ] V8/V9 退化 ≤ 3%
- [ ] 用户反馈无 critical issue
- [ ] Blackwell hardware 用户基础(看 Lynn 用户数 + 5090 持有率)
- [ ] NVFP4 ecosystem 跟进足够(SGLang dev-cu13 stable / vLLM cu130 GA)

任一不满足 → **不进 B,Lynn engine 留 C 阶段状态当 reference + 教学工件**。

---

## B 阶段产出清单(2026-09 → 2027-02)

### B1 · NVFP4 GEMM kernel(Blackwell sm_12x,CUTLASS 3.5+)
- 单 expert + 单 prompt + NVFP4 weights → BF16 acc
- 跟 SGLang dev-cu13 同 prompt 数值对齐 ≤ 1%
- 单层 matmul 速度 ≥ SGLang 同硬件

### B2 · NVFP4 grouped expert FFN
- K=8 active experts,单 kernel call 完成 gate+up+silu+mul+down+weighted sum
- 替代 Phase 3.2.3 的 BF16 Triton 路径
- 性能 vs SGLang grouped GEMM:Lynn ≥ 1.2x(单 prompt)

### B3 · Disk KV cache + SHA1 prefix matching
- 持久化 KV cache 到磁盘
- 进入新 prompt → SHA1 prefix → cache hit → skip prefill
- 重复 prompt(agent 99% 命中)第一 token 延迟 ≤ 100ms

### B4 · Tool call parser(qwen3_coder + Lynn extensions)
- 解析 `<tool_call>...</tool_call>` 块
- Lynn 工具 schema validate
- 错误恢复 / 重试逻辑

### B5 · Streaming SSE / sampling / brain failover
- token-by-token output
- temperature / top_p / top_k
- 错误码体系
- 1 周 soak test

### B 阶段 ship 后状态

```
brain primary:    Lynn engine + Lynn-27B-A3B-NVFP4(Spark / 5090)
brain fallback:   SGLang Qwen 3.6 35B-A3B-FP8(同 Spark)
4090 fallback:    llama-server Lynn-27B-A3B-IQ4_XS(本地用户机器)

Spark 单流性能:   130-180 t/s(vs SGLang 60-70)
重复 prompt:      ~0 prefill cost(disk SHA1 cache)
LoRA 切换:        < 50ms 热加载
```

---

## What we will NOT do(也写下来防 scope creep)

```
❌ Continuous batching / PagedAttention(vLLM 主场)
❌ Multi-batch / 多并发(Lynn brain 单用户场景)
❌ Multi-model loader(锁 Lynn-27B-A3B 家族)
❌ FP8/INT4/AWQ/GPTQ/MXFP4 等 quant(NVFP4 唯一)
❌ Hopper / Ada / Ampere 支持(Blackwell 唯一)
❌ TP / PP / 多机分布式(单 GPU)
❌ Vision encoder 整合(Lynn 不接图像 input;后续看)
❌ 与 vLLM 通用 API 完全兼容(只兼容 brain 用的子集)
❌ 投 paper 当 academic project(Lynn 是产品,不是论文)
```

---

## Risk register

| 风险 | 几率 | 应对 |
|---|---|---|
| Lynn-27B-A3B 剪枝退化 > 5% | 中 | C 阶段 gate 不过 → 保留 35B,Lynn engine archive |
| NVFP4 kernel 写不通(B1 阶段)| 中 | B 阶段不进,Lynn engine 留 BF16 reference |
| Qwen 4 / Qwen 5 出来 | 高(2026 年内可能)| 评估升级成本 — 大改重写 vs 维持 |
| SGLang 一直跑稳 → Lynn engine 替换 ROI 不够 | 中 | 不进 B,这是 fine 的退路 |
| Disk SHA1 cache 复杂度爆炸 | 低 | 简化版只缓存 system prompt,不全段 |
| Blackwell hardware 用户基础不增长 | 中 | 等 5090 普及(2026 H2)|

---

## 修订历史

- 2026-05-09 初版,锁定 Blackwell + NVFP4 + C → B 路线
