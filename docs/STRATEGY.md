# Lynn Engine — Strategy & Long-Term Roadmap

> **Locked decisions**(2026-05-09):
>  - Inference hardware: **Blackwell sm_12x only**(DGX Spark + 5090 / RTX PRO 6000)
>  - Inference quantization: **NVFP4 only**(BF16 reference)— no FP8/FP16/INT4/AWQ/GGUF
>  - Inference scope: **single-prompt, single-batch**(no PagedAttention, no continuous batching)
>  - Model lock: **Lynn-27B-A3B**(剪枝 from Qwen 3.6 35B-A3B + Recovery LoRA)
>  - Position: **vertical companion to Lynn LoRA + pruning pipeline**, not a vLLM replacement

This document is the **north star**. When in doubt about scope creep, refer here.

---

## TL;DR — 36h 蒸馏 → 5/15 出 27B → 验证 → B 阶段 6 月

| Phase | Window | Goal | Production-grade? |
|---|---|---|---|
| **C 阶段 — 蒸馏 + 剪枝(36h 墙钟)** | 2026-05-09 23:30 → 2026-05-15 | **A/B ablation:V4 Pro 蒸 vs V4 Flash 蒸 → 选 winner → 剪 30 expert → Recovery LoRA** | ❌ No, SGLang 跑生产 |
| **验证阶段(2-4 周,B 准备并行)** | 2026-05-15 → 2026-06-15 | brain 接 Lynn-V4-Distill-27B-A3B-NVFP4 走 SGLang 生产 + 用户实测 | — |
| **Transition gate** | 2026-06-15 | 用户实测 ≥ 2 周 OK + B 阶段 ROI 够 = enter B | — |
| **B 阶段** | 2026-06-15 → 2026-12 末(~6 月) | Lynn engine + NVFP4 kernel + agent prefix cache 替 SGLang 当 brain primary | ✅ Yes |

### ⚡ 蒸馏窗口期 — V4 Pro 75% off 截至 2026-05-31

DeepSeek 半年大促,**5/12 完成所有蒸馏**踩在促销窗口内。错过 → 价格回 4x:

| 模型 | Input/M | Output/M | 备注 |
|---|---|---|---|
| **V4 Pro**(promo) | **$0.435** | **$0.87** | ⭐ 75% off 至 5/31 |
| V4 Pro(原价)| $1.74 | $3.48 | 5/31 后回原价 |
| V4 Flash | $0.14 | $0.28 | 标准价 |

### A/B Ablation 设计(双版本同时跑)

| 版本 | 蒸馏教师 | Token 预算 | 成本估算 | 期望强项 |
|---|---|---|---|---|
| **A · Lynn-V4-Pro-Distill** | DeepSeek-V4-Pro | 15K × (800 in + 4000 out) = 72M tok | ~$57(promo)| reasoning / 长文调研 / 难题 |
| **B · Lynn-V4-Flash-Distill** | DeepSeek-V4-Flash | 15K × (800 in + 1500 out) = 34.5M tok | ~$8 | 风格匹配 brain / 速度直接 |

**总预算估算**:~$95-100(含 P1 HAS 已花 + 20% buffer + gate 调试)

**当前 DS 余额 ~$95** → 推荐充 $50 到 $160(安全 + 后续 27B Recovery iter / 调试 buffer)。

### 完整 timetable(5/9 → 5/15,墙钟 ~6 天)

```
🟢 双 A100 时段(剩 ~110h,实际用 ~60h 提前 3 天结束)
─────────────────────────────────────────────────────
5/09 23:14    P1 HAS ✅ 完
5/09 23:30 → 5/10 06:00   API 采 30K (V4 Pro 15K + Flash 15K,并行 P0 ORPO)
5/10 ~05:30                P0 ORPO 完
5/10 06:00 → 07:30         Stage 6' gate 验收
5/10 07:30 → 19:30         训 Version A(V4 Pro distill,~12h on 2×A100 Z3)
5/10 19:30 → 21:30         Gate A
5/10 21:30 → 5/11 09:30    训 Version B(V4 Flash distill,~12h on 2×A100 Z3)
5/11 09:30 → 11:30         Gate B
5/11 ~12:00                ⭐ Winner = Lynn-V4-Distill-Qwen-35B-A3B
                          【双卡 A100 时段提前 3 天结束】

🟡 切 4×A100 时段(5/11 中午起 — 提前升)
─────────────────────────────────────────────────────
5/11 12:00 → 18:00         升级 + 环境迁移 + 校验
                          (数据/checkpoint/yaml 都在 /mnt/data3 持久,无损)
5/11 18:00 → 5/12          Phase 1 W1 激活画像(35B winner 推理 1430 prompts)
5/12 → 5/13                 Phase 1 W2 物理剪枝(砍 30 expert → 27B)+ router fine-tune
5/13 → 5/14                 Phase 1 W3 Recovery LoRA r=384(~12-18h on 4×A100)
5/14                       双卡 A100 到期 ✓(已无关,4 卡承担)
5/15                       ⭐ Gate + Lynn-V4-Distill-Qwen-27B-A3B 出炉
                          → Lynn engine 这边接手:NVFP4 量化 + V9 vs 35B harness 跑
```

**关键限制说明**:
- **双 A100**:35B BF16 单卡 80GB 装不下(70GB weight + activations)→ 必须 Z3 sharding 占双卡 → A/B 不能并行只能顺序
- **4 卡 A100**:5/11 中午切(原计划 5/14,提前 3 天)→ Recovery LoRA 在 4 卡跑更快/更稳,Phase 1 W3 ~12-18h 完成

### 压缩思路

- 原 C 阶段 4 个月 → 现在 **1 周(36h+剪枝)**
- A/B ablation 不浪费时间(顺序训 24h)— 拿数据决定 V4 Pro 是否值多花的钱(7x 价差)
- 验证跟 B 阶段开发并行,不等验证完才开 B
- B 阶段 NVFP4 kernel 工程量是硬上限,~6 个月仍现实
- **2026 年内 Lynn-27B-A3B + Lynn engine 上生产**(原计划 2027-02 → 提前 ~3 个月)

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

## C 阶段产出清单(压缩版 2026-05-09 → 05-15,~1 周)

### C0 · 蒸馏数据采集(5/09 23:30 → 5/10 06:00,~6.5h API)
- 15K queries × V4-Pro = 72M tok 给 Version A
- 15K queries × V4-Flash = 34.5M tok 给 Version B
- 同一批 prompts(覆盖 coding / tool_call / math / finance / longform / creative)
- 工具:`pruning/distillation/collect.py`(待写)
- 跟 P0 ORPO 并行,用 Tencent 的 brain 出口走 API

### C0.5 · A/B Ablation 训练(5/10 07:30 → 5/11 09:30,顺序 ~24h on 2×A100)
- **Version A(Pro 蒸)**:5/10 07:30 → 19:30(~12h)
- **Version A Gate**:5/10 19:30 → 21:30(V8/V9 跑)
- **Version B(Flash 蒸)**:5/10 21:30 → 5/11 09:30(~12h)
- **Version B Gate**:5/11 09:30 → 11:30
- **Compare A vs B → 5/11 11:30 出 winner**
- LLaMA-Factory + ZeRO-3,bf16 + gradient_checkpointing,r=192 蒸馏 LoRA(蒸馏阶段不用 384,数据量足)

### C1 · Phase 3.1/3.2 Lynn engine 验证 ⚠️ (2026-05-10 RTX PRO 6000 BF16,multi-prompt 暴露 bmm/indexed_bmm 数值不稳)

**单 prompt benchmark**(max_new=10):

| Phase | t/s | × baseline | 单 prompt match | 部署 friction |
|---|---|---|---|---|
| 3.2.1 `optimized` | 12.84 | 1.0× | ground truth | 无 |
| 3.2.2 `bmm` | 23.56 | 1.83× | ✓ 单 prompt | 低 |
| 3.2.2.5 `indexed_bmm` | 24.79 | 1.93× | ✓ 单 prompt | 高 |

🚨 **Multi-prompt gate 红色警报 + FP32 fix 验证 FAIL**(2026-05-10 02:30 / 02:35):

**完整 multi-prompt 数据**(N=14 × 3 impl × 8 token):
- bmm: 11/14 (78.6%) match,3 prompt fail (`2+2=` / `Python is` / `I love eating`)
- indexed_bmm: 11/14 完全相同 fail 模式(共享 bmm 数值路径)
- mean tps: optimized 12.53 / bmm 23.24 / indexed_bmm 24.74

**FP32 accumulator fix 实验结果 — FAIL**:
- 应用 `torch.bmm(h.float(), w.float()).to(bf16)` patch
- Smoke 测 `2+2=`:bmm 仍输出 `[20, 271, 248068, ...]`(同 fix 前)
- 已 revert
- 教训:cuBLAS 处理 BF16 input **本来就用 FP32 accumulator**(标准 tensor core 行为),explicit FP32 input 是 redundant

**真正根因(物理结论)**:
- `F.linear` → cuBLAS `gemm` tile size A,reduce schedule X
- `torch.bmm` → cuBLAS `gemmStridedBatched` tile size B,reduce schedule Y
- 两者 FP32 accumulator 都用,但 **reduce 顺序不同 → ε 级 differences**
- BF16 epsilon ~ 8e-5,单层看不出,**40 层 cascade 累积**
- router top-K 在 borderline prompt(多义/数学)时翻转 → token cascade divergence

**bmm vs optimized 在 BF16 下不可能 byte-exact** — 这是 cuBLAS kernel 选择的产物,不是 bug。

### ⚠️ R6000 剩余租期 scope hard rule(2026-05-10 user 锁定)

> **R6000 已付租期内只做"证据链 + 接入准备",不做大重构。**
>
> **可提交资产**:
> - `loader.py` 单文件 safetensors 兼容(✅ 已完成)
> - NVFP4 multi-prompt correctness harness(E4)
> - 27B 下载脚本 + 5/15 接入 prep doc
> - STRATEGY 规则整理(✅ 已完成)
>
> **明确 OFF SCOPE**(留 DGX,无租期压力):
> - functional-state refactor(LynnInferenceState 改 immutable / state-passing)
> - `torch.compile` / CUDA Graph 兼容性 refactor(`_decode_layer` 签名重构)
> - F2.1 / 2.2 / 2.3 cat/copy 深挖实验
> - Phase 3.3 Triton kernel 开发
> - B1 NVFP4 grouped GEMM kernel
>
> 起源:R6000 简单 patch GPU ceiling = 25.9 t/s(全消 cat+copy),实际可达 14-16 t/s。突破 25 必须 Python overhead refactor = 1-2 周。**不为 23 t/s 幻影续租。**

---

### Phase 3.2.x 价值重新定位

| | optimized | bmm / indexed_bmm |
|---|---|---|
| Path | cuBLAS gemm 单算子 | cuBLAS gemmStridedBatched 批算子 |
| Match HF | ✓ 单层 rel<10% 验证 | 不可能 byte-exact match optimized |
| Status | **default ground truth** | **opt-in fast path**(用户显式启用) |
| Speedup | 1.0× (12.53 t/s) | 1.85× (23.24 t/s) on deterministic prompt |
| 风险 | 无 | borderline prompt token divergence |

**结论**:bmm 永远不会 default。生产 default 锁 `optimized`。bmm 留为 `LYNN_MOE_IMPL=bmm` opt-in fast path,文档警告"deterministic prompt 才用"。

### 真正下一跳(profiler 数据驱动)

profiler 揭示 **Python orchestration = 70% of wall time**(12.7 ms GPU vs 43 ms wall per step)。Phase 3.2.x GEMM 加速只动了 30%。

| Priority | 工作 | 预期 | 数值风险 |
|---|---|---|---|
| **P0** | **Python overhead 削减**(40 层 Python loop → fused decoder / CUDA graph)| 12 → 25-30 t/s | 无(同 cuBLAS path)|
| P1 | **B1 NVFP4 grouped GEMM**(weight 4bit → mem-BW 减半)| 25-30 → 80-130 t/s | 有(NVFP4 量化误差,需 retention test) |
| P2 | bmm/indexed_bmm 留 opt-in,不 default | — | — |
| P3 | **Phase 3.3 Triton kernel 取消**(GEMM 已不是瓶颈)| — | — |

### 当前 source 状态

- ✅ `engine/full_forward.py` default = `optimized`(已 revert)
- ✅ `engine/moe_optimized.py` 原始(FP32 fix 已 revert,backup `.bak.fp32fix` 留)
- ✅ STRATEGY.md C1(本节)+ Universal review rule 同步
- ⏳ Day 5 续租 gate 重新评估:不再赌 Phase 3.2 → 25 t/s,改赌 P0 Python overhead → 25-30 t/s + B1 准备

数值正确性 alignment(单层 Phase 3.1):full_attn rel=3.14%,linear_attn rel=2.54% ✓
硬件:RTX PRO 6000 Blackwell sm_120,GDDR7 ~2 TB/s

### ⚠️ Universal review rule(2026-05-10 追加,blood lesson)

> **Single-prompt PASS 永不为任何实现背书。** 所有 default-impl 切换、production-ready 标识、merge-to-main 决策必须以 **multi-prompt exact-match gate** 为唯一依据。
>
> **Gate 最低要求**:
> - N ≥ 14 个 diverse prompt(英/中/code/math/多义续接 mix)
> - Exact-match rate = 100%(即 mismatches_count = 0)
> - 任何 borderline prompt(`2+2=` / `Python is` / `I love eating` 这类多义/数学)mismatch 立即整体 FAIL,不允许 partial credit
>
> **守夜检查防假绿**:
> - 直接检 `exact_match_rate < 1.0` 或 `mismatches_count > 0` → FAIL
> - **不信任任何 "verdict": "PASS" 字段**(可能错标或缺失)
> - **不用 single-prompt phase32_bench 做决策**
>
> 这条规则起源于 2026-05-10 Phase 3.2.2 bmm 单 prompt PASS 但 multi-prompt 78.6% match 的 self-consistent bug 假阳事件。Codex review #2 曾警告过,multi-prompt gate 是兑现保护。

---

### C2 · 在 35B winner 上跑 activation profile(5/11 中午 → 下午,~3h on Spark)
- `pruning/profile_activations.py` 跑 calibration_set v1.1(1436 prompts)
- ⚠️ **关键**:profile 用的是 35B winner(蒸馏后),不是 baseline 35B
- 输出 `activation_profile_lynn_v4_distill_35B.jsonl`

### C3 · Drop 30 expert 决策(5/11 下午,~1 min CPU)
- `pruning/decide_pruning.py` 输出 `drop_candidates_30.json` + rationale.md
- 用户 review + ✅ 才进 C4

### C4 · Recovery LoRA 训练(5/11 晚 → 5/14,~3 天 on A100)
- `pruning/training/recovery_lora_qwen36.yaml` LLaMA-Factory 配置
- 起点 = Version winner 蒸馏 LoRA 已 merged 的 35B
- 剪 30 expert → 27B → r=384 LoRA recover,Stage 1+5+4+6'+rehearsal
- 1-2 epoch,如 V9 退化 > 3% 加 epoch 或调 r=512

### C5 · Lynn-27B-A3B 数值校验 + V9 Gate(5/14 晚 → 5/15)
- Lynn engine BF16 跑 27B,跟 HF 单层 alignment
- `benchmarks/lynn_27b_vs_35b.py` V9 自动化:retention ≥ 97%(or ≥ 100% if distill 红利够)
- ✅ 通过 → ship Lynn-V4-Distill-Qwen-27B-A3B-BF16

### C6 · NVFP4 量化(5/15,~1h on A100)
- `docs/NVFP4_QUANTIZATION.md` pipeline 文档
- llmcompressor v8-RTN 路径
- 输出 `Lynn-V4-Distill-Qwen-27B-A3B-NVFP4-v8-RTN` (~14 GB)
- SGLang dev-cu13 启动验证

### C7 · 教学 + 公开 ✅(已完成,继续维护)
- 7 篇 tutorials + Zhihu 长文 + B 站脚本(待发)+ Twitter thread
- ⭐ **A/B ablation 完整报告写进 model card**:V4 Pro vs V4 Flash 蒸 27B-A3B —— **中文社区暂无人公开做过,价值高**
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
| Lynn-27B-A3B 剪枝退化 > 5% | 中 → **低**(蒸馏后冗余多 → 剪更易)| C5 gate 不过 → 加 epoch / 升 r 到 512 |
| ⚡ V4 Pro 5/31 后回原价(4x)| **高 if 拖到 6 月** | 5/12 前完成所有 V4 Pro 蒸馏 |
| 双 A100 训练 12h 没 24h 内跑完 | 中 | 同时备 cloud GPU(Lambda / RunPod 5090)backup |
| 蒸馏数据 cleaning 不够 → 学到 V4-Pro 错的 | 中 | gate 题里加 trap 测 + reject sampling |
| NVFP4 kernel 写不通(B1 阶段)| 中 | B 阶段不进,Lynn engine 留 BF16 reference |
| Qwen 4 / Qwen 5 出来 | 高(2026 年内可能)| 评估升级成本 — 大改重写 vs 维持 |
| SGLang 一直跑稳 → Lynn engine 替换 ROI 不够 | 中 | 不进 B,这是 fine 的退路 |
| Disk SHA1 cache 复杂度爆炸 | 低 | 简化版只缓存 system prompt,不全段 |
| Blackwell hardware 用户基础不增长 | 中 | 等 5090 普及(2026 H2)|

---

## 修订历史

- 2026-05-09 初版,锁定 Blackwell + NVFP4 + C → B 路线
- 2026-05-09 v2.0:**C 阶段压缩到 1 周**(36h 蒸馏 + 5/15 出 27B)+ V4 Pro/Flash A/B ablation + 利用 V4 Pro 75% off 5/31 截止促销 + 验证跟 B 阶段开发并行 → 2026 年内 Lynn-27B-A3B + Lynn engine 上生产
