# Phase 3.2 复盘:为什么 78.6% 准确率反而是 bug

> [English version](PHASE3.2_POSTMORTEM_EN.md)

> **TL;DR**:bmm/indexed_bmm MoE 优化路径在单 prompt 上 100% token 一致 +
> 1.85× 提速,看起来 production-ready。Multi-prompt N=14 gate 揭穿
> 11/14 (78.6%) 实际匹配率 — 3 个 prompt 从第 2 token 起完全 divergence。
> 根因是 cuBLAS `gemm` 跟 `gemmStridedBatched` 在 BF16 下 reduce schedule
> 不同,40 层 cascade 后翻转 router top-K 选择。Default 仍锁
> `optimized`,bmm 跟 indexed_bmm 改成 **opt-in only**(`LYNN_MOE_IMPL`
> 环境变量切换)。

本文是 [`docs/STRATEGY.md`](STRATEGY.md) 新增的 **Universal review rule**
的工程证据:

> Single-prompt PASS 永不为任何实现背书。所有 default-impl 切换、
> production-ready 标识、merge-to-main 决策必须以 **multi-prompt
> exact-match gate** 为唯一依据。

---

## 1. 起点 — Phase 3.2 三档 MoE 优化

Lynn engine 的 MoE expert FFN 有三档实现,代码在
[`engine/moe_optimized.py`](../engine/moe_optimized.py):

| 档位 | 名称 | 算法 | 状态 |
|---|---|---|---|
| 3.2.1 | `optimized` | active-experts 循环 + 每 expert `F.linear` | DEFAULT |
| 3.2.2 | `bmm` | 8 active expert 一次性 stack + 3× `torch.bmm` | opt-in |
| 3.2.2.5 | `indexed_bmm` | pre-stack 256 expert + indexed bmm | opt-in |

bmm 的设计很自然:T=1 decode 时 router 选 K=8 expert,与其串行 8 次
`F.linear`,不如 `torch.stack` 8 个 expert weight 跑一次 `torch.bmm`:

```python
# engine/moe_optimized.py::moe_forward_decode_bmm (节选)
gate_stack = torch.stack([w[f"mlp.experts.{e}.gate_proj.weight"] for e in expert_ids])
up_stack   = torch.stack([w[f"mlp.experts.{e}.up_proj.weight"]   for e in expert_ids])
down_stack = torch.stack([w[f"mlp.experts.{e}.down_proj.weight"] for e in expert_ids])

h_broadcast = h_flat.unsqueeze(0).expand(K, -1, -1)
gate_out = torch.bmm(h_broadcast, gate_stack.transpose(-1, -2))
up_out   = torch.bmm(h_broadcast, up_stack.transpose(-1, -2))
inter    = F.silu(gate_out) * up_out
ffn_out  = torch.bmm(inter, down_stack.transpose(-1, -2))
```

### 1.1 单 prompt 测试看起来 production-ready

```
prompt: "The capital of France is" (max_new=10)

  optimized:    [11751, 11, 264, 3177, 34756, 364, 1141, 25438, 57902, 1680]
  bmm:          [11751, 11, 264, 3177, 34756, 364, 1141, 25438, 57902, 1680]
  indexed_bmm:  [11751, 11, 264, 3177, 34756, 364, 1141, 25438, 57902, 1680]
  ✓ 10/10 EXACT MATCH

Decode t/s:
  optimized:    12.84 t/s
  bmm:          23.56 t/s   (1.83×)
  indexed_bmm:  24.79 t/s   (1.93×)
```

到这里曾经做过(后来 revert)一个判断:把 `LYNN_MOE_IMPL` 默认值改成
`bmm`,STRATEGY.md 里把 Phase 3.2.2 标为 "production-ready baseline"。

---

## 2. Multi-prompt N=14 gate 揭穿假阳

Routine multi-prompt verification —— 14 个 diverse prompt(英/中/code/
math/极短/长),每个 8 token,3 个 impl 横向对比:

| # | prompt | optimized | bmm | indexed_bmm | match |
|---|---|---|---|---|---|
| 1 | `The capital of France is` | `[11751, 11, 264, 3177, ...]` | 同 | 同 | ✓ |
| 2 | `What is the speed of light` | ✓ | ✓ | ✓ | ✓ |
| 3 | `用一句话解释什么是 transformer` | ✓ | ✓ | ✓ | ✓ |
| 4 | `def fibonacci(n):` | ✓ | ✓ | ✓ | ✓ |
| **5** | **`2+2=`** | **`[20, 369, 264, 220, 16, 24, 23, 18]`** | **`[20, 271, 248068, 198, 8160, 579, 264, 7047]`** | **同 bmm** | **✗** |
| **6** | **`Python is`** | **`[264, 5243, 15019, 3992, 421, 369, 13177, 1429]`** | **`[264, 2172, 3992, 364, 3604, 795, 6157, 11]`** | **同 bmm** | **✗** |
| 7 | `Write a haiku about spring` | ✓ | ✓ | ✓ | ✓ |
| 8 | `今天天气` | ✓ | ✓ | ✓ | ✓ |
| 9 | `import torch` | ✓ | ✓ | ✓ | ✓ |
| 10 | `Hello world` | ✓ | ✓ | ✓ | ✓ |
| 11 | `The largest planet is` | ✓ | ✓ | ✓ | ✓ |
| **12** | **`I love eating`** | **`[3487, 13, 198, 40, 2854, 11834, 3487, 13]`** | **`[3487, 11, 694, 353, 1459, 914, 1040, 16725]`** | **同 bmm** | **✗** |
| 13 | `The quick brown fox` | ✓ | ✓ | ✓ | ✓ |
| 14 | `import numpy as` | ✓ | ✓ | ✓ | ✓ |

最终统计:

```
Multi-prompt gate (N=14, 8 tokens each):
  optimized   ground truth        12.53 t/s
  bmm         11/14 (78.6%) ✗   23.24 t/s
  indexed_bmm 11/14 (78.6%) ✗   24.74 t/s

Mismatch prompts (bmm 跟 indexed_bmm 完全相同 fail pattern):
  - 2+2=
  - Python is
  - I love eating
```

两个关键 property:

1. **token 1 一致,token 2+ 完全 divergence** —— 不是"差一个 logit"的小问题,
   而是 router 选了不同 expert,后续 token cascade 全岔
2. **bmm 跟 indexed_bmm fail 同样 3 个 prompt** —— 问题在它们共享的
   batched-matmul 路径,不是 indexed_bmm 自己的 pre-stack 逻辑

---

## 3. Pattern 分析:multi-meaning prompt 是 canary

仔细看 11 个通过 vs 3 个失败的差异:

| 通过的 prompt | 续接性质 |
|---|---|
| `The capital of France is` | → "Paris" 几乎唯一 |
| `def fibonacci(n):` | → 代码模式,top-K logits 拉得很开 |
| `import torch` / `import numpy as` | → 代码 standard |
| `今天天气` | → 中文 standard 续接 |
| `Hello world` | → exclamation / period |

| 失败的 prompt | 续接性质 |
|---|---|
| `2+2=` | → "4" / "Let me think" / "解" / "答" 多个候选 logits 接近 |
| `Python is` | → 描述性极广,top-K 候选高度接近 |
| `I love eating` | → 情感/食物多义 |

**Pattern 显然**:通过的 prompt 都是**续接 deterministic / top-K logits 拉开**;
失败的都是**续接多义 / borderline / top-K 拥挤**。

这强烈暗示:数值 ε 误差不影响 router 在 logits 拉开时的选择,**只在 top-K
logits 接近时翻转**。

把 ε 误差比作风,把 router 选 expert 比作把球扔进 K=8 个洞:
- 洞 1 离球很近,洞 2-8 远 → 风(ε 误差)吹一下,球还是进洞 1 ✓
- 洞 1 跟洞 2 距离差不多 → 风吹一下,球可能进洞 2 ✗

`2+2=` 这种 prompt 的 router 就处在第二种状态。

---

## 4. 第一直觉错了 — FP32 accumulator promote

很自然的假设:**bmm 用 BF16 accumulator,精度不够**。

加 explicit FP32 accumulator promote 试试:

```python
# Proposed fix:
h_broadcast_fp32 = h_broadcast.float()
gate_w_t_fp32    = gate_stack.transpose(-1, -2).float()
gate_out = torch.bmm(h_broadcast_fp32, gate_w_t_fp32)              # FP32 accumulator
inter    = F.silu(gate_out) * up_out_fp32
ffn_out  = torch.bmm(inter, down_w_t_fp32).to(h.dtype)             # cast back BF16 last
```

应用 patch 后 smoke test `2+2=`:

```
optimized:        [20, 369, 264, 220, 16, 24, 23, 18]
bmm (FP32 fix):   [20, 271, 248068, 198, 8160, 579, 264, 7047]
                  ^ 跟 fix 前 BF16 baseline 一字不差
```

**完全没改变**。立即 revert。

为什么 FP32 fix 无效?第一个值得记住的反直觉点:

> **cuBLAS 处理 BF16 input 时,内部本来就用 FP32 accumulator** —— 这是 BF16
> Tensor Core GEMM 的标准行为。explicit `.float()` cast 是 redundant。

也就是说 `torch.bmm(BF16, BF16) → BF16` 已经是:
- Input dtype: BF16
- Internal accumulator: FP32 (Tensor Core 自动)
- Output dtype: BF16

我手动 promote 等于把 accumulator 从 FP32 提到 FP32,啥也没干。

那真正的问题在哪?

---

## 5. 真相 — cuBLAS `gemm` vs `gemmStridedBatched` 算法 schedule 差异

cross-check `optimized` 跟 `bmm` 调用的具体 cuBLAS kernel:

| 实现 | PyTorch op | cuBLAS kernel |
|---|---|---|
| optimized | `F.linear(x, w)` (8 sequential) | `gemm` |
| bmm | `torch.bmm(x_batch, w_batch)` | `gemmStridedBatched` |

**两个不同的 cuBLAS kernel**。

cuBLAS 文档不直接保证:同样的数学输入(K=8 expert × hidden=2048 矩阵
乘法),`gemm` 单独跑 8 次 vs `gemmStridedBatched` 一次跑 8 个,**数值结果
在 LSB 级别完全相同**。

实际上一定**不**相同,因为:

1. **不同 tile size**:`gemmStridedBatched` 内部为 batch 维度做 tile 切分,
   选的 tile size 跟 `gemm` 单独跑可能不一样
2. **不同 reduce 顺序**:GEMM 内的 sum-reduction 顺序受 tile / kernel
   block 影响,两者顺序不同
3. **浮点加法不结合律**:`(a + b) + c ≠ a + (b + c)` (FP32 也是,只是
   ε 量级不同)

所以即使两个 kernel 都用 FP32 accumulator,**reduce 顺序不同** → ε 量级
differences。

BF16 epsilon ~ 8e-5(仅 7 位 mantissa)。两个 cuBLAS kernel 的 LSB 差就在
这个量级。

### 5.1 为什么单层 ε 误差变成 token divergence

单 layer 看 logits diff 几乎不可见(<10% rel diff,在 Phase 3.1 alignment
的容忍范围内)。

但是 Lynn engine Qwen3.6-35B-A3B 有 **40 层**。每层都做一次 router →
expert FFN → 加回 hidden。ε 误差通过 residual stream 被 carry forward。

40 层 cascade:

```
Layer 1 hidden state diff:    ~1e-4
Layer 5:                       ~5e-4
Layer 10:                      ~2e-3
Layer 20:                      ~1e-2
Layer 40:                      ~5e-2
```

到 layer 40 时,diff 已经在 router top-K logits 的 ε 量级。

Router top-K 在 logits 拉开时不影响选择,**borderline 时翻转一个 expert**。
选了不同 expert → MoE FFN 输出大不同 → 最终 logits 大不同 → next token
完全不同。

第一个 token 还可能侥幸一致(`2+2=` 都选了 token 20 = "5"),但选完
expert 后下一层 input 就 cascade,第二个 token 必然分叉:

```
prompt: 2+2=
optimized first token:   20 ('5')           ← 一致(router gap 大)
bmm first token:         20 ('5')           ← 一致
optimized second:        369 (' is')        ← 这里开始分叉
bmm second:              271 ('\n\n')       ← 不同 expert 选择
```

---

## 6. 物理结论 — BF16 下 bmm ≠ F.linear byte-exact

到这里 root cause 清楚了:

> **bmm 跟 F.linear 在 BF16 下不可能 byte-exact,这是 cuBLAS 算法选择的
> 产物,不是 bug。**

证据:
- 不是 accumulator 精度问题(都用 FP32 accumulator)
- 不是 broadcast vs repeat 问题(SDPA `enable_gqa=True` 的实验也证实
  byte-exact 等价)
- 是 `gemm` vs `gemmStridedBatched` 内部 reduce schedule 物理差异

修复方向必须改变。原本目标"修通 bmm 让它跟 optimized 数值一致"是
**物理不可达**。

实际采取的 fix:

1. Default `LYNN_MOE_IMPL` 回到 `optimized`(byte-exact ground truth)
2. bmm 跟 indexed_bmm 留作 **opt-in fast path**,给愿意接受 22% prompt
   variance 换 1.85× 速度的用户
3. STRATEGY.md 里 Phase 3.2.2 **不**标 production-ready

```bash
# Default(byte-exact ground truth,multi-prompt N=14 14/14 PASS):
python engine/full_forward.py --prompt "..." --mode incremental

# Opt-in fast path(~1.85× speedup,但 ~22% prompt fail rate):
LYNN_MOE_IMPL=bmm python engine/full_forward.py --prompt "..." --mode incremental
```

---

## 7. 教训 1:Multi-meaning prompt 是 ε 误差的 canary

如果当时只跑 deterministic prompt(`The capital of France is` /
`def fibonacci(n):` / `import torch`),N 跑到 100 也 100% PASS — 但
production 上一旦遇到 `2+2=` 这种数学,马上分叉。

**测试集设计教训**:

```
✅ 必含的多义 / borderline prompt:
  数学   "2+2=" / "5+7="
  描述   "Python is" / "AI 是" / "The meaning of life is"
  情感   "I love eating" / "I don't know what to"
  极短   "你好"

❌ 全 deterministic prompt:
  100% PASS 也不算过 — ε 差异看不出
```

我们 N=14 测试集的 14 个 prompt 是混合的,包括 4 个多义(2+2=,Python is,
I love eating,加 transformer 解释),这才是 78.6% 这个数字能浮现的原因。

---

## 8. 教训 2:Universal review rule — single-prompt PASS 永不背书

这是这次事故最大的工程纪律收获,落进 [`docs/STRATEGY.md`](STRATEGY.md):

> **Single-prompt PASS 永不为任何实现背书。** 所有 default-impl 切换、
> production-ready 标识、merge-to-main 决策必须以 **multi-prompt
> exact-match gate** 为唯一依据。
>
> **Gate 最低要求**:
> - N ≥ 14 个 diverse prompt(英/中/code/math/多义续接 mix)
> - Exact-match rate = 100%(即 mismatches_count = 0)
> - 任何 borderline prompt mismatch 立即整体 FAIL,不允许 partial credit
>
> **守夜检查防假绿**:
> - 直接检 `exact_match_rate < 1.0` 或 `mismatches_count > 0` → FAIL
> - **不信任任何 "verdict": "PASS" 字段**(可能错标或缺失)
> - **不用 single-prompt 测试做 default 决策**

这条规则起源于这次 bmm 单 prompt PASS 但 multi-prompt 78.6% 的
self-consistent bug 假阳事件。

跟 self-consistent bug 的另一种形式是同一类问题:**测试代码跟实现代码同源
同错,导致 fixed-point 对齐**,但偏离真值。这次 bmm 跟 optimized 数值不
同源同错,但 single prompt 偶然对齐了。N=14 才把 alignment 打破。

---

## 9. Profiler 分析 — GEMM 不是瓶颈

另一组 profile 数据(10 decode step,`torch.profiler`):

```
Self CUDA time total:   126.86 ms  →  12.7 ms/step GPU 真活
Wall clock(无 profile): 43 ms/step  → 实测 t/s ~23 (bmm)
Python orchestration:   30 ms/step ≈ 70% wall
```

GPU 端 breakdown:

| Category | 占比 | 时间/step |
|---|---|---|
| `cuBLAS gemvx`(attention 路径) | 40.7% | 5.1 ms |
| `aten::cat` + `aten::copy_` | 28% | 3.6 ms |
| Elementwise `mul` + `add` | 15% | 1.9 ms |
| `aten::mm` + `aten::bmm` | 9% | 1.1 ms |
| Norm / router / silu / topk | 7% | 0.9 ms |

**MoE GEMM(`mm` + `bmm`)只占 GPU time 的 9%。** 即使一个完美 fused 的
Triton MoE FFN kernel 最多也只能 reclaim 这 1.1 ms/step 的一部分。原本
计划 Phase 3.3(Triton MoE FFN)作为下一个主优化目标,这个优先级现在
**取消**,工作重心转移到真正的瓶颈 —— Python orchestration。

理论上限 calculation(假设全消 `cat` + `copy_`):

```
GPU = 12.7 - 2.0 (cat) - 1.6 (copy_) = ~9 ms/step
+ Python orchestration: 30 ms/step (GPU 端 patch 改不动)
= ~39 ms/step wall = ~25 t/s
```

GPU-side 简单 patch 路线最高 ~25 t/s。突破要重构 Python orchestration 层
(`_decode_layer` 签名 / LynnInferenceState mutation 模式 / CUDA Graph
兼容)—— 这是单独的工作,记进 STRATEGY.md。

---

## 10. 工程 takeaway

看起来像:

> **1.85× 提速 + 单 prompt 100% token 一致 → production-ready。**

实际上:

> **78.6% multi-prompt 匹配率 + 物理无法消除的 BF16 ε 漂移通过 40 层
> residual cascade,只在 multi-meaning prompt 上 router top-K 拥挤时
> 暴露。**

技术教训简短。工程教训才是 substantial:

- **Single-prompt PASS 不能背书任何实现。**
- **Multi-meaning prompt 是 ε 量级数值漂移的 canary。**
- **测试集必须 by design 包含 borderline cases**,不是 by accident。
- **Self-consistent test(测试 ↔ 实现共享同一假设)是比明显 bug 更糟的失败
  模式**,因为它会 ship。

Default `LYNN_MOE_IMPL=optimized`。bmm 跟 indexed_bmm 是 opt-in。STRATEGY.md
的 Universal review rule 从此适用所有 default-impl 切换决策。

---

## 11. 交叉引用

| 主题 | 位置 |
|---|---|
| Universal review rule | [`docs/STRATEGY.md`](STRATEGY.md) §"⚠️ Universal review rule" |
| Phase 3.2.x 价值重新定位 | [`docs/STRATEGY.md`](STRATEGY.md) §"Phase 3.2.x 价值重新定位" |
| 避坑指南(BF16 / cuBLAS / profiler / multi-prompt gate 陷阱) | [`docs/AVOIDANCE_GUIDE_2026-05-10.md`](AVOIDANCE_GUIDE_2026-05-10.md) |
| `LYNN_MOE_IMPL` switch | [`engine/full_forward.py`](../engine/full_forward.py) `_decode_layer` |
| Pre-stack hook(indexed_bmm)| [`engine/full_forward.py`](../engine/full_forward.py) `generate_incremental` |
| SDPA `enable_gqa` patch | [`engine/incremental_decode.py`](../engine/incremental_decode.py) `decode_full_attn` |
| 单文件 safetensors fallback | [`engine/loader.py`](../engine/loader.py) |
| Multi-prompt correctness harness | [`benchmarks/nvfp4_multi_prompt_correctness.py`](../benchmarks/nvfp4_multi_prompt_correctness.py) |

---

*在 Blackwell sm_120(RTX PRO 6000 / GDDR7)上验证,CUDA 12.8,
PyTorch 2.8.0+cu128,Qwen3.6-35B-A3B-FP8。*
