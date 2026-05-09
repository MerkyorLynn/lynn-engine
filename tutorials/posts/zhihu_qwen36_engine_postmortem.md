# 我从零写 Qwen 3.6 35B-A3B 推理引擎,踩了 3 个 invisible bug 才跟 vLLM 数值对齐

> Lynn Engine — 一个为 Blackwell sm_12x(DGX Spark / RTX PRO 6000)从头写的 Qwen 3.6 单模型推理引擎。本文记录一周内从 4 个 Triton kernel 到 40 层端到端 forward 跟 vLLM 字符级一致的过程,以及为什么"P1.1 全部通过"实际上是个假阳。
>
> Repo:[github.com/MerkyorLynn/lynn-engine](https://github.com/MerkyorLynn/lynn-engine)(完整代码 + 5 篇细节深解)

---

## 一句话动机

Qwen 3.6 35B-A3B 在我自己的 brain 项目(Lynn,一个有写作 / 工具调用能力的私人 AI 助理)生产 ToolAbstain-31 benchmark 上跑 29/31 — 全场 25+ 模型(含 DeepSeek V4-Pro / GLM-5.1)第一。这个模型值得自己写一个推理引擎,而不是依赖 vLLM 这种"通用"框架的额外 overhead。

vLLM 在 DGX Spark 上跑 35B-A3B-FP8 当前 60-70 t/s(SGLang+MTP)。理论上限基于 Spark 273 GB/s 带宽 / 3B active params × 1B/byte = 91 t/s。**vLLM 已经吃掉 ~80% 带宽**,但 vLLM 是通用框架,有 batch scheduling / KV cache / generic op dispatch overhead。**为单一模型从头写,理论上能再吃 30-50%**。

这是 Lynn Engine 的目标:Spark 100-150 t/s,RTX PRO 6000 300-700 t/s。

但本文不是讲性能 — 是讲 **写正确的过程比写快的过程难**。

## Phase 2 完成的事

| 阶段 | 内容 | 状态 |
|---|---|---|
| Triton kernels | 4 个(attention / RoPE / RMSNorm / MoE router)BF16 ULP floor 对齐 | ✅ |
| Single-block alignment | Qwen 3.6 layer 3 weights real load,full block forward 对齐 | ✅ |
| All full_attention layers | 10 个 layer(3,7,...,39)single-block alignment | ✅ "通过" — 但!|
| **Linear_attention block** | **GatedDeltaNet 完整 port,vs HF bit-exact 10/10 layers** | ✅ |
| **40 层端到端 forward** | "The capital of France is" → " Paris" 字符级匹配 vLLM | ✅ |
| **多 prompt 验证** | 8 prompts,top-K (10) 平均 9.8/10 跟 vLLM 重合 | ✅ |
| **真实文本生成** | "The capital of France is Paris, a"(3 tokens 跟 vLLM 完全一致) | ✅ |

数值正确性达成。性能(KV cache / 自写 Triton kernel / NVFP4 grouped FFN)是 Phase 3 工作。

## 真正的故事:三个 self-consistent bug

P1.1 阶段我们用真权重测了 10 个 full_attention 层,每层 rel_diff 0.000-0.117% PASS。当时已经写了 P1.1 PASS 的 commit。

**真相**:这 10 个 PASS 都是假阳。Reference 跟 lynn 实现都同样错了 3 处,所以"自一致" — 数值上一致,但都偏离 HF / vLLM。

P1.3 把 40 层全拼起来,跟生产 vLLM 比 logits — 输出 `'arra' / 'arre' / ' RESPONSABIL'` 一片垃圾词。然后逐个排查出三个 bug。

下面具体讲。

### Bug 1:RMSNorm 公式错了

**症状**:输出垃圾词。
**根因**:`Qwen3_5MoeRMSNorm` 不是 Llama 风格。

```python
# 我们抄 Llama 写的
def rms_norm_wrong(x, weight, eps=1e-6):
    var = x.pow(2).mean(-1, keepdim=True)
    x_norm = x * torch.rsqrt(var + eps)
    return x_norm * weight   # ❌

# HF 真实实现
def rms_norm_qwen3(x, weight, eps=1e-6):
    var = x.pow(2).mean(-1, keepdim=True)
    x_norm = x * torch.rsqrt(var + eps)
    return x_norm * (1.0 + weight)   # ✅ 1.0 + weight !
```

**为什么**:Llama RMSNorm 的 weight 初始化为 `ones`,所以 `weight × x` 起点是 identity。Qwen 3 系初始化为 `zeros`,所以需要 `(1.0 + weight) × x` 起点也是 identity。两种初始化语义不同(Llama 学"保留多少",Qwen 学"在 identity 上叠多少 delta")。

**踩坑面积**:整个模型有 **131 个 RMSNorm**(40 层 input_layernorm + 40 层 post_attention_layernorm + 10 q_norm + 10 k_norm + 1 final_norm + 30 linear_attn 内部 norm)。每个都得改。

来源:[HF transformers PR #29402](https://github.com/huggingface/transformers/pull/29402)

### Bug 2:RoPE 三个连环坑

**症状**:RMSNorm 修了,还是垃圾词,但是不同的垃圾(`'arre'` 取代 `'arra'`)。
**根因**:RoPE 至少有 3 处错。

#### 坑 a:rope_theta 在哪?

config.json 顶层和嵌套里都叫 `rope_theta`,值不同:

```json
{
  "text_config": {
    "rope_theta": 1000000.0,                       // ← 这个看起来是
    "rope_parameters": {
      "rope_theta": 10000000,                       // ← 但 HF 真实用这个
      "partial_rotary_factor": 0.25,
      "mrope_interleaved": true,
      "mrope_section": [11, 11, 10]
    }
  }
}
```

HF 从 `rope_parameters` 读,我们从顶层读了 1e6 — **频率差 10x,RoPE 完全错位**。

#### 坑 b:partial_rotary_factor=0.25

`head_dim=256` 但**只 rotate 前 64 个 dim**(0.25 × 256 = 64),后 192 dim pass-through。

直接 cos/sin 在全 head_dim 上展开 → 后 192 dim 多了不该加的 RoPE 噪声 → attention 失效。

#### 坑 c:GPT-NeoX 半切 vs Qwen 2 even/odd 交错

RoPE 把 dim 两两配对旋转。怎么配对有两种:

```python
# Qwen 2 / 早期 Llama: even/odd 交错
配对: (x[0], x[1]), (x[2], x[3]), (x[4], x[5]), ...

# GPT-NeoX / Qwen 3+: 半切
配对: (x[0], x[32]), (x[1], x[33]), (x[2], x[34]), ...
def rotate_half(x):
    return torch.cat([-x[..., x.shape[-1]//2:], x[..., :x.shape[-1]//2]], dim=-1)
```

我们写了 Qwen 2 风格,Qwen 3.6 用的是 GPT-NeoX 半切。

### Bug 3:q_proj 切分方向错了

**症状**:RoPE 修了,还是垃圾词,但 magnitudes 变了。
**根因**:Qwen 3.6 的 `q_proj` 输出维度是 `2 × H_Q × head_dim`(因为有 attn_output_gate),切分必须 per-head reshape 后再 chunk。

```python
# 我们写的(❌)
q_full = F.linear(h, q_proj.weight)             # [B, M, 8192]
q, gate = q_full.chunk(2, dim=-1)               # ❌ flat 切
q = q.view(B, M, H_Q, head_dim)                 # 错位 reshape

# HF 写的(✅)
q_full_view = q_full.view(B, M, H_Q, 2 * head_dim)   # 先 per-head reshape
q, gate = q_full_view.chunk(2, dim=-1)              # ✅ per-head 内 chunk
```

q_proj 输出的 8192 个 dim 实际 layout 是:

```
[h0_q(256), h0_g(256), h1_q(256), h1_g(256), ..., h15_q(256), h15_g(256)]
```

flat chunk(2) 把它切成前 4096 当 q,后 4096 当 gate — 但前 4096 实际是 `h0_q, h0_g, h1_q, h1_g, h2_q, h2_g, h3_q, h3_g`(8 个 head 的 q+gate 混着),后 4096 是 head 8-15 的 q+gate 混着。

切完 view 成 [H_Q, head_dim] 后:
- "head 0" 的 q = h0_q ✓
- "head 1" 的 q = h0_g ✗ (本来应该是 h1_q,实际是 head 0 的 gate)
- "head 2" 的 q = h1_q ✗ (本来应该是 head 2,实际是 head 1)
- ...

**16 个 head 全部错位**。

修法:先 view 成 (H_Q, 2 × head_dim) 再 chunk last dim,这样每个 head 内自己的 [q, gate] 分开。

## 修完三个 bug:`' Paris' (logit 17.88)`

```bash
$ python3 engine/full_forward.py --prompt "The capital of France is"

prompt: 'The capital of France is'
... 40 层 forward (251s 加载 + 1s 推理) ...

=== Lynn Engine top-1 next token: 11751 (' Paris') ===
Top-10:
   17.875  11751  ' Paris'      ← 同 vLLM ' Paris' (-0.54 logprob)
   16.375    264  ' a'          ← 同 vLLM ' a'
   15.500    279  ' the'        ← 同 vLLM ' the'
   ...
```

## 为什么 P1.1 测试通不出 bug

我们写了一个 PyTorch reference(`qwen36_reference`),写了一个 Triton-kernel 版的实现(`qwen36_lynn`),两个都跑同样的输入,比 output。

**reference 是从 HF 抄的,但抄的时候照"Llama 直觉"简化了 3 处**(用 Llama RMSNorm + Qwen 2 RoPE + flat chunk)。**lynn 实现也是照同款直觉写的**。

结果:reference 跟 lynn 错得**一模一样**,output 数值完全一致,**测试 PASS**。

更阴险的是:测试用的是真权重(layer 3 of Qwen 3.6 35B-A3B-FP8),不是 random — 所以"PASS"看起来非常 robust。但 **reference 错的话,跟它对齐就是错** —— 这就是 self-consistent bug 的核心问题。

## 教训

### 1. Reference 必须真独立

测试结构有两种:
- **lynn vs lynn-style reference**(自一致测试)— 只能 catch lynn 单边的 bug
- **lynn vs HF / vLLM real reference**(真独立测试)— 能 catch 全部 bug

P3a 我们改了套路:实例化 HF `Qwen3_5MoeGatedDeltaNet`,把权重 copy 进去,lynn vs HF forward 比 output。结果 10/10 layer **bit-exact**(rel_diff 0.000%)。

```python
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeGatedDeltaNet
hf = Qwen3_5MoeGatedDeltaNet(cfg, layer_idx=0).cuda().to(torch.bfloat16)
hf.load_state_dict(weights)

ref_out = hf(h)
lynn_out = lynn_linear_attn_forward(h, weights)

assert (ref_out - lynn_out).abs().max() < 5e-2
```

### 2. 比 logits,不比 hidden states

中间 hidden state 数值看起来"差不多"完全可能掩盖 bug。**End-to-end logits 比对最严格** — 错了直接 top-1 token 不一样。

```bash
# vLLM (生产真理)
$ curl -s vllm/v1/completions -d '{"prompt":"...", "max_tokens":1, "logprobs":10}'
" Paris" (logprob -0.54)

# Lynn engine
$ python3 full_forward.py --prompt "..."
top-1 = ' Paris' (logit 17.88) ✓
```

### 3. 抄的时候别"简化"

每一处"我看着这跟 Llama 一样"都是潜在的 self-consistent bug。新模型的 ops 必须看 modeling 源码确认,不照其他模型的实现抄:

- RMSNorm:Llama / Qwen / DeepSeek 各种变种(weight init / 1.0 偏置 / 是否 gated)
- RoPE:theta 位置 / partial / 半切 vs 交错 / MROPE
- attention 切分:standard vs gate-augmented(Qwen 3+)
- MoE expert 存储:per-expert vs grouped 张量
- KV cache layout:[B,H,T,D] vs [B,T,H,D]

哪个 op 都可能跟你印象中的"标准"差一点。

### 4. Debug script 一步步 dump

P1.3 修第 3 个 bug 时,关键工具是 step-by-step diff dump:

```python
# 加载 lynn 跟 hf 同样的权重
weights = load_layer(...)
copy_into_hf(hf_module, weights)

# 同输入,逐步比
h = randn(...)
for stepname, lynn_op_fn, hf_op in [
    ("1) qkv proj",       lambda h: F.linear(h, qkv_w),  hf.qkv(h)),
    ("2) conv1d+silu",    ...                          ),
    ("3a) q after split", ...                          ),
    ...
    ("9) RMSNormGated",   ...                          ),
    ("10) out_proj",      ...                          ),
]:
    diff = (lynn_op_fn(h) - hf_op).abs()
    print(f"{stepname}  max={diff.max()}  rel={...}")
```

第 N 步突然偏离 = 问题在第 N-1 → N 之间。不用全读代码,看哪步断。我们 P3a 的输出:

```
1) in_proj_qkv (B,T,conv)  rel=0.000%
2) conv1d+silu (B,T,conv)  rel=0.000%
3a) q after split          rel=0.000%
...
8) core_attn_out           rel=0.000%
9) RMSNormGated            rel=689.171%   ← BUG HERE
10) out_proj (final)       rel=53.234%
```

定位精确到 9) RMSNormGated 一行,5 分钟修好。

## 完整文章

GitHub repo:[github.com/MerkyorLynn/lynn-engine](https://github.com/MerkyorLynn/lynn-engine)

`tutorials/` 里有 5 篇深度文,每篇专门讲一处:

1. [01 RMSNorm `(1.0 + weight) × x_norm`](https://github.com/MerkyorLynn/lynn-engine/blob/main/tutorials/01_rmsnorm_one_plus_weight.md)
2. [02 RoPE 三个连环坑](https://github.com/MerkyorLynn/lynn-engine/blob/main/tutorials/02_rope_three_gotchas.md)
3. [03 attn_output_gate q_proj 2× 切分](https://github.com/MerkyorLynn/lynn-engine/blob/main/tutorials/03_attn_output_gate.md)
4. [04 linear_attention = GatedDeltaNet 完整数学](https://github.com/MerkyorLynn/lynn-engine/blob/main/tutorials/04_gated_delta_net.md)
5. [05 三个 invisible bug 复盘 + checklist](https://github.com/MerkyorLynn/lynn-engine/blob/main/tutorials/05_three_invisible_bugs.md)

## 适合谁读

- 想自己写 Qwen 3 推理引擎 / Triton kernel 的
- 训练 / 微调 / 量化 Qwen 3 想搞清楚架构怪癖的
- 看 vLLM / SGLang 内核代码在跨这些坑前想先理解的
- 单纯想知道 H1 2026 中文最强 35B-A3B 长什么样的

## 不适合谁读

- 想快速 fine-tune 跑通的(直接用 LLaMA-Factory + Unsloth)
- 想理解"什么是 transformer"的(去看 Karpathy / nanoGPT)
- 想看 paper / benchmark 数据的(去看 Qwen 技报 + 我下篇 ToolAbstain-31 论文)

---

**下篇预告**:Phase 3 — KV cache + Triton-fused linear_attention + CUTLASS NVFP4 grouped expert FFN,从今天 ~3 t/s brute-force 到 100+ t/s。

如果觉得有用,GitHub repo 顺手给个 star。有问题留言或 issue。

---

**TAG**:#LLM #推理引擎 #Qwen3 #Triton #vLLM #DGX-Spark #Blackwell #开源
