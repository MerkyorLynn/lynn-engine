# 05 · 三个 invisible bug 怎么从 P1.1 通过到 P1.3 暴露

## TL;DR

写自己的推理引擎时,最 dangerous 的不是"明显的 bug",是"reference 和实现同源同错"的 self-consistent bug。

P1.1 我们用真权重测了 10 个 full_attention 层,**rel_diff 0.000-0.117% 全 PASS**。但里面藏了 3 个 bug。**P1.3 把整个 40 层 forward 拼起来跟 vLLM 比 logits,生成 `'arra' 'arre' ' RESPONSABIL'` 垃圾词,才暴露**。

---

## 三个 bug 是什么

| # | Bug | 自一致 invisible? |
|---|---|---|
| 1 | RoPE 用 1e6 不是 1e7,partial_rotary 没用,GPT-NeoX vs even/odd 风格也错 | ✓ 完全 invisible |
| 2 | q_proj split 用 `chunk(2, dim=-1)` flat 切,没 per-head reshape | ✓ 完全 invisible |
| 3 | RMSNorm 用 `weight × x_norm` 不是 `(1.0 + weight) × x_norm` | ✓ 完全 invisible |

每个详情见:
- [01 RMSNorm `(1.0 + weight)`](01_rmsnorm_one_plus_weight.md)
- [02 RoPE 三个连环坑](02_rope_three_gotchas.md)
- [03 attn_output_gate q_proj split](03_attn_output_gate.md)

## 为什么 P1.1 通过

P1.1 测试结构:

```python
# engine/qwen36_block.py
def qwen36_reference(hidden, position_ids, w, cfg):
    """PyTorch reference."""
    # ❌ 用错的 RoPE
    # ❌ flat chunk q_proj
    # ❌ Llama 风格 RMSNorm
    return ...

def qwen36_lynn(hidden, position_ids, w, cfg, ...):
    """Lynn engine using 4 Triton kernels."""
    # 同样用错的 RoPE
    # 同样 flat chunk q_proj
    # 同样 Llama 风格 RMSNorm (Triton kernel 里实现的)
    return ...

# Test
out_ref = qwen36_reference(...)
out_lynn = qwen36_lynn(...)
assert (out_ref - out_lynn).abs().max() < 5e-2   # ✅ PASS
```

我们写的 reference 是从 HF code 抄来的,但**抄的时候照 Llama 直觉简化了 3 处**:
- RoPE:照 Qwen 2 风格的 `0::2 / 1::2` 切分
- q_proj:照"output 是 H × D"的直觉直接 chunk
- RMSNorm:照 Llama 的 `weight × x` 写

然后 lynn 实现也是照同款直觉写的。**reference 跟 lynn 错得一模一样,所以测试发现不了**。

更糟糕的是 *real weights* 这种"严格"测试也通过了 — 因为 weights 经过同样错误的 ops 后, output 在 reference 和 lynn 之间是数值一致的(每个 step 都同样错位地操作同样的权重,得出同样错位的输出)。

## P1.3 怎么暴露

P1.3 不再是 "lynn vs lynn-style reference",而是 **lynn vs production vLLM**:

```bash
# vLLM 用的是 Qwen 团队官方 patches + HF 真实代码,
# 跑 'The capital of France is' 应该输出 ' Paris'

# Lynn 端到端 forward 同 prompt
python3 engine/full_forward.py --prompt "The capital of France is"
# 输出:'arra' (top-1 logit 7.66)
#      'arre' (logit 7.22)
#      ' RESPONSABIL' (logit 6.50)
#       — 完全垃圾词
```

vLLM 不是 lynn 同源,它的实现是独立的 — Qwen 团队 official + vLLM 团队 ports。两边一定有一边对,而 vLLM 在生产用着所以是对的,**所以 lynn 错了**。

## 修 bug 的过程

我们花了 3 轮 forward 排查:

### 轮 1:RoPE

最先怀疑 RoPE 是因为多 token 比单 token 错得多。看了 `Qwen3_5MoeTextRotaryEmbedding` 代码,发现 `rope_parameters.rope_theta = 1e7`,我们用的 `text_config.rope_theta = 1e6` — 差 10x。改了 + 加 partial_rotary_factor=0.25 + GPT-NeoX 半切。

跑一下 — 还是垃圾,但是不同的垃圾(`'arre'` 取代 `'arra'` 当 top-1)。说明改的有用,但还有别的 bug。

### 轮 2:q_proj split

写 `test_decoder_layer_alignment.py` 跟 HF 单层 DecoderLayer 对照,看 attention 内部哪步偏。h_in 一致,但 attention 出来后偏 huge。

读 HF `Qwen3_5MoeAttention.forward` 源码,看到:

```python
query_states, gate = torch.chunk(
    self.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2),
    2, dim=-1,
)
```

「先 view 成 `(..., H_Q, 2*head_dim)` 再 chunk」 — 我们没做这步。修。

跑一下 — 还是垃圾,但 magnitudes 变了。说明又改了一些,还有最后一个 bug。

### 轮 3:RMSNorm

写 `engine/debug_linear_attn.py` 跟 HF 单 GatedDeltaNet 对照,逐步 dump 中间值。前 8 步全 0.000% 一致,**第 9 步 RMSNormGated 突然 689% rel diff**。

看 HF `Qwen3_5MoeRMSNorm`:

```python
output = output * (1.0 + self.weight.float())   # ⚠️ 1.0 + weight !
```

这是 PR #29402 加的(2024 年)。我们直接照 Llama 抄的 `weight × x_norm`。改了 — 全过 ✅。

跑端到端 — `' Paris' (logit 17.88)` 一次成功。

## 教训

### 1. Reference 必须是真独立的

Self-consistent test(ref + lynn 我自己写)只能 catch *bugs in lynn but not in ref*。
**Catch 不到 *bugs in both*** — 这是最危险的一类。

修法:reference 用 HF 真模块(`hf.module.forward(input)`),lynn 是 from-scratch 实现。

我们 P3a (linear_attn) 就这么做的:实例化 HF `Qwen3_5MoeGatedDeltaNet`,把权重 copy 进去,然后 lynn vs hf forward 比 output。结果 10/10 bit-exact 一次成功 — 因为 reference 是真的独立。

### 2. 比 logits,不比 hidden states

中间 hidden state 的 numerical comparison 误导性强 — 即使错了,数值 magnitude 可能依然合理。

**End-to-end logits 比对是最严格的**:错了直接 top-1 token 不一样。一致了说明所有 layers 的所有 ops 全部正确。

### 3. 抄的时候不要 "简化"

"照 Llama 直觉抄"听起来人畜无害,但每一处简化都可能引入 self-consistent bug。

新模型的 ops **必须看 modeling 源码确认**,**不照其他模型的实现抄**:
- RMSNorm:Llama / Qwen / DeepSeek 各种变种
- RoPE:theta 位置 / partial / 半切 vs 交错 / MROPE
- attn split:standard vs gate-augmented
- MoE expert 存储:per-expert vs grouped
- KV cache layout:[B,H,T,D] vs [B,T,H,D]

哪个 op 都可能跟你印象中的"标准"差一点。

### 4. 比对工具:debug script 一步一步 dump

P1.3 修第 3 个 bug 时,debug_linear_attn.py 是关键:

```python
# 加载 lynn 跟 hf 同样的权重
weights = load_layer(...)
copy_into_hf(hf_module, weights)

# 同输入,逐步 dump
h = randn(...)
for stepname, lynn_op, hf_op in [
    ("1) qkv proj", lynn_qkv, hf.qkv(h)),
    ("2) conv1d+silu", ...),
    ...
]:
    diff = (lynn_op - hf_op).abs()
    print(f"{stepname}  max={diff.max()}  rel={...}")
```

第 N 步突然偏离 = 问题在第 N-1 → N 之间。不用全部读代码,看哪步偏。

## 影响范围

P1.1 通过的"正确"声明,**实际 invalid 了**。我们 commit `adac5d9` 写明:

> P1.1 had self-consistent versions of all three [bugs] so it passed against
> its own reference, but they diverge from HF / vLLM math.

`engine/qwen36_block.py` 的 reference + lynn 现在都还有这 3 个 bug。**没改回去**,因为:
- P1.1 的目的就是验"4 个 Triton kernel 跟 PyTorch reference 一致" — 这部分仍成立(只是数学跟 HF 不一致)
- 真正端到端正确实现在 `engine/full_forward.py` 的 `_full_attn_forward`、`_layer_forward`、`_rms_norm`

如果你直接用 `engine/qwen36_block.py` 的 `qwen36_reference` 当生产代码,**会数值错**。要正确 forward,用 `engine/full_forward.py`。

## 给读者的 checklist

写 / 验自己的引擎时:

- [ ] Reference 是 HF 真模块还是自写?自写的话有没有 self-consistent bug 风险?
- [ ] 端到端测过 logits 对齐了吗?用 vLLM / HF 当 ground truth?
- [ ] RMSNorm 公式确认过了吗?是 `w × x` 还是 `(1+w) × x`?
- [ ] RoPE config 在哪个 key 下?theta 多少?partial 还是 full?半切还是交错?
- [ ] 有 MROPE 吗?text-only 怎么 collapse?
- [ ] q_proj 有 attn_output_gate 吗?切分时 per-head reshape 了吗?
- [ ] MoE expert 存储是 per-expert 还是 grouped(`[E, 2*intermediate, hidden]`)?
- [ ] 有 q_norm / k_norm / final_norm? 都改了吗?

不全 ✓ 之前别说 "通过"。
