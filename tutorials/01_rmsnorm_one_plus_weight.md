# 01 · RMSNorm 公式 = `(1.0 + weight) × x_norm`

## 一句话

**Qwen 3 / 3.5 / 3.6 全家的 RMSNorm 不是 Llama 风格 `weight × x_norm`,而是 `(1.0 + weight) × x_norm`。**

如果直接套 Llama 的 RMSNorm 实现,会得到一个看起来"差不多但又不对"的输出 — 数值差异在 BF16 floor 之上(~10x 量级),最终 logits 跟 vLLM 完全不像。

---

## 标准 RMSNorm(Llama / Qwen 2 / GPT-3 派系)

$$
\text{out} = \frac{x}{\sqrt{\frac{1}{D}\sum x_i^2 + \epsilon}} \cdot w
$$

代码:

```python
def rms_norm_llama(x, weight, eps=1e-6):
    var = x.pow(2).mean(-1, keepdim=True)
    x_norm = x * torch.rsqrt(var + eps)
    return x_norm * weight
```

权重 `weight` 通常初始化为 1.0(全 1 张量),训练后 deviating from 1.0。

## Qwen 3 RMSNorm

$$
\text{out} = \frac{x}{\sqrt{\frac{1}{D}\sum x_i^2 + \epsilon}} \cdot (1.0 + w)
$$

代码 (从 HF `transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py`):

```python
class Qwen3_5MoeRMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))   # ← 注意 zeros,不是 ones

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float())
        # Llama does x.to(float16) * w whilst Qwen3_5Moe is (x * w).to(float16)
        # See https://github.com/huggingface/transformers/pull/29402
        output = output * (1.0 + self.weight.float())   # ← `(1.0 + weight)`
        return output.type_as(x)
```

## 为什么这样设计

**关键:权重初始化 = 0,不是 1**。

- Llama 风格:`weight ← ones(D)` — 训练前 RMSNorm 是 identity。`weight × x = 1 × x = x`。
- Qwen 3 风格:`weight ← zeros(D)` — 训练前 RMSNorm 是 identity。`(1 + 0) × x = 1 × x = x`。

两者起点一样(identity),但 **训练时学的语义不同**:
- Llama: 学的是 "保留多少原信号"(weight 接近 1 = 接近 identity)
- Qwen 3: 学的是 "在 identity 上叠多少 delta"(weight 远离 0 = 远离 identity)

效果:Qwen 3 的 weight 在训练中典型值是 -0.5 ~ +1.0,围绕 0 对称分布,数值稳定性更好。Llama 的 weight 围绕 1.0 分布,梯度路径不一样。

来源:[HF transformers PR #29402](https://github.com/huggingface/transformers/pull/29402)

## 影响哪些 norm

Qwen 3.6 35B-A3B 里所有用 `Qwen3_5MoeRMSNorm` 的地方:

```
DecoderLayer.input_layernorm                         (40 层 × 1 = 40 处)
DecoderLayer.post_attention_layernorm                (40 层 × 1 = 40 处)
self_attn.q_norm  (Qwen3 trick: norm 在 RoPE 之前)   (10 full_attn 层)
self_attn.k_norm  (同上)                              (10 full_attn 层)
final_norm                                            (1 处)
```

**131 个 RMSNorm**,每个都是 `(1.0 + w)` 公式。任何一处直接套 Llama 实现,整条链路误差累积爆炸。

## 例外:`Qwen3_5MoeRMSNormGated` 不带 +1

linear_attention 里用的是 *gated* RMSNorm(单独的类),它的公式是:

```python
class Qwen3_5MoeRMSNormGated(nn.Module):
    def __init__(self, hidden_size, eps=1e-6, **kwargs):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))   # ← ones,不是 zeros!
        ...

    def forward(self, x, gate=None):
        x = x.float()
        var = x.pow(2).mean(-1, keepdim=True)
        x_norm = x * torch.rsqrt(var + self.variance_epsilon)
        x = self.weight * x_norm.to(input_dtype)              # ← 普通 weight × x
        x = x * F.silu(gate.to(torch.float32))                # ← 跟 silu(gate) 乘
        return x.to(input_dtype)
```

**weight 初始化为 1.0,公式回归 Llama 风格**(因为 1.0 + 0 = 1.0 跟 1.0 等价,但 init 不同导致语义不同)。

记住:
- `Qwen3_5MoeRMSNorm` (普通) → `(1.0 + w) × x_norm`,init `zeros`
- `Qwen3_5MoeRMSNormGated` (gated) → `w × x_norm × silu(gate)`,init `ones`

## Lynn engine 实现

[`engine/full_forward.py`](../engine/full_forward.py) 第 21 行:

```python
def _rms_norm(x, weight, eps=1e-6):
    """Qwen3_5MoeRMSNorm — note the `(1.0 + weight)` factor, not plain `weight`."""
    in_dtype = x.dtype
    x_f = x.float()
    var = x_f.pow(2).mean(-1, keepdim=True)
    x_n = x_f * torch.rsqrt(var + eps)
    return (x_n * (1.0 + weight.float())).to(in_dtype)
```

[`engine/qwen36_linear_attn_block.py`](../engine/qwen36_linear_attn_block.py) 里的 `rms_norm_gated`:

```python
def rms_norm_gated(x, weight, gate, eps=1e-6):
    """Qwen3_5MoeRMSNormGated — bit-equivalent to HF, NO +1 offset."""
    in_dtype = x.dtype
    x_f = x.float()
    var = x_f.pow(2).mean(-1, keepdim=True)
    x_norm = x_f * torch.rsqrt(var + eps)
    # 关键精度顺序:weight × x_norm.to(BF16) → BF16,再 × silu(gate FP32) → FP32
    x_normed_in_dtype = x_norm.to(in_dtype)
    out_low = weight * x_normed_in_dtype
    out_fp = out_low * F.silu(gate.float())
    return out_fp.to(in_dtype)
```

注意 RMSNormGated 还有个**精度顺序坑**:`weight × x_norm` 必须先 round 回 BF16 再乘 silu(gate),不能全程 FP32。如果全 FP32,会跟 HF 数值偏 ~10x(P3a 调试时实测)。

## 我们踩的坑

P1.3 端到端 forward 出"arra/arre/esta..."垃圾词时,RoPE 修了,q_proj split 也修了,还是垃圾。最后追到这里 — 所有 131 个 RMSNorm 都用了 Llama 风格 `weight × x`,改成 `(1.0 + weight) × x` 后立刻产出 " Paris"。

教训:**任何 norm 的实现先去 HF 模型卡 / 源码确认公式 — Qwen / DeepSeek / Mistral 各有变种,不能照 Llama 抄。**

## 相关资料

- [HF PR #29402 — Qwen 系 RMSNorm 提案](https://github.com/huggingface/transformers/pull/29402)
- [Lynn 引擎 commit `adac5d9`](https://github.com/MerkyorLynn/lynn-engine/commit/adac5d9) — 修这个 bug 的端到端 P1.3
