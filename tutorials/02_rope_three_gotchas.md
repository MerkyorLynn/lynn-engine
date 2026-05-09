# 02 · RoPE 三个连环坑

## 一句话

Qwen 3.6 的 RoPE 至少踩三个坑:**(1) `rope_theta` 在 `text_config.rope_parameters` 里,不在 `text_config.rope_theta`(那个值是 1e6 假象);(2) `partial_rotary_factor=0.25` — 只 rotate 前 64 个 head_dim;(3) 是 GPT-NeoX 半切(split-halves)风格,不是 Qwen 2 / Llama 的 even/odd 交错风格**。

加上 multimodal MROPE 干扰,几乎每条都会让你 RoPE 出错。

---

## 坑 1:rope_theta 在哪?

config.json 看起来:

```json
{
  "text_config": {
    "rope_theta": 1000000.0,         ← 这个看起来是 RoPE θ
    "rope_parameters": {
      "rope_theta": 10000000,        ← 但 HF 实际用这个
      "rope_type": "default",
      "partial_rotary_factor": 0.25,
      "mrope_interleaved": true,
      "mrope_section": [11, 11, 10]
    }
  }
}
```

**两个 `rope_theta` 值**:
- 顶层 `text_config.rope_theta = 1e6` — **不被读取**,可能是历史遗留或默认 fallback
- 嵌套 `text_config.rope_parameters.rope_theta = 1e7` — **真正使用的值**

HF transformers 5.8.0 `Qwen3_5MoeTextRotaryEmbedding.compute_default_rope_parameters`:

```python
@staticmethod
def compute_default_rope_parameters(config, ...):
    base = config.rope_parameters["rope_theta"]   # ← 从 rope_parameters 读
    partial_rotary_factor = config.rope_parameters.get("partial_rotary_factor", 1.0)
    head_dim = getattr(config, "head_dim", None) or ...
    dim = int(head_dim * partial_rotary_factor)
    ...
```

如果你写自己的 RoPE,从 `config.rope_theta` 读会拿到 `1e6`,生成的 cos/sin 频率不对,RoPE 旋转幅度不对,attention 全错。

**修法**:`tc.rope_parameters.rope_theta` 优先,fallback `tc.rope_theta`,fallback `1e6`。

```python
rope_p = tc.get("rope_parameters", {})
rope_theta = rope_p.get("rope_theta", tc.get("rope_theta", 1e6))
```

---

## 坑 2:partial_rotary_factor=0.25

`head_dim = 256`,但只有 **前 64 个 dim** rotate。后 192 个 pass-through。

```python
rotary_dim = int(head_dim * partial_rotary_factor)   # 256 × 0.25 = 64

def apply_partial_rope(x, cos, sin):
    x_rot = x[..., :rotary_dim]            # [..., 64]
    x_pass = x[..., rotary_dim:]           # [..., 192]
    x_rotated = (x_rot * cos) + (rotate_half(x_rot) * sin)
    return torch.cat([x_rotated, x_pass], dim=-1)   # 拼回 [..., 256]
```

**为什么 partial?** 经验上,只 rotate 一部分 dim 的 RoPE 在长 context 时(我们的模型 max_position_embeddings=262144)外推性能更好。Qwen 团队一贯做法。Llama 默认是 1.0(全 rotate)。

**踩坑**:如果直接 cos/sin 在全 head_dim=256 上展开,前 64 dim 算对,后 192 dim 多了不该加的 RoPE 噪声,attention 完全失效。

---

## 坑 3:GPT-NeoX 半切 vs Qwen 2 / Llama even/odd 交错

RoPE 把 Q 的每两个 dim 当一对 (x_a, x_b) 旋转 θ:

$$
\begin{pmatrix} x_a' \\ x_b' \end{pmatrix} = \begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix} \begin{pmatrix} x_a \\ x_b \end{pmatrix}
$$

但**怎么把 head_dim 里的 64 个值组成 32 对**有两种风格:

### 风格 A:Even/Odd 交错(Qwen 2 / 早期 Llama)

```
配对:(x[0], x[1]), (x[2], x[3]), (x[4], x[5]), ...
即:x_a = x[0::2],x_b = x[1::2]
```

代码:

```python
def apply_rope_qwen2(x, cos, sin):
    x_e = x[..., 0::2]   # even index
    x_o = x[..., 1::2]   # odd index
    out = torch.empty_like(x)
    out[..., 0::2] = x_e * cos[0::2] - x_o * sin[0::2]
    out[..., 1::2] = x_e * sin[0::2] + x_o * cos[0::2]
    return out
```

### 风格 B:Split-Halves(GPT-NeoX / Qwen 3+)

```
配对:(x[0], x[32]), (x[1], x[33]), (x[2], x[34]), ...
即:x_a = x[:32],x_b = x[32:]
```

代码:

```python
def rotate_half(x):
    half = x.shape[-1] // 2
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)

def apply_rope_neox(x, cos, sin):
    return (x * cos) + (rotate_half(x) * sin)
```

### 数学上等价吗?

**两种风格在同一组 cos/sin 下结果不同**!

- Even/odd 风格:cos/sin 长度 = head_dim/2,每个频率重复一次
- Split-halves 风格:cos/sin 长度 = head_dim,前后两半重复同样的频率

如果训练用 split-halves 但推理用 even/odd,等于把每个 head_dim 的 dim 错配了 — Q 跟 K 的 inner product 完全不一样,attention 失效。

**Qwen 3 / 3.5 / 3.6 全用 split-halves(GPT-NeoX 风格)**。代码确认:

```python
# transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
    q_embed = (q_rot * cos) + (rotate_half(q_rot) * sin)   # ← split-halves
    ...
```

**踩坑**:Qwen 2 时代写的旧 RoPE 代码可能仍是 even/odd 风格。直接拿来用 = 错。

---

## 坑 4(bonus):MROPE for multimodal

`mrope_interleaved=True` + `mrope_section=[11, 11, 10]` 是 **multi-modal RoPE**:position 不是单一标量,而是 (T, H, W) 三元组(temporal, height, width)。

频率分配:

```
head_dim/2 = 32 个频率 split into 3 sections:
  position 0, 3, 6, 9, ..., 30  (11 个) → 用 T 的 position
  position 1, 4, 7, 10, ..., 31 (11 个) → 用 H 的 position
  position 2, 5, 8, 11, ..., 29 (10 个) → 用 W 的 position
```

**对纯文本 input,T = H = W**(同一个 position 编码三遍),所以 MROPE collapses 到普通 RoPE。

**对图像 input**,(T, H, W) 不同(图像 token 有 grid 位置),MROPE 才生效。

如果你写的引擎暂时不接图像(像 Lynn engine 现在),可以**直接当普通 RoPE** 处理:

```python
freqs = position_ids.float()[:, :, None] * inv_freq[None, None, :]
# 对 text-only 输入跳过 apply_interleaved_mrope
emb = torch.cat([freqs, freqs], dim=-1)
cos = emb.cos()[:, None, :, :]
sin = emb.sin()[:, None, :, :]
```

接图像时再补 MROPE 处理。

---

## Lynn engine 实现

[`engine/full_forward.py`](../engine/full_forward.py) `_full_attn_forward` 第 39 行:

```python
def _full_attn_forward(h, position_ids, w, cfg):
    rope_theta = cfg["rope_theta"]                    # 从 rope_parameters 读 = 1e7
    partial = cfg["partial_rotary_factor"]            # 0.25
    rotary_dim = int(head_dim * partial)              # 64

    # 标准 RoPE on first 64 dims, GPT-NeoX 半切
    inv_freq = 1.0 / (
        rope_theta ** (torch.arange(0, rotary_dim, 2, device=h.device, dtype=torch.float32) / rotary_dim)
    )
    freqs = position_ids.float()[:, :, None] * inv_freq[None, None, :]
    emb = torch.cat([freqs, freqs], dim=-1)
    cos = emb.cos()[:, None, :, :]
    sin = emb.sin()[:, None, :, :]

    def rotate_half(x):
        half = x.shape[-1] // 2
        return torch.cat([-x[..., half:], x[..., :half]], dim=-1)

    def apply_partial_rope(x):
        x_rot, x_pass = x[..., :rotary_dim], x[..., rotary_dim:]
        c, s = cos.to(x.dtype), sin.to(x.dtype)
        x_rotated = (x_rot * c) + (rotate_half(x_rot) * s)
        return torch.cat([x_rotated, x_pass], dim=-1)

    q = apply_partial_rope(q)
    k = apply_partial_rope(k)
```

## 我们踩的坑

P1.1 测的时候用了 `rope_theta=1e6` + 全 rotate + even/odd 风格 — 完全是 Qwen 2 时代的 RoPE 套法。reference 跟 lynn 都同样错,**self-consistent 通过测试**。P1.3 跟 vLLM 比 logits 时炸出 `'arra' 'arre'` 垃圾词,才意识到 RoPE 全错。

教训:**新一代模型的 RoPE 配置一定先去 `transformers/models/<model>/modeling_<model>.py` 里搜 `class.*RotaryEmbedding`,看一遍他们的 `rope_parameters` 用法和 `apply_rotary_pos_emb` 实现**,不能照旧版本抄。

## 相关资料

- HF source: `transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py`,line ~86 (`Qwen3_5MoeTextRotaryEmbedding`) + line ~556 (`apply_rotary_pos_emb`)
- [RoFormer 原 paper](https://arxiv.org/abs/2104.09864) — RoPE 原始数学
- [GPT-NeoX-20B paper](https://arxiv.org/abs/2204.06745) §3.4 — split-halves 实现
- [Qwen2-VL paper](https://arxiv.org/abs/2409.12191) §3 — MROPE 提出
