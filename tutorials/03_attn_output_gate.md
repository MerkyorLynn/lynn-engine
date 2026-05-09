# 03 · attn_output_gate — q_proj 是 2× H × head_dim

## 一句话

Qwen 3.6 的 `self_attn.q_proj` 输出维度是 `2 × H_Q × head_dim`(普通 attention 是 `H_Q × head_dim`)。**前一半是 Q,后一半是 attn_output_gate(per-head 的 sigmoid 门控)**。

切分必须 **per-head reshape 后再 chunk**。直接 `chunk(2, dim=-1)` 在 flat 表示上切会把 head 0 的 gate 混进 head 0 的 q,**且自一致测试很难暴露**。

---

## 配置

```json
"attn_output_gate": true,
"num_attention_heads": 16,
"head_dim": 256
```

`q_proj.weight` 实际形状:`[2 × 16 × 256, 2048]` = `[8192, 2048]`(对比标准 attention 的 `[4096, 2048]`)。

## HF 实现

`Qwen3_5MoeAttention.forward` (transformers 5.8.0):

```python
input_shape = hidden_states.shape[:-1]   # [B, M]
hidden_shape = (*input_shape, -1, self.head_dim)   # [B, M, ?, 256]

query_states, gate = torch.chunk(
    self.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2),
    #                                              ^^^^^^^^^^^^^^^^^^^
    #                                              先 view 成 [B, M, H_Q, 2*head_dim]
    2,
    dim=-1,
)
gate = gate.reshape(*input_shape, -1)   # [B, M, H_Q*head_dim]
```

关键:**先 `view` 把 last dim 拆成 `(H_Q, 2 × head_dim)`,再 `chunk(2, dim=-1)` 把每个 head 的 2×head_dim 砍成 [Q, gate]**。

每个 head 内布局:

```
head 0: [q_0, q_1, ..., q_{255}, gate_0, gate_1, ..., gate_{255}]
head 1: [q_0, q_1, ..., q_{255}, gate_0, gate_1, ..., gate_{255}]
...
head 15: [q_0, q_1, ..., q_{255}, gate_0, gate_1, ..., gate_{255}]
```

整个 q_proj 输出 8192 dim 的 layout:

```
[h0_q(256), h0_g(256), h1_q(256), h1_g(256), ..., h15_q(256), h15_g(256)]
```

## 错误做法:flat chunk

如果不先 view 直接 chunk,你会得到:

```python
q_full = F.linear(h, q_proj.weight)   # [B, M, 8192]
q, gate = q_full.chunk(2, dim=-1)     # [B, M, 4096] each
# q 包含的实际是:
#   [h0_q(256), h0_g(256), h1_q(256), h1_g(256), h2_q(256), h2_g(256), h3_q(256), h3_g(256)]
# 也就是 head 0..7 的 q 和 gate 都进 q,head 8..15 的 q 和 gate 都进 gate

q.view(B, M, H_Q, head_dim)
# H_Q = 16,head_dim = 256
# 重新解释:
#   "head 0" = h0_q[0..255]    ← 实际是 h0_q ✓
#   "head 1" = h0_g[0..255]    ← 实际是 h0_g  ✗  (本来应该是 h1_q)
#   "head 2" = h1_q[0..255]    ← 实际是 h1_q  ✗
#   "head 3" = h1_g[0..255]    ← 实际是 h1_g  ✗
```

**结果**:奇数 head 都是 gate(应该是 q),偶数 head 都是错位的 q。整个 attention 的 query 完全乱了。

## 正确做法

```python
B, M = h.shape[:2]
q_full = F.linear(h, q_proj.weight)                # [B, M, 8192]

# 关键:per-head reshape 先
q_full_view = q_full.view(B, M, H_Q, head_dim * 2) # [B, M, 16, 512]

# 然后 chunk last dim per-head 内 [q | gate]
q, gate = q_full_view.chunk(2, dim=-1)             # each [B, M, 16, 256]

q = q.transpose(1, 2)                              # [B, 16, M, 256]
gate = gate.transpose(1, 2)                        # [B, 16, M, 256]
```

## attn_output_gate 怎么用

attention 出来后乘 sigmoid(gate):

```python
attn_out = scaled_dot_product_attention(q, k, v, ...)   # [B, H_Q, M, head_dim]
attn_out = attn_out * torch.sigmoid(gate.float()).to(attn_out.dtype)
attn_out = attn_out.transpose(1, 2).contiguous().view(B, M, H_Q * head_dim)
output = F.linear(attn_out, o_proj.weight)
```

直觉:
- 标准 attention:每个 head 输出 = softmax(QK/√d) @ V
- Qwen 3.6:每个 head 输出 = sigmoid(gate) ⊙ (softmax(QK/√d) @ V)

`gate` 是从 hidden state 投影出来的(跟 Q 来自同一个 projection),sigmoid 把它压到 [0, 1] 当作 attention 输出的"开关"。让模型可以**选择性抑制某个 head 的 attention 输出**。

## 跟其他模型对比

| 模型 | q_proj output dim | attn_output_gate |
|---|---|---|
| Llama 2 / 3 | H_Q × head_dim | ✗ |
| Mistral | H_Q × head_dim | ✗ |
| Qwen 2 | H_Q × head_dim | ✗ |
| **Qwen 3 / 3.5 / 3.6** | **2 × H_Q × head_dim** | **✓** |
| DeepSeek V2 / V3 | (MLA — 完全不同的结构) | ✗ |

## Lynn engine 实现

[`engine/full_forward.py`](../engine/full_forward.py) `_full_attn_forward`:

```python
q_full = F.linear(h, w["self_attn.q_proj.weight"])
k = F.linear(h, w["self_attn.k_proj.weight"])
v = F.linear(h, w["self_attn.v_proj.weight"])

# Critical: q_proj output is [B, M, H_Q*2*head_dim]. HF first reshapes to
# [B, M, H_Q, 2*head_dim] (per-head 2x slot) then chunks along last dim
# into [q, gate]. Doing chunk(2, dim=-1) on the flat representation
# incorrectly mixes head0_gate into "q" and head_last_q into "gate".
q_full_view = q_full.view(B, M, H_Q, head_dim * 2)
q, attn_output_gate = q_full_view.chunk(2, dim=-1)
q = q.transpose(1, 2)
attn_output_gate = attn_output_gate.transpose(1, 2)
k = k.view(B, M, H_KV, head_dim).transpose(1, 2)
v = v.view(B, M, H_KV, head_dim).transpose(1, 2)

# ... q_norm, k_norm, RoPE, attention ...

attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
attn_out = attn_out * torch.sigmoid(attn_output_gate.float()).to(attn_out.dtype)
attn_out = attn_out.transpose(1, 2).contiguous().view(B, M, H_Q * head_dim)
return F.linear(attn_out, w["self_attn.o_proj.weight"])
```

## 我们踩的坑

P1.1 (commit `450fd0b`) 的 `qwen36_block.py` 写的就是 flat chunk:

```python
q_full = F.linear(h_norm, w["self_attn.q_proj.weight"])
q, attn_output_gate = q_full.chunk(2, dim=-1)               # ❌ flat chunk
q = q.view(B, M, H_Q, head_dim).transpose(1, 2)             # ❌ 错位 reshape
```

**P1.1 通过测试是因为 reference 跟 lynn 都写的同款 flat chunk**,数值自一致。直到 P1.3 端到端跟 vLLM 比 logits 才暴露。

教训:**任何"切分多 head 的 stacked tensor"操作必须先 view 成 (..., H, per_head_dim) 再切 last dim**,绝不在 flat 表示上 chunk。这条规则适用 q_proj、MoE expert 切分、k_proj GQA reshape 等所有同类操作。

## 相关资料

- HF source: `transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py` line ~672 (`Qwen3_5MoeAttention.forward`)
- 类似设计:[Llama-Pro paper](https://arxiv.org/abs/2401.02415) 中提到的 sigmoid gating(虽然形式不同)
- [Qwen 2.5 技术报告](https://arxiv.org/abs/2412.15115) §3.1 — Qwen 团队首次描述 attn_output_gate 设计
