# 06 · MoE Router — softmax 在 top-K 之前还是之后?

## 一句话

Qwen 3.6 的 router 是 **先 softmax 再 top-K**(不是常见的 top-K 再 softmax)。两种顺序数学上等价(top-K 选的 expert 一样、归一化后权重一样),但中间数值精度路径不同 — **如果你的 softmax 用 FP32 而 top-K 用 BF16,顺序错可能引入 ULP 漂移**。

更非显然的是:**`router_top_value /= router_top_value.sum(...)` 这步 re-normalize 不是冗余**。

---

## HF 完整 router 实现

```python
# transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py
class Qwen3_5MoeTopKRouter(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.top_k = config.num_experts_per_tok        # 8
        self.num_experts = config.num_experts          # 256
        self.hidden_dim = config.hidden_size           # 2048
        self.weight = nn.Parameter(torch.zeros(self.num_experts, self.hidden_dim))

    def forward(self, hidden_states):
        hidden_states = hidden_states.reshape(-1, self.hidden_dim)
        router_logits = F.linear(hidden_states, self.weight)   # [N_tok, 256]

        # Step 1: softmax over ALL 256 experts (in FP32 for stability)
        router_probs = F.softmax(router_logits, dtype=torch.float, dim=-1)

        # Step 2: top-K from probs
        router_top_value, router_indices = torch.topk(router_probs, self.top_k, dim=-1)

        # Step 3: re-normalize so chosen K probs sum to 1
        router_top_value /= router_top_value.sum(dim=-1, keepdim=True)
        router_top_value = router_top_value.to(router_logits.dtype)
        return router_logits, router_top_value, router_indices
```

## 跟"top-K 然后 softmax"对比

我们最初(P1.1 时期)写的 router 是常见的 top-K 然后 softmax:

```python
def router_naive(h, gate_w, top_k=8):
    logits = F.linear(h, gate_w)
    weights, indices = torch.topk(logits, top_k, dim=-1)       # top-K on raw logits
    weights = F.softmax(weights, dim=-1, dtype=torch.float32)  # softmax over only K
    return weights.to(h.dtype), indices
```

### 数学等价性证明

设 logits $\ell = [\ell_0, \ell_1, ..., \ell_{N-1}]$,top-K 索引为 $\mathcal{T}$。

**HF 风格**(softmax-then-topk-then-renorm):

$$
p_i = \frac{e^{\ell_i}}{\sum_j e^{\ell_j}}, \quad w_i^{\text{HF}} = \frac{p_i}{\sum_{k \in \mathcal{T}} p_k} = \frac{e^{\ell_i}}{\sum_{k \in \mathcal{T}} e^{\ell_k}}
$$

**naive 风格**(topk-then-softmax):

$$
w_i^{\text{naive}} = \frac{e^{\ell_i}}{\sum_{k \in \mathcal{T}} e^{\ell_k}}
$$

**完全相等**。Renormalize 后的 HF 公式跟 naive softmax-of-K 公式同一个东西。

### 为什么 HF 还要 softmax-all + renorm?

数值稳定性 + 训练目标:
1. **训练时辅助 loss**(load balancing)需要完整 256-d softmax 分布,不只是 top-K。HF 让 router_probs 走完整 softmax,可以同时输出 router_logits 和完整 router_probs。
2. **inference 时 top-K + renorm 跟 naive 数值一致**(在数学上),所以保持对齐。
3. **内存上稍贵但精度上稳定**:full softmax 在 FP32 下做,然后 top-K 选,然后 renorm 在 FP32 下完成,最后 cast 回 BF16。如果 naive 风格把 softmax 后置,top-K 可能在 BF16 logits 上做,损失精度(尤其当多个 logits 接近时,top-K 可能错位)。

## 三个易踩坑

### 坑 1:dtype 转换顺序

错的:
```python
weights = weights.to(h.dtype)   # ❌ cast 到 BF16
weights /= weights.sum(...)     # ❌ 然后才 normalize — sum 在 BF16 损精度
```

对的:
```python
weights /= weights.sum(...)      # ✅ FP32 内 normalize
weights = weights.to(h.dtype)    # ✅ 最后 cast 到 BF16
```

HF 实现是对的。我们 P1.1 写的 router_fn 也是对的(F.softmax dtype=float32 + 之后才 cast)。

### 坑 2:topk 用 logits 还是 probs?

如果用 logits 的 top-K(naive),结果和 probs 的 top-K 等价(softmax 单调)。**但**:
- 如果有 NaN/Inf 在 logits 里,softmax 会 propagate(整行变 NaN),top-K 选的就是 garbage indices
- HF 用 probs top-K 不会被一两个 inf 全行污染(softmax 把 inf 变成 1.0,其他变 0.0,top-K 仍然能选)

边缘案例,但 production 环境训过的模型偶尔会 emit overflow logits,HF 路径更鲁棒。

### 坑 3:shared expert + sigmoid_gate

Qwen 3.6 除了 256 routed experts 还有 1 个 **shared expert**(始终激活,带 sigmoid gate):

```python
class Qwen3_5MoeSparseMoeBlock(nn.Module):
    def forward(self, hidden_states):
        h_flat = hidden_states.view(-1, hidden_dim)
        shared_expert_output = self.shared_expert(h_flat)              # 1 个 always-on
        _, routing_weights, selected_experts = self.gate(h_flat)       # 256 中选 8
        expert_output = self.experts(h_flat, selected_experts, routing_weights)

        # ⚠️ shared expert 由一个 sigmoid gate 控制(per-token scalar gate)
        shared_expert_output = F.sigmoid(self.shared_expert_gate(h_flat)) * shared_expert_output
        # shared_expert_gate.weight: [1, hidden_dim] → 输出 [N_tok, 1]
        # sigmoid 后得 [N_tok, 1] 在 [0, 1] 之间 — 当作 shared expert 的强度旋钮

        expert_output = expert_output + shared_expert_output           # 简单加法,不 normalize
        return expert_output.reshape(batch_size, sequence_length, hidden_dim)
```

**关键**:
- shared expert 输出**不归一化进 routing_weights**(sigmoid 在 [0, 1],routing_weights 也归一化了,加在一起没"合并 1.0"约束 — 总输出 magnitude 跟标准 MoE 不同,这是设计 by intent)
- shared_expert 是 single 1024-intermediate dense MLP(`shared_expert_intermediate_size = 1024`),不是 routed 256 expert 之一
- `shared_expert_gate.weight` shape `[1, hidden_dim]`,只 1 个标量 logit per token

## Lynn engine 实现

[`engine/full_forward.py`](../engine/full_forward.py) `_moe_forward`:

```python
def _moe_forward(h, w, cfg):
    B, M, D = h.shape
    E = cfg["num_experts"]              # 256
    K = cfg["num_experts_per_tok"]      # 8

    h_flat = h.view(B * M, D)

    # Router: softmax-all → topk → renormalize
    router_logits = F.linear(h_flat, w["mlp.gate.weight"])
    routing_weights, expert_indices = torch.topk(router_logits, K, dim=-1)  # ⚠️ shortcut
    routing_weights = F.softmax(routing_weights, dim=-1, dtype=torch.float32).to(h.dtype)

    moe_out = torch.zeros_like(h_flat)
    for e in range(E):
        mask = (expert_indices == e)
        if not mask.any():
            continue
        token_idx, slot_idx = mask.nonzero(as_tuple=True)
        x_e = h_flat[token_idx]
        gate_e = F.linear(x_e, w[f"mlp.experts.{e}.gate_proj.weight"])
        up_e = F.linear(x_e, w[f"mlp.experts.{e}.up_proj.weight"])
        ffn_e = F.linear(F.silu(gate_e) * up_e, w[f"mlp.experts.{e}.down_proj.weight"])
        weight_e = routing_weights[token_idx, slot_idx].unsqueeze(-1)
        moe_out.index_add_(0, token_idx, ffn_e * weight_e)

    # Shared expert (always active, sigmoid-gated)
    if "mlp.shared_expert.gate_proj.weight" in w:
        gate_s = F.linear(h_flat, w["mlp.shared_expert.gate_proj.weight"])
        up_s = F.linear(h_flat, w["mlp.shared_expert.up_proj.weight"])
        shared_ffn = F.linear(F.silu(gate_s) * up_s, w["mlp.shared_expert.down_proj.weight"])
        if "mlp.shared_expert_gate.weight" in w:
            shared_gate = torch.sigmoid(F.linear(h_flat, w["mlp.shared_expert_gate.weight"]))
            shared_ffn = shared_ffn * shared_gate
        moe_out = moe_out + shared_ffn

    return moe_out.view(B, M, D)
```

注意我们 lynn 实现走的是 **topk-then-softmax** 风格(naive),不是 HF 的 softmax-then-topk。**数学等价**(见上证明),但 inference 时实测 logits 一致(P1.3 多 prompt 验证 9.8/10 top-K 重合,这级偏差在 BF16↔FP8 噪声底之下,看不出 router 偏)。

## Per-expert vs grouped 存储

HF 5.8.0 的 `Qwen3_5MoeExperts` 用 **grouped tensor** 存所有 256 expert:

```python
class Qwen3_5MoeExperts(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_up_proj = nn.Parameter(torch.empty(self.num_experts, 2 * self.intermediate_dim, self.hidden_dim))
        # shape [256, 1024, 2048] — gate_proj 跟 up_proj 拼接(top half = gate,bottom half = up)
        self.down_proj = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim, self.intermediate_dim))
        # shape [256, 2048, 512]
```

但 **safetensors 文件存储是 per-expert**:`mlp.experts.0.gate_proj.weight`、`mlp.experts.0.up_proj.weight`、`mlp.experts.0.down_proj.weight`,各 256 份。

加载时 HF 内部 stack 起来变成 grouped tensor。Lynn 不 stack,直接 per-expert 走 Python 循环 — **慢但正确**。Phase 3 优化点:用 CUTLASS grouped GEMM 一次性算 256 expert,~50× 加速。

## 跟其他 MoE 模型对比

| 模型 | top-K | softmax 时机 | shared expert | shared gate |
|---|---|---|---|---|
| Mixtral 8×7B | 2 | top-K 后 | ✗ | — |
| DeepSeek-V3 / V4 | 8 | softmax-all 后 | ✓(始终激活)| 无 sigmoid(直接相加)|
| **Qwen 3.6 35B-A3B** | **8** | **softmax-all 后** | **✓** | **per-token sigmoid scalar** |
| Step-3.5-Flash | top-1 (?) | top-K 后 | ✗ | — |

## 我们没踩的坑(但容易踩)

### Router weight 初始化

HF:`self.weight = nn.Parameter(torch.zeros(self.num_experts, self.hidden_dim))` — **全零初始化**!

意味着初始 router_logits 全相等(0),softmax 输出 uniform 1/256,top-K 选哪 8 个不可预测。需要训练让 weight 偏离 0 才有 routing 决策。

**对推理无影响**(权重已训练好),但**写训练代码 / fine-tune router 时**,初始 router 完全 random,前几个 forward 跟训练前的"intuition"不一致。

### Routing 频次极不平衡

256 expert × top-8 理论上每个 expert 平均被选 3.1% 时间。**实际**激活分布远不均匀 — 部分 expert 永远不被激活,部分被高频激活。

我们 [Phase 1 剪枝路线](https://github.com/MerkyorLynn/Lynn/blob/main/CLAUDE.md#L70)(35B → 27B,砍 30 expert)就是利用这个不均匀:**对 Lynn 用户分布跑激活画像,识别命中率 < 5% 的 niche expert(生物 / 法律 / 小语种)**,物理删除。

详见 calibration_set v1.1 的设计(880 lynn-keep + 560 lynn-drop prompts)。

## 相关资料

- HF source: `transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py` line ~765 (`Qwen3_5MoeTopKRouter`) + line ~784 (`Qwen3_5MoeSparseMoeBlock`)
- [Mixtral 8×7B paper](https://arxiv.org/abs/2401.04088) — 经典 sparse MoE router
- [DeepSeek-V3 paper](https://arxiv.org/abs/2412.19437) §2.1 — shared expert 设计起源
- [Qwen 2.5 技报](https://arxiv.org/abs/2412.15115) §3 — Qwen 团队 MoE 设计哲学
