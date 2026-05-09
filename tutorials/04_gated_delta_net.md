# 04 · linear_attention = GatedDeltaNet

## 一句话

Qwen 3.6 的 30 个"linear_attention"层不是普通 attention 也不是纯 Mamba,是 **Gated Delta Net** — Mamba 风格的 chunk 递推 + delta rule(Schlag 2021)修正项 + 强制 Q/K l2norm。代码在 `transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py` 的 `Qwen3_5MoeGatedDeltaNet`。

40 层里有 **30 层是 GatedDeltaNet,10 层是普通 GQA full_attention**(每 4 层 1 个 full,index 3, 7, 11, ..., 39)。这是 Qwen 团队 2025-Q4 提出的混合架构,linear_attention 为长 context 服务,full_attention 为局部精确度补强。

---

## 为什么要 linear_attention

普通 attention(GQA / MHA / MLA)是 **O(T²)** 内存 + 计算 cost — context 长 1 倍,attention 4 倍贵。
linear_attention 把 attention 做成**状态空间递推 (state-space recurrence)**,状态是个固定大小的矩阵,每个 token 更新一次状态、读一次状态,**O(T)** cost。

代价:linear_attention 的"记忆"压在固定大小状态里(本模型是 [16 head × 128 k_dim × 128 v_dim] = ~256K 参数的状态),不像 full attention 能精确 retrieve 任意历史 token。

混合架构的直觉:linear_attention 当 long-context 干渠,full_attention 当精确锚点。Qwen 3.6 的 4:1 比例就是这个意思。

## 数学(prefill 路径,chunk gated delta rule)

几个状态参数:

- **Q, K**:像普通 attention 但 l2-normed(`q = q / ||q||`)
- **V**:value
- **g**:per-head decay(随时间衰减状态)
- **β**:per-head delta rule 强度(0-1 sigmoid)
- **状态 S**:[batch, num_heads, k_dim, v_dim] 矩阵,初始 0

### 单步递推(逐 token,decode 路径)

每个 token 进来:

```
g_t   = -exp(A_log) * softplus(in_proj_a(h_t) + dt_bias)        # decay (FP32, per-head)
β_t   = sigmoid(in_proj_b(h_t))                                  # delta rate, per-head
q_t,k_t,v_t = chunked_qkv(h_t)                                   # standard projections + conv1d
q_t   = l2norm(q_t); k_t = l2norm(k_t)                           # ⚠️ 强制单位长

S = S * exp(g_t)                                                 # decay state
kv_mem = (S * k_t).sum(-2)                                       # current memory readout under k_t
delta = (v_t - kv_mem) * β_t                                     # delta rule correction
S = S + outer(k_t, delta)                                        # state update
out_t = (S * q_t).sum(-2)                                        # readout under q_t
```

### Chunked 路径(prefill 用,batch 多个 token 一起算)

为了 GPU 并行,把序列切成 chunk(默认 64 token),chunk 内部用矩阵代替 loop:

```
1. chunk decay matrix D[i, j] = exp(g_cum[i] - g_cum[j]) (lower triangular)
2. chunk-internal "inverse delta matrix":
   attn = -((k_β @ k.T) * D).masked_fill(diagonal, 0)
   for i in 1..chunk_size:
       attn[i, :i] = attn[i, :i] + (attn[i, :i] * attn[:i, :i]).sum(-2)
   attn = attn + I
   v_eff = attn @ v_β
3. q @ k.T * D + (q * exp(g_cum)) @ S_prev = chunk attention
4. update S_prev with cumulative kvβ products
```

完整代码在 [`transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py) 的 `torch_chunk_gated_delta_rule`(line 235),~80 行。

我们 [`engine/qwen36_linear_attn_block.py`](../engine/qwen36_linear_attn_block.py) 的 `chunk_gated_delta_rule_torch` 就是直接 port,bit-exact。

## Block 拓扑

`Qwen3_5MoeGatedDeltaNet` 的完整 forward:

```
hidden_states  [B, T, 2048]
   │
   ├─ in_proj_qkv(linear, 2048→8192)
   │  → mixed_qkv [B, T, 8192]
   │  → conv1d (kernel=4, depthwise) ── causal, silu
   │  → split → q [B,T,16,128] · k [B,T,16,128] · v [B,T,32,128]
   │  → q,k repeat_interleave×2 (匹配 v 的 32 head)
   │  → l2norm q, l2norm k
   │
   ├─ in_proj_z(linear, 2048→4096)
   │  → z [B,T,32,128]   (gate value path)
   │
   ├─ in_proj_b(linear, 2048→32)
   │  → β = sigmoid(b)
   │
   ├─ in_proj_a(linear, 2048→32)
   │  → g = -exp(A_log) * softplus(a + dt_bias)
   │
   ├─ chunk_gated_delta_rule(q, k, v, g, β)
   │  → core_attn_out [B,T,32,128]
   │
   ├─ RMSNormGated(core_attn_out, z)   ← 注意 *gated* RMSNorm,不带 +1
   │  out = w * x_norm * silu(z)
   │
   └─ out_proj(linear, 4096→2048)
      → output [B, T, 2048]
```

跟 full_attention 的关键区别:

| 维度 | full_attention(layer 3,7,...) | linear_attention(其他 30 层) |
|---|---|---|
| 状态结构 | KV cache(随 T 增长) | 固定大小 recurrent state |
| 内存复杂度 | O(T) | O(1) |
| 计算复杂度(prefill) | O(T²) | O(T) |
| 计算复杂度(decode) | O(T) | O(1) |
| 投影模式 | q_proj 2× + k/v_proj | in_proj_qkv 共享 + conv1d |
| 位置编码 | RoPE | 隐含在 conv1d + decay |
| Q/K 归一化 | q_norm/k_norm RMSNorm | l2norm(简单除以 norm)|

## 为什么 conv1d 在 QKV 上?

Mamba/RWKV 系列常见技巧 — 在 token 维度上的 depthwise causal conv1d 给 token 间一些局部信息混合,弥补 linear attention 缺少的 short-range pairwise interaction。

depthwise = 每个 channel 自己的 4-tap kernel,不跨 channel。这让 conv1d 计算极其便宜(O(D × 4)),但每个 channel 都能 "看" 自己最近 4 个 token。

```python
self.conv1d = nn.Conv1d(
    in_channels=conv_dim,        # 8192
    out_channels=conv_dim,       # 8192
    kernel_size=4,
    groups=conv_dim,             # depthwise
    padding=conv_kernel_size - 1,  # causal padding
    bias=False,
)
```

## 为什么 l2norm Q 和 K?

普通 attention 用 `softmax(QK/√d)`,softmax 自动归一化 attention 权重和为 1。
linear attention 没 softmax,K-V 累积是 `Σ outer(k_t, v_t)`,如果 k 的 magnitude 没控制,状态会爆炸。

l2norm 把每个 head 的 Q 和 K 强制单位长(`||q|| = ||k|| = 1`),保证 inner product 在 [-1, 1] 区间,delta 累积稳定。

```python
def l2norm(x, dim=-1, eps=1e-6):
    return x * torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
```

## 为什么 silu(z) 当 gate?

Mamba 同款思路 — 给一个 value-path 的"参考信号",让模型决定 attention output 跟 z 的混合权重。

`x = w * x_norm`(经过 RMSNorm 和 weight scale 的 attention output)
`x = x * silu(z)`(silu 在 z 上 squashing,让模型学一个 [0, ~z] 的开关)

效果:每个 head 输出可以被 z 抑制 / 放大。跟 attn_output_gate 在 full_attention 上的作用类似。

## 跟 Mamba / RWKV / GLA 比

| 模型 | recurrent state | decay | delta rule | gate |
|---|---|---|---|---|
| Mamba (S6) | linear state | 学习的 A | ✗ | silu(z) |
| RWKV-7 | linear state + W_dec | per-channel decay | ✓(类似)| silu |
| GLA | matrix state | 学习的 G | ✗ | silu |
| **GatedDeltaNet** | **matrix state** | **per-head g** | **✓ (用 β 控制)** | **silu(z)** |

GatedDeltaNet 跟 GLA 最近,加上了 delta rule。比 RWKV-7 的递推更"线性代数"(矩阵 outer product),比 Mamba 的 S6 更"explicit attention-like"。

## Lynn engine 验证

实现在 [`engine/qwen36_linear_attn_block.py`](../engine/qwen36_linear_attn_block.py)。

测试在 [`engine/test_lynn_linear_attn.py`](../engine/test_lynn_linear_attn.py) — 加载 Qwen 3.6 35B-A3B 的真权重,跑 layer 0/4/8/12/16/20/24/28/32/36 共 10 个 linear_attention 层,跟 HF `Qwen3_5MoeGatedDeltaNet` 比 output:

```
Lynn linear_attention validation — 10 layers
Passed: 10/10
  rel_diff: avg=0.000%  max=0.000%   ← bit-exact
```

跑命令:

```bash
docker run --rm --gpus all --user $(id -u):$(id -g) \
  -v /path/to/models:/models -v /path/to/lynn-engine:/work -w /work \
  -e PYTHONPATH=/work \
  nvcr.io/nvidia/vllm:26.03.post1-py3 \
  bash -c "pip install -q --user transformers==5.8.0 && \
           python3 engine/test_lynn_linear_attn.py --layers 0,4,8,12,16,20,24,28,32,36 --seq-len 128"
```

## 我们踩的坑

**RMSNormGated 精度顺序**(详见 [01](01_rmsnorm_one_plus_weight.md))。
全 FP32 的 norm 跟 HF 偏 ~10x。修法:`weight × x_norm` 必须先 round 回 BF16 再乘 silu(gate)。

```python
# 错的(全 FP32 → 跟 HF 偏 ~10x)
out = (weight.float() * x_norm) * F.silu(gate.float())

# 对的(跟 HF bit-exact)
x_normed_in_dtype = x_norm.to(in_dtype)
out_low = weight * x_normed_in_dtype             # BF16
out_fp = out_low * F.silu(gate.float())          # FP32 promote
return out_fp.to(in_dtype)
```

## 相关资料

- HF source: [`Qwen3_5MoeGatedDeltaNet`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py#L359) (line 359)
- 数学:[Schlag et al., 2021 — Linear Transformers Are Secretly Fast Weight Programmers](https://arxiv.org/abs/2102.11174)(delta rule 起源)
- [Yang et al. 2024 — Gated Linear Attention Transformers](https://arxiv.org/abs/2312.06635)(GLA,接 chunk_gated_delta_rule 的实现)
- [Zhang et al. 2024 — Hybrid Attention](https://arxiv.org/abs/2412.06464)(混合 linear / full attention 经验)
