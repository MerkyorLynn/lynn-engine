# Phase 3 · KV cache + Recurrent State Cache 设计

> Lynn engine Phase 2 完成正确性,brute-force 端到端 ~3 t/s。
> Phase 3 第一步:**incremental decode**(KV cache for full_attn + recurrent state cache for linear_attn)— 把 0.3 s/token 砍到 30-50 ms。
>
> 本文是设计文档,非实现日志。实现 commit 后 close 本文。

---

## 1. 当前性能瓶颈分析

### 1.1 brute-force re-prefill 的复杂度

`engine/full_forward.py` 的 `generate_greedy` 现状:每生成 1 个 token,重跑整个 prompt 的 prefill。

每步 forward 经历:

| 操作 | 复杂度 |
|---|---|
| 30× linear_attention(GatedDeltaNet) | O(T × chunk_size) = O(T × 64) |
| 10× full_attention(GQA + RoPE) | O(T²) |
| 40× MoE(256-expert top-8 + shared) | O(T × active_params) |

T 增长一倍,full_attention 部分 cost ~4×。生成 N tokens 总 cost = $\sum_{t=T}^{T+N} t^2 \cdot 10$ + linear/MoE 项,约 $O(N \cdot T^2)$。

实测(40 层全 BF16 resident):
- T=5:0.99 s
- T=8:0.31 s + load(已 amortized 了)
- 真实 incremental:$O(T \cdot \text{layers})$ 每步

### 1.2 incremental decode 的目标

每生成 1 个 token,只算 1 个新 token 的 forward(用 cache 替代历史 token re-compute):

| 操作 | 复杂度(incremental,T=context len) |
|---|---|
| 30× linear_attention | O(num_heads × k_dim × v_dim) ≈ O(state_size),完全独立于 T |
| 10× full_attention | O(T)(Q 是新 token 1 个,K/V cache 全长 T+1)|
| 40× MoE | O(active_params),独立于 T |

总:$O(T \cdot 10) + O(\text{state} \cdot 30) + O(\text{active} \cdot 40)$。

对 T=128, layers=40:
- 当前 brute-force 3-step:大约 0.3 s/token
- 目标 incremental:大约 30-50 ms/token(10-15 t/s on Spark single-stream)

进一步靠 Triton-fused linear_attn(Phase 3 第二步)上 60-100 t/s。靠 NVFP4 grouped expert FFN(Phase 3 第三步)上 100-150 t/s。

### 1.3 跟 vLLM 比

vLLM 35B-A3B-FP8 + SGLang+MTP 现状 60-70 t/s on Spark。本 Phase 第一步目标只到 10-15 t/s — 这是合理的中间里程碑。完整 Phase 3 完成预期 100+ t/s,跟 vLLM 持平或略高。

---

## 2. 两种 cache 类型

### 2.1 full_attention KV cache(标准)

**10 个 full_attention 层各自一个 KV cache**。每层结构:

```
K_cache: tensor [B, num_kv_heads, max_T, head_dim]   shape [1, 2, 32768, 256] @ BF16 = 64 MB/layer
V_cache: tensor [B, num_kv_heads, max_T, head_dim]   shape [1, 2, 32768, 256] @ BF16 = 64 MB/layer
```

每层 128 MB × 10 layers = **1.28 GB total** for max_T=32K,B=1。可控。

**操作**:
- prefill(T tokens): K_cache[..., :T, :] = k_proj(h_norm), V_cache[..., :T, :] = v_proj(h_norm)
- decode step (1 new token at position T): K_cache[..., T, :] = k_new, V_cache[..., T, :] = v_new
- attention: q_new[1, ...] @ K_cache[..., :T+1, :].T → attn over T+1 tokens

**RoPE 注意**:K 写入 cache 之前必须 apply RoPE(否则下次 attention 时 RoPE 位置错)。Q 是临时的不存。

### 2.2 linear_attention recurrent state cache

**30 个 linear_attention 层各自一个 recurrent_state**。状态结构:

```
recurrent_state: tensor [B, num_v_heads, head_k_dim, head_v_dim]
                   shape [1, 32, 128, 128] @ FP32 = 2 MB/layer
conv_state: tensor [B, conv_dim, conv_kernel_size - 1]
                   shape [1, 8192, 3] @ BF16 = ~50 KB/layer
```

每层 ~2 MB × 30 layers = **60 MB total**。极小。

**操作**:
- prefill(T tokens): 跑 `chunk_gated_delta_rule`,得到 last_recurrent_state + last_conv_state,存
- decode step: 跑 `recurrent_gated_delta_rule`(单 token 路径),用 cached state 推 1 步,output + new_state
- 状态更新 in-place

linear_attn 的 recurrent state 是固定大小(不随 T 增长),正是它叫"linear"的原因 — O(1) 内存 + O(1) 计算 per token。

---

## 3. 总体架构

### 3.1 LynnInferenceState 数据结构

```python
@dataclass
class LynnInferenceState:
    """Per-request inference state."""
    
    # Tracking
    seq_len: int                    # current generated length (incl. prompt)
    max_seq_len: int                # cache 最大容量
    device: str
    dtype: torch.dtype
    
    # Per-layer caches
    full_attn_kv: dict[int, tuple[Tensor, Tensor]]
        # {layer_idx: (K_cache, V_cache)} for layer_idx in {3, 7, 11, ..., 39}
        # K, V: [B, num_kv_heads, max_T, head_dim]
    
    linear_attn_state: dict[int, tuple[Tensor, Tensor]]
        # {layer_idx: (recurrent_state, conv_state)} for other layer indices
        # recurrent_state: [B, num_v_heads, head_k_dim, head_v_dim] FP32
        # conv_state: [B, conv_dim, conv_kernel_size - 1] BF16
    
    def reset(self):
        """Clear cache, prep for new prompt."""
        ...

    def memory_bytes(self) -> int:
        """Total memory footprint."""
        ...
```

预算 max_T = 32K, B = 1:
- full_attn KV: 10 × (2 tensors × 2 KV heads × 32K × 256 × 2 B) = 160 MB
  
  Actually K + V each layer: [1, 2, 32768, 256] × 2 bytes (BF16) = 32 MB; × 2 (K and V) = 64 MB; × 10 layers = 640 MB
- linear_attn state: 30 × 2 MB = 60 MB  
- conv_state: 30 × 0.05 MB = 1.5 MB
- **Total per request:** ~700 MB at max context

可以接受。多并发(B>1)按比例增长。

### 3.2 forward 模式:prefill vs decode

```python
def forward(state, input_ids, *, mode):
    """
    mode = 'prefill': input_ids has T tokens, populate cache, return logits[-1]
    mode = 'decode':  input_ids has 1 new token, use cache, return logits[-1]
    """
    if mode == 'prefill':
        # full_forward.py 的现有路径,加上 cache 写入
        for i in range(40):
            if layer_types[i] == 'full_attention':
                h, K, V = _full_attn_prefill(h, position_ids, w, cfg)
                state.full_attn_kv[i] = (K, V)
            else:  # linear_attention
                h, last_state, last_conv = _linear_attn_prefill(h, w)
                state.linear_attn_state[i] = (last_state, last_conv)
            # MoE 部分不变
        state.seq_len = T
    else:  # decode
        for i in range(40):
            if layer_types[i] == 'full_attention':
                h = _full_attn_decode(h, position_ids[..., -1:], w, cfg, *state.full_attn_kv[i])
                # in-place K/V append
            else:
                h, new_state, new_conv = _linear_attn_decode(h, w, *state.linear_attn_state[i])
                state.linear_attn_state[i] = (new_state, new_conv)
            # MoE on 1 token
        state.seq_len += 1
    
    # final norm + lm_head
    ...
```

---

## 4. 实现细节

### 4.1 Full attention decode kernel

```python
def _full_attn_decode(h_new, position_id, w, cfg, K_cache, V_cache):
    """
    h_new: [B, 1, hidden]              新 token 的 hidden state (post input_layernorm)
    position_id: [B, 1] long           = state.seq_len (新 token 位置)
    K_cache: [B, num_kv_heads, max_T, head_dim]   已分配,前 seq_len 已 fill
    V_cache: [B, num_kv_heads, max_T, head_dim]
    
    Returns: attn_out [B, 1, hidden], updates K_cache/V_cache in place at position_id
    """
    T = position_id.max().item() + 1
    
    # 1. q/k/v 投影 (B, 1, *)
    q_full = F.linear(h_new, w["self_attn.q_proj.weight"])
    k_new = F.linear(h_new, w["self_attn.k_proj.weight"])
    v_new = F.linear(h_new, w["self_attn.v_proj.weight"])
    
    # q split
    q_full_view = q_full.view(B, 1, H_Q, head_dim * 2)
    q, gate = q_full_view.chunk(2, dim=-1)
    q = q.transpose(1, 2)  # [B, H_Q, 1, head_dim]
    gate = gate.transpose(1, 2)
    
    k_new = k_new.view(B, 1, H_KV, head_dim).transpose(1, 2)
    v_new = v_new.view(B, 1, H_KV, head_dim).transpose(1, 2)
    
    # 2. q_norm, k_norm
    q = _rms_norm(q, w["self_attn.q_norm.weight"])
    k_new = _rms_norm(k_new, w["self_attn.k_norm.weight"])
    
    # 3. partial RoPE on q and k_new (position = T-1)
    q = apply_partial_rope_at(q, T - 1)
    k_new = apply_partial_rope_at(k_new, T - 1)
    
    # 4. Append k_new, v_new to cache at position T-1
    K_cache[..., T-1:T, :] = k_new   # in-place
    V_cache[..., T-1:T, :] = v_new
    
    # 5. Repeat KV for GQA
    K_full = K_cache[..., :T, :].repeat_interleave(H_Q // H_KV, dim=1)
    V_full = V_cache[..., :T, :].repeat_interleave(H_Q // H_KV, dim=1)
    
    # 6. attention: Q [B, H, 1, D] @ K [B, H, T, D]
    attn_out = F.scaled_dot_product_attention(q, K_full, V_full, is_causal=False)
    # is_causal=False because Q is just 1 token, attending to all of K_full (including current)
    
    # 7. attn_output_gate
    attn_out = attn_out * torch.sigmoid(gate.float()).to(attn_out.dtype)
    
    # 8. o_proj
    attn_out = attn_out.transpose(1, 2).contiguous().view(B, 1, H_Q * head_dim)
    return F.linear(attn_out, w["self_attn.o_proj.weight"])
```

Memory in-place 关键点:`K_cache[..., T-1:T, :] = k_new` 直接写入预分配的 buffer,不分配新张量。

### 4.2 Linear attention decode kernel

走 `torch_recurrent_gated_delta_rule`(HF 已经写好的 single-token 路径,line 315 in modeling_qwen3_5_moe.py):

```python
def _linear_attn_decode(h_new, w, recurrent_state, conv_state):
    """
    h_new: [B, 1, hidden]
    recurrent_state: [B, num_v_heads, head_k_dim, head_v_dim] FP32
    conv_state: [B, conv_dim, conv_kernel_size - 1] BF16
    
    Returns: output [B, 1, hidden], new_recurrent_state, new_conv_state
    """
    # 1. QKV proj + split (跟 prefill 同款)
    mixed = F.linear(h_new, w["linear_attn.in_proj_qkv.weight"])
    mixed = mixed.transpose(1, 2)  # [B, conv_dim, 1]
    
    # 2. Causal conv1d UPDATE — 用 conv_state + 新值滚 1 位
    # HF 的 causal_conv1d_update:
    #   把 [conv_state | new_input] 做 conv1d,取最后 1 个 output
    new_conv_state = torch.cat([conv_state, mixed], dim=-1)  # [B, conv_dim, kernel]
    out = F.conv1d(new_conv_state, w["linear_attn.conv1d.weight"], padding=0,
                   groups=CONV_DIM)
    out = F.silu(out)  # [B, conv_dim, 1]
    out = out.transpose(1, 2)  # [B, 1, conv_dim]
    
    new_conv_state = new_conv_state[..., 1:]  # 滚一位丢老头
    
    # 3. Split q/k/v 同 prefill
    # 4. z, b, a, beta, g 同 prefill (用 h_new)
    # ...
    
    # 5. recurrent_gated_delta_rule(单 token 路径)
    # 用 recurrent_state 算一步,得 new_state + output
    output, new_recurrent_state = recurrent_gated_delta_rule(
        q, k, v, g, beta, recurrent_state
    )
    
    # 6. RMSNormGated + out_proj 同 prefill
    return out_proj_output, new_recurrent_state, new_conv_state


def recurrent_gated_delta_rule(q, k, v, g, beta, S_prev):
    """单 token 递推。q,k,v shape [B, 1, num_v_heads, head_k_dim/v_dim]."""
    q = l2norm(q, dim=-1)
    k = l2norm(k, dim=-1)
    
    # 转到 [B, num_v_heads, head_k_dim/v_dim]
    q = q.transpose(1, 2).squeeze(2).float()
    k = k.transpose(1, 2).squeeze(2).float()
    v = v.transpose(1, 2).squeeze(2).float()
    g = g.transpose(1, 2).squeeze(2).float()
    beta = beta.transpose(1, 2).squeeze(2).float()
    
    scale = 1.0 / math.sqrt(q.shape[-1])
    q = q * scale
    
    # 单步递推
    g_exp = g.exp().unsqueeze(-1).unsqueeze(-1)         # [B, H, 1, 1]
    S = S_prev * g_exp                                  # decay
    kv_mem = (S * k.unsqueeze(-1)).sum(dim=-2)          # [B, H, v_dim]
    delta = (v - kv_mem) * beta.unsqueeze(-1)           # [B, H, v_dim]
    S = S + k.unsqueeze(-1) * delta.unsqueeze(-2)       # update
    out = (S * q.unsqueeze(-1)).sum(dim=-2)             # [B, H, v_dim]
    
    # 还原 shape [B, 1, H, v_dim]
    out = out.unsqueeze(1)
    return out, S
```

**关键**:state 是 FP32 内部累积(避免 BF16 累积漂移),输出最后 cast 回 BF16。

### 4.3 prefill 路径修改

prefill 跑现有 `chunk_gated_delta_rule_torch`,但**结尾要返回 last_recurrent_state**(目前是 discarded)。

修改 `chunk_gated_delta_rule_torch` 增加 `output_final_state=True` 选项(HF 原本就有,我们 port 没保留)。

---

## 5. 实施 sequence

### 5.1 阶段 1:LynnInferenceState 数据结构(2-3h)

- `engine/inference_state.py` — class definition + memory accounting + reset
- 单元测试:reset / 多 layer index / 边界

### 5.2 阶段 2:full_attention decode(3-4h)

- 改 `engine/full_forward.py::_full_attn_forward` 接受 KV cache 参数
- 加 `_full_attn_prefill`(返回 K, V 张量)和 `_full_attn_decode`(用 cache)
- 写测试:同 prompt prefill + 1-step decode 跟一次性 prefill last logits 对齐

### 5.3 阶段 3:linear_attention decode(4-6h)

- port `torch_recurrent_gated_delta_rule` 到 `engine/qwen36_linear_attn_block.py`
- 改 `lynn_linear_attn_forward` 接受 cache 参数
- 加 `lynn_linear_attn_prefill`(返回 last_state)和 `lynn_linear_attn_decode`
- 测试:同 prompt prefill + 1-step decode 跟 brute-force re-prefill 对齐(bit-exact)

### 5.4 阶段 4:整合到 generate_greedy(2h)

- 把 `generate_greedy` 改成 prefill 一次 + decode N-1 次的模式
- 测试:5 token 生成跟 brute-force 5 token 生成对齐

### 5.5 阶段 5:性能基准(1h)

- 跑 50-100 token 生成,measure t/s
- 期望:10-15 t/s on Spark single-stream
- 跟 brute-force baseline 对比

总 effort:**12-16h**。一周内可完成。

---

## 6. Phase 3 后续步骤(本 doc 范围之外)

按优先级:

1. **本 doc(KV cache + recurrent state)** — 10-15 t/s
2. **Triton-fuse `chunk_gated_delta_rule` + `recurrent_gated_delta_rule`** — 30-60 t/s
3. **CUTLASS NVFP4 grouped expert FFN** — 60-100 t/s
4. **OpenAI HTTP server wrapper**(给 brain 接入)
5. **Disk KV cache + SHA1 prefix matching**(agent prompt 99% 重复)
6. **多 batch / 并发**

---

## 7. 设计原则 / 不做什么

- **不做 PagedAttention**:Lynn 主要是 single-prompt 场景,vLLM 那种 GPU 内存 KV cache 分页用于多 batch 调度的复杂度不需要。直接 contiguous K/V buffer 简单 + 快。
- **不做 disk KV cache**(Phase 4 再做):本 doc 只关心 in-memory incremental decode。
- **不做 streaming Triton kernel**:本 doc 只补上 cache 数据流,kernel 仍走 PyTorch。Triton-fuse 是下个 milestone。
- **不做多 batch**:暂时 B=1 only,接 brain primary 时再扩。

---

## 8. 风险点

| 风险 | 概率 | 应对 |
|---|---|---|
| `recurrent_gated_delta_rule` 跟 `chunk_gated_delta_rule` prefill 后续 decode 不一致 | 中 | 阶段 3 必须严格 bit-exact 测试,差 0.5% 都不行 |
| KV cache 写入位置 off-by-one(RoPE position 错) | 中 | 阶段 2 用单元测试覆盖 prefill+decode 多种 T |
| FP32 recurrent_state 累积漂移 | 低 | HF 也是 FP32,跟 HF 同精度 |
| max_T 超限崩溃 | 低 | 在 LynnInferenceState 加 assertion |
| MoE 在 single token 下 expert 选择极不平衡(all 1 expert) | 低 | 不影响正确性,可能影响 caching expert 权重 locality(Phase 3 后期优化) |

---

## 9. 验收标准

完成本 Phase 后,以下都必须 ✓:

- [ ] `engine/inference_state.py` 单元测试 100% 通过
- [ ] `_full_attn_decode` vs brute force re-prefill last logits bit-exact(10/10 prompts)
- [ ] `_linear_attn_decode` vs brute force re-prefill last logits bit-exact(10/10 prompts)
- [ ] `generate_greedy` 用 incremental decode 5 token,跟 brute-force 5 token bit-exact 输出
- [ ] Spark 单流 decode benchmark ≥ 10 t/s(50 token 平均,prefill 不计)
- [ ] vs vLLM 同 prompt 50 token,top-K (5) 平均 overlap ≥ 4/5
- [ ] DESIGN.md §13 加 Phase 3 阶段 1 closing log

完成后 commit message:`Phase 3.1 PASSED — incremental decode (KV cache + recurrent state) at X t/s`,DESIGN.md 加 §14.1 close 本 phase。
