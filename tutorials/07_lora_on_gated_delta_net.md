# 07 · 给 GatedDeltaNet 加 LoRA(以及哪些组件不该加)

## 一句话

LoRA 在 Qwen 3.6 35B-A3B 上**只能 adapt 线性投影矩阵**(`in_proj_*`、`out_proj`、`q/k/v/o_proj`、`mlp.experts.*.{gate,up,down}_proj`)。GatedDeltaNet 内部的 conv1d、RMSNormGated、A_log、dt_bias、recurrent state 都**不应该加 LoRA**。

Lynn 27B-A3B 剪枝路线规划用 **r=384 LoRA on routed experts + GatedDeltaNet projections** 做 recovery 训练 — 选这个 rank 不是拍脑袋,是因为剪枝后要 recover 28% expert 的能力空间(详见下文)。

---

## 0. LoRA 基础(快速复习)

LoRA 的核心:把每个线性层 `W ∈ R^{d_out × d_in}` 的参数更新约束在一个低秩子空间:

$$
W_{\text{new}} = W + \Delta W = W + B A
$$

其中 $B \in R^{d_out × r}$,$A \in R^{r × d_in}$,$r \ll \min(d_{in}, d_{out})$。

可训练参数从 $d_{in} \cdot d_{out}$(full fine-tune)降到 $r \cdot (d_{in} + d_{out})$。

实际推理:`y = W x + B (A x) * α / r`,其中 α 是 LoRA scaling。

PEFT 库通过 `target_modules` 指定哪些 nn.Linear 加 LoRA。**关键决策:在 GatedDeltaNet 上,哪些 nn.Linear 该加,哪些不该加。**

## 1. GatedDeltaNet 的线性层清单

回顾 [`tutorials/04_gated_delta_net.md`](04_gated_delta_net.md) 里的 forward:

```
Qwen3_5MoeGatedDeltaNet.forward(h):
  in_proj_qkv: Linear(2048 → 8192)         ← 可 LoRA
  conv1d:      DepthwiseConv1d(8192 → 8192) ← 不 LoRA
  in_proj_z:   Linear(2048 → 4096)         ← 可 LoRA
  in_proj_b:   Linear(2048 → 32)           ← 可 LoRA(但太小,通常跳过)
  in_proj_a:   Linear(2048 → 32)           ← 可 LoRA(同上)
  A_log:       Parameter [32]              ← 不 LoRA
  dt_bias:     Parameter [32]              ← 不 LoRA
  norm:        RMSNormGated [128]          ← 不 LoRA
  out_proj:    Linear(4096 → 2048)         ← 可 LoRA
```

可 LoRA 的 5 个线性层(每个 GatedDeltaNet)。其中 `in_proj_b` / `in_proj_a` 输出维度只 32(num_v_heads),LoRA 几乎没意义,通常跳过。

**实际加 LoRA 的 3 个**:`in_proj_qkv`、`in_proj_z`、`out_proj`。

## 2. 为什么 conv1d 不能 LoRA

`Qwen3_5MoeGatedDeltaNet.conv1d` 是 depthwise conv1d:

```python
self.conv1d = nn.Conv1d(
    in_channels=8192, out_channels=8192,
    kernel_size=4, groups=8192,            # depthwise = 每 channel 自己的 4-tap kernel
    padding=conv_kernel_size - 1, bias=False,
)
```

参数量:`8192 × 4 = 32768`(单个 channel 的 4 个 tap weights × 8192 channels)。

理论上 PEFT 支持 `nn.Conv1d` 的 LoRA(`target_modules=["conv1d"]`),但:

1. **参数量本来就只 32K**,LoRA(r=8 都 = 8192×8 + 8×4 ≈ 65K)反而更大
2. **depthwise 没有 cross-channel mixing**,LoRA 的 `BA` 是密集矩阵,数学上不匹配 depthwise 结构

**结论**:不加。conv1d 想 fine-tune 直接 unfreeze + 全更新,32K 参数无所谓。

## 3. 为什么 A_log / dt_bias 不能 LoRA

```python
self.A_log = nn.Parameter(torch.log(A))       # shape [32]
self.dt_bias = nn.Parameter(torch.ones(32))    # shape [32]
```

这俩是**纯标量参数**(per-head 的 decay rate + bias),不是矩阵。LoRA 的 `BA` 分解需要矩阵形态。

PEFT 不支持 `nn.Parameter` 直接 LoRA(只支持 `nn.Linear`、`nn.Conv*` 等)。

**想 fine-tune?** 直接加进 trainable parameters 列表,full 更新。32 个参数,无所谓。

## 4. 为什么 RMSNormGated.weight 不能 LoRA

```python
self.norm = Qwen3_5MoeRMSNormGated(head_v_dim=128)   # weight: [128]
```

同理 — 单 vector,不是矩阵。

**结论**:RMSNorm 系参数全 unfreeze 直接 fine-tune,加起来 200 个参数,inflation 微不足道。

## 5. recurrent_state 完全无关

state 是 **运行时计算结果**,不是模型权重。无 LoRA 概念。

但有意思的衍生问题:**LoRA 改 in_proj_qkv 会间接改 recurrent state 的演化轨迹**(因为 state update 公式是 `S = S * exp(g) + outer(k, delta)`,其中 k 来自 in_proj_qkv 的 K-path)。

所以 GatedDeltaNet LoRA 的有效作用面是:
- 改 `in_proj_qkv` → 改 K, V → 改 state 内容
- 改 `in_proj_z` → 改 silu(z) gate → 改 output 强度
- 改 `out_proj` → 改最后映射回 hidden 的方式

## 6. 完整的 Qwen 3.6 35B-A3B target_modules 清单

```python
TARGET_MODULES = [
    # full_attention layers (10 layers, indices 3, 7, 11, ..., 39)
    "self_attn.q_proj",       # → 8192 × 2048
    "self_attn.k_proj",       # → 512 × 2048
    "self_attn.v_proj",       # → 512 × 2048
    "self_attn.o_proj",       # → 2048 × 4096

    # linear_attention (GatedDeltaNet) layers (30 layers)
    "linear_attn.in_proj_qkv",   # → 8192 × 2048
    "linear_attn.in_proj_z",     # → 4096 × 2048
    "linear_attn.out_proj",      # → 2048 × 4096

    # MoE per-expert (256 experts × 40 layers = 10240 expert FFN sets)
    "mlp.experts.*.gate_proj",   # 每个: 512 × 2048
    "mlp.experts.*.up_proj",     # 每个: 512 × 2048
    "mlp.experts.*.down_proj",   # 每个: 2048 × 512

    # MoE shared expert (1 per layer × 40 = 40)
    "mlp.shared_expert.gate_proj",
    "mlp.shared_expert.up_proj",
    "mlp.shared_expert.down_proj",
    "mlp.shared_expert_gate",   # → 1 × 2048

    # MoE router (1 per layer × 40 = 40)
    "mlp.gate",                  # → 256 × 2048
]

# NOT in target_modules:
# - linear_attn.conv1d / A_log / dt_bias / norm
# - self_attn.q_norm / k_norm
# - input_layernorm / post_attention_layernorm
# - embed_tokens / lm_head / final norm
```

## 7. r 该选多大?

通用经验:r ∈ [4, 64] 适合 task-specific fine-tune。但 **Lynn 27B-A3B 剪枝 recovery LoRA 选 r=384**。

理由(per [memory project_lynn_27b_pruning_plan_0509.md](https://github.com/MerkyorLynn/lynn-engine)):

剪枝丢掉 30 expert ≈ 7.5B 参数 = 模型容量的 21%。要在剩余 226 expert 上 LoRA recover 这 21% 的能力,r 必须够大。

实测经验(从 [reference_qwen36_nvfp4_v8_rtn](https://github.com/MerkyorLynn/lynn-engine) 类似规模实验):
- r=64:能 fix 简单的 distribution shift,但损失 5-10pp on 强能力题
- r=128:中等,loss 3-5pp
- r=256:大部分能 recover,loss 1-3pp
- r=384:loss < 1pp on 多数 benchmark
- r=512:边际收益小,但显存够就上

**Lynn 当前 baseline LoRA 用 r=256**,**剪枝后 recovery 计划升 r=384**。

## 8. 训练参数规模估计

按上述 target_modules 配置,r=384,Qwen 3.6 35B-A3B:

| 模块 | 数量 | 单层 LoRA 参数 |
|---|---|---|
| full_attn q/k/v/o_proj | 10 layers × 4 | r × (d_in + d_out) per matmul |
| linear_attn in_proj_qkv/z/out_proj | 30 layers × 3 | 同上 |
| MoE expert gate/up/down × 256 | 40 layers × 768 | r × (d_in + d_out) per matmul |
| MoE shared expert + gate | 40 layers × 4 | 同上 |
| Router gate | 40 layers × 1 | 同上 |

具体算下来:
- Self-attn (10 × 4): r=384, 4×(2048+8192)/2 weighted ≈ 16 MB
- Linear-attn (30 × 3): ≈ 30 MB
- MoE experts (40 × 768): r=384, 768×(2048+512)×384 / 1e6 = 754 MB **per layer**!
- MoE experts total: × 40 layers = **30 GB LoRA parameters**

⚠️ **30 GB LoRA 比基础模型 35B FP8 (35GB) 还接近!**

实际**不会对每个 expert 都加 LoRA**。常见做法:

1. **只 LoRA top-K 高频 expert**(activation profile 出热点)— 256 → 50 个
2. **shared_expert 必加**(每次都激活)
3. **router 必加**(决定剪枝后 routing 决策)
4. **routed expert 选 30 个高激活的加 LoRA**

最终 LoRA 参数 ~5-8 GB,可控。

## 9. Lynn 训练流程图(从 stage 0 到 ship pruned model)

```
Stage 0  (建 baseline,不加 LoRA)
  Lynn-35B-A3B 完整跑 Stage 1 → 2 → 5 → 4 → 6' → 7
  ⚠️ 必须先有 baseline 才能比较剪枝后退化

Stage 1  (Phase 1 Week 1: 激活画像)
  跑 calibration_set_v1.1 (1440 prompts) 通过 Lynn-35B-A3B baseline
  记录每个 expert 在每个 prompt 上的激活率
  分类: must-keep / edge / drop / redundant

Stage 2  (Phase 1 Week 2: 物理剪枝)
  砍 30 expert (7.5B params)
  35B → 27B,物理 delete weights
  router fine-tune:防止剪后 routing 决策候选少

Stage 3  (Phase 1 Week 3: Recovery LoRA)
  TARGET_MODULES = [
      "self_attn.q_proj/k_proj/v_proj/o_proj",   # full_attn × 10 layers
      "linear_attn.in_proj_qkv/z/out_proj",      # linear_attn × 30 layers
      "mlp.experts.{top_30_active}.{gate,up,down}_proj",  # 30 hot experts
      "mlp.shared_expert.{gate,up,down}_proj",
      "mlp.shared_expert_gate", "mlp.gate",
  ]
  LoRA r=384, alpha=192 (alpha = r/2 = scaling 0.5)
  训练数据: Stage 1+5 v2 + Stage 4 + Stage 6' (累积所有 Lynn stage 数据)
  rehearsal mix: 混 N-1 stage 数据 20% 防遗忘

Stage 4  (Phase 1 Week 4: Gate test → ship 决策)
  跑 V8/V9 benchmark
  vs Stage 0 baseline: 退化 ≤ 3% 就 ship
  退化 > 3% 回去再训 LoRA / 再调 r
```

## 10. 跟标准 Llama LoRA 的差异

| 维度 | Llama 2/3 LoRA | Qwen 3.6 35B-A3B LoRA |
|---|---|---|
| Target modules 数 | ~7 (q/k/v/o + gate/up/down)| ~12+ (加 linear_attn 3 + MoE per-expert 768 + router) |
| 参数总量 (r=64) | ~0.5 GB | ~5 GB(因 MoE 多)|
| GatedDeltaNet | N/A | 有 3 个新 linear |
| Expert routing | N/A | router LoRA 改剪枝后 routing 决策 |
| 推理时 LoRA 合并 | 直接加到 W | 同,但 MoE 每 expert 单独合 |

## 11. 实际代码示例(PEFT)

```python
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForImageTextToText  # ⚠️ 必须 ImageTextToText (memory 铁律)

model = AutoModelForImageTextToText.from_pretrained(
    "/path/to/Qwen3.6-35B-A3B-FP8",
    dtype=torch.bfloat16,
    device_map="auto",
)

# 1. 拿激活画像决定哪些 expert 加 LoRA
HOT_EXPERT_IDS = [3, 7, 12, 17, 22, ...]   # top 30 from calibration profile

target_modules = [
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
    "linear_attn.in_proj_qkv", "linear_attn.in_proj_z", "linear_attn.out_proj",
    "mlp.shared_expert.gate_proj", "mlp.shared_expert.up_proj", "mlp.shared_expert.down_proj",
    "mlp.shared_expert_gate", "mlp.gate",
] + [
    f"mlp.experts.{e}.{p}"
    for e in HOT_EXPERT_IDS
    for p in ("gate_proj", "up_proj", "down_proj")
]

config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=384,
    lora_alpha=192,
    target_modules=target_modules,
    lora_dropout=0.0,
    bias="none",
    init_lora_weights="gaussian",
)

model = get_peft_model(model, config)
model.print_trainable_parameters()
# trainable params: 4,XXX,XXX,XXX (≈ 5 GB) || all params: 35,XXX,XXX,XXX || trainable%: ~14%
```

## 12. ⚠️ 必须用 AutoModelForImageTextToText 不是 AutoModelForCausalLM

per memory `feedback_lora_multimodal_loading.md`(2026-05-07 踩坑):

> Qwen3.6-A3B 是 `Qwen3_5MoeForConditionalGeneration`,LoRA file keys 含 `model.language_model.layers...`,**`AutoModelForCausalLM` 自动剥语言塔丢 `language_model.` 段 → PEFT 加载完全静默失败**(logits diff = 0.000000,15/15 char-identical,没任何错)。

**修法**:用 `AutoModelForImageTextToText.from_pretrained(...)` 保留完整多模态结构,实测 logits diff mean=1.0 max=8.2 LoRA 生效。

**任何 LoRA 评估前必跑 differential forward sanity check**(开/关 adapter 比 logits diff,必须 > 0.01)。

## 相关资料

- HF source: [`Qwen3_5MoeGatedDeltaNet`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py#L359)
- Lynn 27B-A3B 剪枝路线: [memory project_lynn_27b_pruning_plan_0509.md](https://github.com/MerkyorLynn/lynn-engine)
- Calibration set v1.0: [pruning/calibration/seeds.jsonl](../pruning/calibration/seeds.jsonl)
- LoRA paper: [Hu et al. 2021 — LoRA: Low-Rank Adaptation](https://arxiv.org/abs/2106.09685)
- Lynn 多模态 LoRA 加载坑: memory `feedback_lora_multimodal_loading.md`
- Lynn 累积 LoRA 流水线规范: memory `feedback_lora_pipeline_stacking.md`
