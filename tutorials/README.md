# Qwen 3.6 35B-A3B 架构深度解析

> 一份从零实现 Lynn 推理引擎过程中,把 Qwen 3.6 跟 Llama / Qwen 2 不一样的地方挨个挖出来的笔记。
>
> 不是论文综述,是踩过坑后的总结。每篇都有数学 + HF 参考代码 + Lynn 实现 + 我们当时 ship 的 bug 复盘。

## 为什么写这个

2026 年 5 月写自己 Qwen 3.6 35B-A3B 推理引擎时,在每个"应该 5 分钟搞定"的环节都被 Qwen 3 / Qwen 3.5 / Qwen 3.6 的奇怪小怪癖搞了几个小时。这些点 HF transformers 源码里都有,但分散在 2400 行 `modeling_qwen3_5_moe.py` 里,容易看不到。

把它们集中讲一遍,**主要给以后读 / 复刻 Qwen 3.6 / 训 LoRA / 量化 / 自写引擎的人**。

## 系列文章

按"读起来意外程度"排序,**最反直觉的放最前**。

| # | 主题 | 一句话总结 |
|---|---|---|
| [01](01_rmsnorm_one_plus_weight.md) | **RMSNorm 公式 = `(1.0 + weight) × x_norm`** | Qwen 3 不是 Llama 风格 `weight × x_norm`。直接照 Llama 抄会数值偏差 ~10x |
| [02](02_rope_three_gotchas.md) | **RoPE 三个连环坑** | theta 在 `rope_parameters` 里(不是 `rope_theta`)+ partial_rotary_factor=0.25 + GPT-NeoX 半切风格 |
| [03](03_attn_output_gate.md) | **q_proj 是 2× H × head_dim — per-head 切 [q\|gate],别 flat chunk** | flat chunk 会把 head 0 的 gate 混进 head 0 的 q,自一致测试容易看不出 |
| [04](04_gated_delta_net.md) | **linear_attention = GatedDeltaNet,不是标准 attention 也不是 Mamba** | Mamba 风格的 chunk 递推 + 带 delta rule 修正 + l2norm Q/K |
| [05](05_three_invisible_bugs.md) | **三个 bug 怎么从 P1.1 通过到 P1.3 暴露** | reference + lynn 同源同错 = self-consistent 假阳。教训:reference 必须真独立 |

## 配套代码

所有讲解用的代码都在这个 repo:

```
lynn-engine/
├── engine/
│   ├── qwen36_block.py             # 早期 P1.1 实现(注意有 3 个 bug 后来在 P1.3 修)
│   ├── qwen36_linear_attn_block.py # GatedDeltaNet 移植 — 跟 HF bit-exact
│   ├── full_forward.py             # 修好的 40 层端到端 forward
│   ├── loader.py                   # FP8 e4m3 dequant
│   └── convert_fp8_to_bf16.py      # 离线 FP8→BF16 转换
└── docs/DESIGN.md                  # 整体路线图 + 实施日志
```

参考的 HF 源码:`transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py`(transformers 5.8.0 起)

## 模型基本面(快速参考)

```
Qwen 3.6 35B-A3B (multimodal)
  hidden_size:           2048
  num_hidden_layers:     40
  layer_types:           [linear×3, full×1] × 10
                         即 indices 0,1,2 linear / 3 full / 4,5,6 linear / 7 full / ...
                         共 30 linear_attention + 10 full_attention
  num_attention_heads:   16  (Q heads)
  num_key_value_heads:   2   (KV heads, GQA ratio 8:1)
  head_dim:              256
  attn_output_gate:      true   ⚠️ q_proj output = 2× H_Q × head_dim
  rope:                  partial 0.25 + theta 1e7 + GPT-NeoX 半切 + MROPE   ⚠️
  rms_norm:              (1.0 + weight) × x_norm   ⚠️
  num_experts:           256
  num_experts_per_tok:   8
  shared_expert:         有,sigmoid_gate
  linear_attention:      GatedDeltaNet (chunk_size=64, l2norm Q/K)   ⚠️
  vocab_size:            248320
  tie_word_embeddings:   false
```

## 适合谁读

- 想自己写 Qwen 3 推理引擎 / Triton kernel 的
- 训练 / 微调 / 量化 Qwen 3 想搞清楚架构怪癖的
- 看 vLLM / SGLang 内核代码在跨这些坑前想先理解的
- 单纯想知道 H1 2026 中文最强 35B-A3B 长什么样的

## 不适合谁读

- 想快速 fine-tune 跑通的(直接用 LLaMA-Factory + Unsloth)
- 想理解"什么是 transformer"的(去看 Karpathy / nanoGPT)
- 想看 paper / benchmark 数据的(去看 [Qwen 技术报告](https://huggingface.co/Qwen))

## 致谢 + 声明

实现验证用的是 HF transformers 5.8.0 + Qwen 3.6 35B-A3B-FP8 模型权重。所有 bug 都是我们写引擎时实打实踩出来的;HF 代码本身是正确的,bug 是"照 Llama 直觉抄"产生的。

如果你写自己的引擎时撞到这些坑,这个系列省你时间。
