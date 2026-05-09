# NVFP4 Quantization Pipeline for Lynn-27B-A3B

> Phase 1 Week 5+ of pruning roadmap. After Recovery LoRA training completes
> and the BF16 model passes V9 gate, this pipeline produces the production
> NVFP4 weights consumed by SGLang dev-cu13(C 阶段)和 Lynn engine B1+B2(B 阶段).

## Goal

```
Input:  Lynn-27B-A3B-Recovered-BF16   (54 GB on disk)
Output: Lynn-27B-A3B-NVFP4-v8-RTN     (~14 GB on disk, ~2 GB scales)
        Lynn-27B-A3B-NVFP4-multimodal (vision encoder repacked)
```

## Why v8-RTN(not GPTQ / AWQ / modelopt full SVD)

Per memory `reference_qwen36_nvfp4_v8_rtn.md`(2026-05-05 实证):

- **v8-RTN(llmcompressor RTN)**:16 min for 35B-A3B,**production 路径**
- v7-MTQ(modelopt full): 8h+,精度同等但慢 30x
- AWQ:cuda graph 死锁(Lynn 2026-04-27 实证)
- GPTQ:NVFP4 不支持

**v8-RTN 决策**:speed 优势压倒一切,精度差距 < 1%。

## Hardware + software stack

```
Hardware:        A100 80GB / H100 80GB / 5090(BF16 → NVFP4 量化用 GPU)
Disk:            > 100 GB free(for BF16 input + intermediates + NVFP4 output)
CUDA:            12.x(量化阶段不需要 cu13)
PyTorch:         2.4+
modelopt:        0.43+
llmcompressor:   latest(含 NVFP4 RTN modifier)
transformers:    5.8+(支持 qwen3_5_moe / qwen3_5_moe_text)
```

## Step-by-step

### Step 1: 安装 llmcompressor + dependencies

```bash
pip install llmcompressor==0.4.0 modelopt==0.43.0
pip install "transformers>=5.8" accelerate safetensors
```

### Step 2: 量化(从 Recovery LoRA merged BF16 模型)

`quantize_lynn_27b_nvfp4.py`:

```python
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.transformers import oneshot
from transformers import AutoModelForImageTextToText, AutoTokenizer

model_path = "/path/to/Lynn-27B-A3B-Recovered-BF16"
output_path = "/path/to/Lynn-27B-A3B-NVFP4-v8-RTN"

model = AutoModelForImageTextToText.from_pretrained(
    model_path, dtype="bfloat16", device_map="auto", trust_remote_code=False,
)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# NVFP4 RTN config
recipe = QuantizationModifier(
    targets="Linear",
    scheme="NVFP4",
    ignore=["lm_head"],   # keep lm_head BF16 for max output quality
)

# RTN: no calibration data needed (round-to-nearest)
oneshot(
    model=model, recipe=recipe,
    output_dir=output_path,
    save_compressed=True,
)

print(f"NVFP4 model saved to {output_path}")
print(f"Approximate size: {sum(p.numel() for p in model.parameters()) / 2:.1f} GB")
```

Runtime on H100/A100: ~16 min for Lynn-27B-A3B.

### Step 3: Repack multimodal weights(vision encoder + 拼回主架构)

Per memory `reference_qwen36_nvfp4_v8_rtn.md`,llmcompressor RTN 输出的
NVFP4 模型架构字段被改成 `Qwen3_5MoeForCausalLM`,需要 patch 回
`Qwen3_5MoeForConditionalGeneration` + 拼回 vision 权重(原 BF16).

`repack_multimodal_nvfp4.py`:

```python
import json
import shutil
import torch
from pathlib import Path
from safetensors.torch import save_file, safe_open

src_bf16 = Path("/path/to/Lynn-27B-A3B-Recovered-BF16")     # has vision encoder
nvfp4_dir = Path("/path/to/Lynn-27B-A3B-NVFP4-v8-RTN")       # llmcompressor output

# 1. Patch config.json
cfg = json.loads((nvfp4_dir / "config.json").read_text())
cfg["architectures"] = ["Qwen3_5MoeForConditionalGeneration"]
# Restore multimodal config sections from src_bf16 if missing
src_cfg = json.loads((src_bf16 / "config.json").read_text())
for k in ("vision_config", "image_token_id", "video_token_id"):
    if k in src_cfg and k not in cfg:
        cfg[k] = src_cfg[k]
(nvfp4_dir / "config.json").write_text(json.dumps(cfg, indent=2))

# 2. Copy vision encoder weights from BF16 source
print("Copying vision encoder weights from BF16 source ...")
vision_weights = {}
with safe_open(src_bf16 / "outside.safetensors", framework="pt") as f:
    for k in f.keys():
        if k.startswith("model.visual."):
            vision_weights[k] = f.get_tensor(k)
print(f"  {len(vision_weights)} vision tensors")

# 3. Append to NVFP4 model.safetensors.index
nvfp4_index = json.loads((nvfp4_dir / "model.safetensors.index.json").read_text())
vision_shard = "vision_extra.safetensors"
save_file(vision_weights, str(nvfp4_dir / vision_shard))
for k in vision_weights:
    nvfp4_index["weight_map"][k] = vision_shard
(nvfp4_dir / "model.safetensors.index.json").write_text(
    json.dumps(nvfp4_index, indent=2, ensure_ascii=False))

# 4. Verify load
print("Verifying load ...")
from transformers import AutoConfig
c = AutoConfig.from_pretrained(nvfp4_dir)
print(f"  arch: {c.architectures}")
print(f"  multimodal sections present: vision={'visual' in str(c)}")
print(f"\nLynn-27B-A3B-NVFP4 ready at {nvfp4_dir}")
```

### Step 4: 验证 NVFP4 模型在 SGLang dev-cu13 启动

```bash
docker run --rm --gpus all -v /home/merkyor/models:/models -p 18099:18099 \
  lmsysorg/sglang:dev-cu13 \
  python -m sglang.launch_server \
  --model-path /models/Lynn-27B-A3B-NVFP4-v8-RTN \
  --host 0.0.0.0 --port 18099 \
  --mamba-scheduler-strategy extra_buffer \
  --page-size 64
```

⚠️ 关键 SGLang flags(per memory):
- `--mamba-scheduler-strategy extra_buffer` 必须(linear_attention layers 需要)
- `--page-size 64` 必须(KV cache 分页 size)

启动后 ~90s ready,curl 测一发:

```bash
curl http://localhost:18099/v1/chat/completions -d '{
  "model": "lynn-27b",
  "messages": [{"role":"user","content":"今天上海天气?"}],
  "max_tokens": 100
}'
```

### Step 5: V9 Gate test(同 BF16 baseline 比)

```bash
python3 benchmarks/lynn_27b_vs_35b.py \
  --baseline-url http://127.0.0.1:18002/v1 \
  --baseline-model Qwen3.6-35B-A3B-FP8 \
  --pruned-url http://127.0.0.1:18099/v1 \
  --pruned-model Lynn-27B-A3B-NVFP4 \
  --out report_nvfp4_vs_35b.md
```

通过 = retention ≥ 97% → ship 到 HuggingFace

## NVFP4 模型结构(Lynn engine B 阶段需要懂)

llmcompressor RTN 输出的 NVFP4 weight 文件结构:

```
Lynn-27B-A3B-NVFP4-v8-RTN/
├── config.json                          arch=Qwen3_5MoeForConditionalGeneration
├── tokenizer.json
├── tokenizer_config.json
├── chat_template.jinja
├── generation_config.json
├── model.safetensors.index.json         maps key → shard
├── model-00001-of-NN.safetensors        NVFP4 weights
├── ...
├── vision_extra.safetensors             BF16 vision encoder(post-repack)
└── recipe.yaml                          quantization config(reproducibility)
```

每个量化 Linear 层在 safetensors 里有:

```
{name}.weight            float8_e4m3fn   ← 实际是 NVFP4 packed,不是 FP8
                                          per llmcompressor compressed-tensors
                                          format,2 个 NVFP4 weights pack 进 1 字节
{name}.weight_scale      float8_e4m3fn   ← per-block scale, block_size=16
{name}.weight_global_scale  float32      ← per-tensor global scale
```

**Lynn engine NVFP4 loader(B1)需要做的**:
1. 读 weight 字节流(每字节 = 2 个 4-bit weights)
2. 读 weight_scale(每 16 weights 一个 FP8 scale)
3. 读 weight_global_scale(per-tensor)
4. dequantize formula:
   ```
   w_bf16[i] = (unpack_4bit(weight_bytes[i // 2], i % 2)
                × weight_scale[i // 16]
                × weight_global_scale)
   ```

5. 或保留 NVFP4 直接走 Blackwell tensor cores(B 阶段优化方向)

## Disk size 估算

```
Lynn-27B-A3B-Recovered-BF16       54 GB    (input)
Lynn-27B-A3B-NVFP4-v8-RTN         15 GB    (weights packed 4-bit + scales)
  - 4-bit packed weights:        13.5 GB
  - block scales (FP8):           1.0 GB
  - global scales (FP32):         <0.1 GB
  - vision_extra (BF16):          ~ 0.5 GB
  - 其他(config, tokenizer):     < 0.1 GB
```

## 与 SGLang dev-cu13 的兼容性

- llmcompressor `compressed-tensors` 格式 → SGLang dev-cu13 可直接加载
- vLLM 0.17 / 0.20.x:不支持(memory 第六轮 spike 验过)
- TRT-LLM:格式不同,需要 NVIDIA modelopt 路径(慢,Lynn 不走)

## Lynn engine B 阶段 NVFP4 集成路线

### B1 · NVFP4 dequant loader(C 阶段就可以写,不需 GPU 测)

```python
# engine/loader_nvfp4.py
def load_qwen36_nvfp4_layer(model_dir, layer_idx):
    """Load one layer's NVFP4 weights → dequant to BF16 in place."""
    # Read .safetensors compressed-tensors format
    # For each Linear:
    #   - weight (NVFP4 packed)
    #   - weight_scale (per-block FP8)
    #   - weight_global_scale (FP32)
    # Dequant per formula above → BF16 tensor
    # Return BF16 dict (compatible with engine/full_forward.py path)
```

跟 Phase 3.1 BF16 path 一致,**只是加载逻辑不同**。可在 Lynn engine BF16 reference 模式下跑 NVFP4 模型(性能不优但可校验 NVFP4 量化精度).

### B2 · NVFP4 native kernel(GEMM,真正 Phase B 工作)

不 dequant,直接拿 NVFP4 字节 + scales → Blackwell tensor cores。
CUTLASS NVFP4 templates(`cutlass-3.5+/include/cutlass/gemm/...nvfp4.hpp`)直接调。

详见 `docs/NVFP4_LOADER_DESIGN.md`(B 阶段 prep 待写)。

## Open questions(C → B transition 时回答)

- [ ] llmcompressor RTN 跟 modelopt v7 v8 数值差异多少?(实测决定走哪个)
- [ ] NVFP4 KV cache 在 SGLang 用了吗?Lynn engine 是否也要 NVFP4 KV?
- [ ] Lynn engine NVFP4 dequant on-the-fly vs 一次性 dequant 哪个对 Spark 273 GB/s 带宽更好?

## 相关 memory

- [reference_qwen36_nvfp4_v8_rtn.md](https://github.com/MerkyorLynn) — production NVFP4 量化路径
- [feedback_nvfp4_hybrid_attn_vllm_blocker.md](https://github.com/MerkyorLynn) — vLLM NVFP4 blocker(Lynn 不走 vLLM 路径)
- [feedback_mtp_moe_only.md](https://github.com/MerkyorLynn) — MTP 只在 MoE 加速,Lynn-27B-A3B 是 MoE 适用
- [reference_dgx_spark_llm_candidates_0501.md](https://github.com/MerkyorLynn) — NVFP4 vs MXFP4 vs AWQ 选型评估
