# Qwen3.6 Official MTP → Lynn Fused Converter Plan (V2)

**Date:** 2026-05-19
**Status:** CONVERTER V2 READY — block-scale FP8 dequant, 256 experts
**Model:** Qwen3.6-35B-A3B-FP8 official MTP head (1560 keys, ~815 MiB)

---

## Problem

The official Qwen3.6-35B-A3B-FP8 model ships MTP heads in `mtp.safetensors` with:
- **256 experts** (per-expert layout: `experts.{0..255}.gate_proj.weight`)
- FP8 E4M3 weights with **block-wise** `*_scale_inv` tensors (128×128 blocks)
  - Example: gate_proj weight [512, 2048] → scale_inv [4, 16]
  - Example: down_proj weight [2048, 512] → scale_inv [16, 4]
- Pre-FC norms present: `mtp.pre_fc_norm_embedding.weight`, `mtp.pre_fc_norm_hidden.weight`
- hidden=2048, intermediate=512 (inferred from tensor shapes)

Lynn's `engine/mtp_sidecar.py` expects a fused layout:
- `mtp.layers.0.mlp.experts.gate_up_proj` — shape `[256, 1024, 2048]` (fused gate+up)
- `mtp.layers.0.mlp.experts.down_proj` — shape `[256, 2048, 512]` (stacked down)
- All tensors in BF16

The blocker is the layout/FP8 adapter, not missing norms.

## Solution

### scripts/qwen36_official_mtp_inventory.py

CPU-only scan of official `mtp.safetensors`:
- Reports key_count, dtype_counts, expert_count
- Classifies keys into groups (attn, shared_expert, per-expert, norms, fc, scales)
- Detects pre_fc_norm presence
- No GPU, no torch.cuda

### scripts/qwen36_convert_official_mtp_to_lynn_fused.py

Converts official → Lynn fused:

| Official Layout | Lynn Fused Layout |
|-----------------|-------------------|
| `experts.{N}.gate_proj.weight` (FP8) + `scale_inv` [4,16] | `mtp.layers.0.mlp.experts.gate_up_proj` [256, 1024, 2048] BF16 |
| `experts.{N}.up_proj.weight` (FP8) + `scale_inv` [4,16] | (fused with gate above) |
| `experts.{N}.down_proj.weight` (FP8) + `scale_inv` [16,4] | `mtp.layers.0.mlp.experts.down_proj` [256, 2048, 512] BF16 |
| `self_attn.q_proj.weight` (FP8) + `scale_inv` | `mtp.layers.0.self_attn.q_proj.weight` BF16 |
| `shared_expert.gate_proj.weight` (FP8) + `scale_inv` | `mtp.layers.0.mlp.shared_expert.gate_proj.weight` BF16 |
| `mtp.pre_fc_norm_embedding.weight` | `mtp.pre_fc_norm_embedding.weight` (passthrough) |
| `mtp.pre_fc_norm_hidden.weight` | `mtp.pre_fc_norm_hidden.weight` (passthrough) |
| `*.fc.weight` | `mtp.fc.weight` (passthrough/dequant) |
| `*.norm.weight` | `mtp.norm.weight` (passthrough) |

### FP8 Block-Scale Dequantization (V2)

The official FP8 format uses 128×128 block quantization:
```
weight shape: [R, C]  (e.g. [512, 2048])
scale_inv shape: [R/128, C/128]  (e.g. [4, 16])

dequant:
  scale_expanded = scale_inv.repeat_interleave(128, dim=0)[:R].repeat_interleave(128, dim=1)[:, :C]
  bf16_value = fp8_value.float() * scale_expanded
```

This is NOT a per-channel scale. Each 128×128 block of the weight matrix shares
one scale factor. The converter expands the block scale to full weight shape
before pointwise multiplication.

## Commands

```bash
# Step 1: Inventory (on machine with mtp.safetensors access)
python scripts/qwen36_official_mtp_inventory.py \
  --mtp /home/merkyor/models/Qwen3.6-35B-A3B-FP8/mtp.safetensors \
  --config /home/merkyor/models/Qwen3.6-35B-A3B-FP8/config.json \
  --out reports/mtp/qwen36_official_mtp_inventory.json

# Step 2: Convert (CPU, ~30s for 815MB)
python scripts/qwen36_convert_official_mtp_to_lynn_fused.py \
  --mtp /home/merkyor/models/Qwen3.6-35B-A3B-FP8/mtp.safetensors \
  --config /home/merkyor/models/Qwen3.6-35B-A3B-FP8/config.json \
  --out /home/merkyor/models/Qwen3.6-35B-A3B-FP8/mtp_lynn_fused.safetensors

# Step 3: Verify load (requires GPU for actual forward)
python -c "
from engine.mtp_sidecar import load_mtp_sidecar, mtp_layer_weights
s, inv = load_mtp_sidecar('/path/mtp_lynn_fused.safetensors', device='cuda', dtype=torch.bfloat16)
w = mtp_layer_weights(s)
print('LOAD OK:', sorted(w.keys()))
"
```

## Output Key List (Expected — 256 experts, hidden=2048, intermediate=512)

```
mtp.fc.weight                                          bfloat16     [2048, 4096]
mtp.layers.0.input_layernorm.weight                    bfloat16     [2048]
mtp.layers.0.mlp.experts.down_proj                     bfloat16     [256, 2048, 512]
mtp.layers.0.mlp.experts.gate_up_proj                  bfloat16     [256, 1024, 2048]
mtp.layers.0.mlp.gate.weight                           bfloat16     [256, 2048]
mtp.layers.0.mlp.shared_expert.down_proj.weight        bfloat16     [2048, ...]
mtp.layers.0.mlp.shared_expert.gate_proj.weight        bfloat16     [..., 2048]
mtp.layers.0.mlp.shared_expert.up_proj.weight          bfloat16     [..., 2048]
mtp.layers.0.mlp.shared_expert_gate.weight             bfloat16     [1]
mtp.layers.0.post_attention_layernorm.weight           bfloat16     [2048]
mtp.layers.0.self_attn.k_norm.weight                   bfloat16     inferred from official header
mtp.layers.0.self_attn.k_proj.weight                   bfloat16     inferred from official header
mtp.layers.0.self_attn.o_proj.weight                   bfloat16     inferred from official header
mtp.layers.0.self_attn.q_norm.weight                   bfloat16     inferred from official header
mtp.layers.0.self_attn.q_proj.weight                   bfloat16     inferred from official header
mtp.layers.0.self_attn.v_proj.weight                   bfloat16     inferred from official header
mtp.norm.weight                                        bfloat16     [2048]
mtp.pre_fc_norm_embedding.weight                       bfloat16     [2048]
mtp.pre_fc_norm_hidden.weight                          bfloat16     [2048]
```

NOTE: Attention and shared-expert shapes must come from the official header.
The converter deliberately infers these shapes and does not trust config
defaults, because earlier 64-expert / 1408-intermediate defaults were a false
positive for this artifact.

## Metadata Written

```json
{
  "source_path": "/home/merkyor/models/Qwen3.6-35B-A3B-FP8/mtp.safetensors",
  "source_sha256_prefix": "...",
  "source_key_count": "1560",
  "detected_expert_count": "256",
  "hidden_inferred": "2048",
  "intermediate_inferred": "512",
  "scale_block": "128,128",
  "conversion": "fp8_block_scale_to_bf16_fused",
  "fp8_tensors_dequanted": "~780",
  "output_key_count": "19",
  "converter": "qwen36_convert_official_mtp_to_lynn_fused.py_v2"
}
```

## Self-Test

```bash
python scripts/qwen36_convert_official_mtp_to_lynn_fused.py --self-test
```

Validates block-scale dequant with synthetic data:
- weight [512,2048] + scale [4,16] → correct block expansion
- weight [384,2048] + scale [3,16] → no broadcast error
- Verifies each 128×128 block gets its own scale value

## Next Steps After Conversion

1. `load_mtp_sidecar` + `mtp_layer_weights` — verify no KeyError
2. `mtp_logits` forward pass — verify output shape [1, 1, vocab_size]
3. Accept probe: greedy decode with/without MTP, measure accept_rate
4. If accept_rate ≥ 0.60 → Phase 1 MTP is production-ready
