# Qwen3.6 Official MTP → Lynn Fused Converter Plan

**Date:** 2026-05-19
**Status:** CONVERTER READY — needs official `mtp.safetensors` on Spark/R6000 to run
**Model:** Qwen3.6-35B-A3B-FP8 official MTP head (1560 keys, ~815 MiB)

---

## Problem

The official Qwen3.6-35B-A3B-FP8 model ships MTP heads in `mtp.safetensors` with:
- Per-expert layout: `mtp.layers.0.mlp.experts.{N}.gate_proj.weight` (64 experts)
- FP8 E4M3 weights with per-channel `*_scale_inv` tensors
- Pre-FC norms present: `mtp.pre_fc_norm_embedding.weight`, `mtp.pre_fc_norm_hidden.weight`

Lynn's `engine/mtp_sidecar.py` expects a fused layout:
- `mtp.layers.0.mlp.experts.gate_up_proj` — shape `[64, 2816, 2048]` (fused gate+up)
- `mtp.layers.0.mlp.experts.down_proj` — shape `[64, 2048, 1408]` (stacked down)
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
| `experts.{N}.gate_proj.weight` (FP8) + `scale_inv` | `mtp.layers.0.mlp.experts.gate_up_proj` [64, 2816, 2048] BF16 |
| `experts.{N}.up_proj.weight` (FP8) + `scale_inv` | (fused with gate above) |
| `experts.{N}.down_proj.weight` (FP8) + `scale_inv` | `mtp.layers.0.mlp.experts.down_proj` [64, 2048, 1408] BF16 |
| `self_attn.q_proj.weight` (FP8) + `scale_inv` | `mtp.layers.0.self_attn.q_proj.weight` BF16 |
| `shared_expert.gate_proj.weight` (FP8) + `scale_inv` | `mtp.layers.0.mlp.shared_expert.gate_proj.weight` BF16 |
| `mtp.pre_fc_norm_embedding.weight` | `mtp.pre_fc_norm_embedding.weight` (passthrough) |
| `mtp.pre_fc_norm_hidden.weight` | `mtp.pre_fc_norm_hidden.weight` (passthrough) |
| `*.fc.weight` | `mtp.fc.weight` (passthrough/dequant) |
| `*.norm.weight` | `mtp.norm.weight` (passthrough) |

### FP8 Dequantization

Since Lynn runtime currently does NOT consume FP8 `scale_inv` directly:
- All FP8 tensors are dequantized: `bf16_value = fp8_value * scale_inv`
- Output is pure BF16 safetensors
- Metadata records: `source_dtype=fp8_e4m3fn`, `conversion=fp8_to_bf16_fused`

Future: if Lynn adds native FP8 MTP path, produce FP8 fused output instead.

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

## Output Key List (Expected)

```
mtp.fc.weight                                          bfloat16     [2048, 4096]
mtp.layers.0.input_layernorm.weight                    bfloat16     [2048]
mtp.layers.0.mlp.experts.down_proj                     bfloat16     [64, 2048, 1408]
mtp.layers.0.mlp.experts.gate_up_proj                  bfloat16     [64, 2816, 2048]
mtp.layers.0.mlp.gate.weight                           bfloat16     [64, 2048]
mtp.layers.0.mlp.shared_expert.down_proj.weight        bfloat16     [2048, 5632]
mtp.layers.0.mlp.shared_expert.gate_proj.weight        bfloat16     [5632, 2048]
mtp.layers.0.mlp.shared_expert.up_proj.weight          bfloat16     [5632, 2048]
mtp.layers.0.mlp.shared_expert_gate.weight             bfloat16     [1]
mtp.layers.0.post_attention_layernorm.weight           bfloat16     [2048]
mtp.layers.0.self_attn.k_norm.weight                   bfloat16     [128]
mtp.layers.0.self_attn.k_proj.weight                   bfloat16     [512, 2048]
mtp.layers.0.self_attn.o_proj.weight                   bfloat16     [2048, 2048]
mtp.layers.0.self_attn.q_norm.weight                   bfloat16     [128]
mtp.layers.0.self_attn.q_proj.weight                   bfloat16     [2048, 2048]
mtp.layers.0.self_attn.v_proj.weight                   bfloat16     [512, 2048]
mtp.norm.weight                                        bfloat16     [2048]
mtp.pre_fc_norm_embedding.weight                       bfloat16     [2048]
mtp.pre_fc_norm_hidden.weight                          bfloat16     [2048]
```

## Metadata Written

```json
{
  "source_path": "/home/merkyor/models/Qwen3.6-35B-A3B-FP8/mtp.safetensors",
  "source_key_count": "1560",
  "expert_count": "64",
  "hidden_size": "2048",
  "intermediate_size": "1408",
  "conversion_mode": "fp8_to_bf16_fused",
  "fp8_tensors_dequanted": "198",
  "output_key_count": "19",
  "converter": "qwen36_convert_official_mtp_to_lynn_fused.py",
  "sha256_prefix": "a1b2c3d4..."
}
```

## Next Steps After Conversion

1. `load_mtp_sidecar` + `mtp_layer_weights` — verify no KeyError
2. `mtp_logits` forward pass — verify output shape [1, 1, vocab_size]
3. Accept probe: greedy decode with/without MTP, measure accept_rate
4. If accept_rate ≥ 0.60 → Phase 1 MTP is production-ready
