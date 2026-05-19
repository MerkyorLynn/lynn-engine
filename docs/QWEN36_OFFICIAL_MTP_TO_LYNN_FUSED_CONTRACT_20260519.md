# Qwen3.6 Official MTP → Lynn Fused Contract

**Date:** 2026-05-19
**Priority:** HIGH — MTP speedup is gated on this conversion
**Status:** CONVERTER READY

---

## Purpose

Enable Lynn Engine to use the official Qwen3.6 NextN MTP head for speculative
decoding without retraining. The converter bridges the official FP8 per-expert
layout to Lynn's fused BF16 layout that `engine/mtp_sidecar.py` expects.

## Architecture

```
Official mtp.safetensors (FP8, per-expert)
    │
    ├── 64 × experts.{N}.gate_proj.weight (FP8 + scale_inv)
    ├── 64 × experts.{N}.up_proj.weight (FP8 + scale_inv)
    ├── 64 × experts.{N}.down_proj.weight (FP8 + scale_inv)
    ├── self_attn.{q,k,v,o}_proj.weight (FP8 + scale_inv)
    ├── shared_expert.{gate,up,down}_proj.weight (FP8 + scale_inv)
    ├── mtp.pre_fc_norm_embedding.weight (BF16)
    ├── mtp.pre_fc_norm_hidden.weight (BF16)
    ├── *.fc.weight (BF16 or FP8)
    └── *.norm.weight (BF16)
           │
           ▼  qwen36_convert_official_mtp_to_lynn_fused.py
           │
    Lynn mtp_lynn_fused.safetensors (BF16, fused)
    │
    ├── mtp.fc.weight [2048, 4096]
    ├── mtp.pre_fc_norm_embedding.weight [2048]
    ├── mtp.pre_fc_norm_hidden.weight [2048]
    ├── mtp.norm.weight [2048]
    ├── mtp.layers.0.mlp.experts.gate_up_proj [64, 2816, 2048]
    ├── mtp.layers.0.mlp.experts.down_proj [64, 2048, 1408]
    ├── mtp.layers.0.mlp.gate.weight [64, 2048]
    ├── mtp.layers.0.mlp.shared_expert.{gate,up,down}_proj.weight
    ├── mtp.layers.0.mlp.shared_expert_gate.weight
    ├── mtp.layers.0.self_attn.{q,k,v,o}_proj.weight
    ├── mtp.layers.0.self_attn.{q,k}_norm.weight
    └── mtp.layers.0.{input,post_attention}_layernorm.weight
```

## Conversion Rules

| Rule | Description |
|------|-------------|
| FP8 block-scale → BF16 | `scale_expanded = scale_inv.repeat_interleave(128, dim=0/1)`, then `bf16 = fp8.float() * scale_expanded` |
| gate + up fuse | `cat([gate_proj, up_proj], dim=0)` per expert, then `stack` across 256 experts |
| down stack | `stack([down_proj_0, ..., down_proj_255], dim=0)` |
| Passthrough | Norms, fc, gate remain as-is (cast to BF16) |
| Metadata | Records source_dtype, block_size=(128,128), conversion_mode, sha256_prefix |
| Expert validation | Assert IDs contiguous 0..E-1; BLOCKER if not |
| Dim inference | hidden/intermediate from actual tensor shapes, NOT config defaults |

## Verification Contract

After conversion, `load_mtp_sidecar` must succeed AND `mtp_layer_weights` must
return all 16 required keys without error:

```python
required_keys = {
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
    "self_attn.q_norm.weight",
    "self_attn.k_norm.weight",
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "mlp.gate.weight",
    "mlp.experts.gate_up_proj",
    "mlp.experts.down_proj",
    "mlp.shared_expert.gate_proj.weight",
    "mlp.shared_expert.up_proj.weight",
    "mlp.shared_expert.down_proj.weight",
    "mlp.shared_expert_gate.weight",
}
```

Plus top-level: `mtp.fc.weight`, `mtp.pre_fc_norm_embedding.weight`,
`mtp.pre_fc_norm_hidden.weight`, `mtp.norm.weight`.

## Performance Budget

| Operation | Time Budget | Notes |
|-----------|-------------|-------|
| Conversion (CPU) | < 60s | One-time, offline |
| Load fused sidecar | < 5s | Startup cost only |
| MTP forward (1 token) | < 2ms | Must not exceed main model decode time |
| Accept probe | N/A | Depends on model quality |

## What This Contract Does NOT Cover

1. FP8 native runtime (future: skip dequant, use FP8 directly)
2. Retraining or fine-tuning the MTP head
3. Multiple offset heads (offset>1 is Phase 2)
4. 9B dense MTP (9B may use different layout)
5. GGUF/llama.cpp MTP integration

## Files

| Path | Purpose |
|------|---------|
| `scripts/qwen36_official_mtp_inventory.py` | Scan official mtp.safetensors structure |
| `scripts/qwen36_convert_official_mtp_to_lynn_fused.py` | FP8 per-expert → BF16 fused conversion |
| `reports/mtp/QWEN36_OFFICIAL_MTP_CONVERTER_PLAN_20260519.md` | Execution plan |
| `docs/QWEN36_OFFICIAL_MTP_TO_LYNN_FUSED_CONTRACT_20260519.md` | This contract |
| `engine/mtp_sidecar.py` | Runtime loader (existing, not modified) |
