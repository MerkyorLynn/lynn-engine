# P199 · Qwen3.5-9B NVFP4 Artifact Size Audit

**Date:** 2026-05-19
**Author:** Qwen Code (auto-generated)
**Status:** 🟢 R6000 executed — artifact size source identified

---

## Problem

Lynn-native NVFP4 W4A16 9B model is ~8.3 GiB, vs Q4_K_M 5.3 GiB (+3 GiB).
The extra size comes from kept BF16 tensors (`embed_tokens`, `lm_head`) and
quantization metadata (scales, global_scales). Need to understand exactly where
the bytes go and what shrink options exist without breaking W4A16 safe quality.

## Method

1. Read `lynn_quant_manifest.json` (if present) for `keep_regex` and tensor inventory
2. Scan all `.safetensors` headers to get per-tensor shape, dtype, byte count
3. Classify each tensor into categories:
   - `quantized_packed` — uint8 packed FP4 weights (`.packed` / `.weight_packed`)
   - `quantized_scale` — per-group scales (`.scale` / `.weight_scale`)
   - `quantized_global_scale` — scalar global scales
   - `kept_bf16_<category>` — BF16 tensors matching `keep_regex`
   - `non_tensor_metadata` — tokenizer, config, manifest, README
4. Break kept BF16 tensors into: `embed_tokens`, `lm_head`, `norms`, `rope`, `mlp_gate`, `visual`, `mtp`, `other`
5. Show top 30 largest kept BF16 tensors
6. Propose three shrink tiers with expected sizes and quality risk

## JSON Schema

```json
{
  "schema": "lynn-nvfp4-size-audit-v1",
  "model_dir": "/path/to/model",
  "total_gib": 8.3,
  "total_bytes": 8912345678,
  "n_files": 5,
  "n_tensors": 500,
  "q4km_reference_gib": 5.3,
  "delta_gib": 3.0,
  "manifest_present": true,
  "manifest_schema": "lynn-variable-nvfp4-pack-v1",
  "keep_regex": "(embed_tokens|lm_head|visual|rotary|norm|mlp\\.gate\\.weight)",
  "category_breakdown": {
    "quantized_packed": {"gib": 4.0, "count": 300},
    "quantized_scale": {"gib": 0.3, "count": 300},
    "quantized_global_scale": {"gib": 0.001, "count": 300},
    "quantized_total": {"gib": 4.3},
    "kept_bf16_total": {"gib": 3.8},
    "non_tensor_metadata": {"gib": 0.05}
  },
  "kept_bf16_breakdown": {
    "embed_tokens": {"gib": 1.9, "count": 1},
    "lm_head": {"gib": 1.9, "count": 1},
    "norms": {"gib": 0.003, "count": 64},
    "rope": {"gib": 0.001, "count": 1},
    "mlp_gate": {"gib": 0.005, "count": 32},
    "visual": {"gib": 0.0, "count": 0},
    "mtp": {"gib": 0.0, "count": 0},
    "other": {"gib": 0.0, "count": 0}
  },
  "kept_bf16_top30": [
    {"key": "model.language_model.lm_head.weight", "category": "lm_head", "shape": [151936, 4096], "dtype": "BF16", "nbytes": 1244397568, "gib": 1.159}
  ],
  "shrink_options": [
    {
      "tier": "SAFE_NO_CHANGE",
      "description": "Ship current artifact as-is.",
      "expected_size_gib": 8.3,
      "quality_risk": "none",
      "changes": []
    },
    {
      "tier": "MODERATE_QUANTIZE_EMBED_LMHEAD",
      "description": "...",
      "expected_size_gib": 5.5,
      "quality_risk": "high",
      "changes": ["quantize embed_tokens", "quantize lm_head"]
    },
    {
      "tier": "AGGRESSIVE_QUANTIZE_MORE_KEEPERS",
      "description": "...",
      "expected_size_gib": 4.5,
      "quality_risk": "very high",
      "changes": ["+ prune visual/MTP"]
    }
  ]
}
```

## Expected Findings

Based on prior analysis (QWEN35_9B_NVFP4_SIZE_SHRINK_PLAN_20260519.md):

| Bucket | Expected Size |
|--------|--------------|
| Quantized dense MLP | ~2.8 GiB |
| Quantized linear attention | ~0.9 GiB |
| Quantized full attention | ~0.3 GiB |
| Quantized visual | ~0.3 GiB |
| Quantized MTP | ~0.1 GiB |
| `lm_head.weight` BF16 | ~1.9 GiB |
| `embed_tokens.weight` BF16 | ~1.9 GiB |
| norms / small | ~0.003 GiB |
| **Total** | **~8.3 GiB** |

Key constraint: `lm_head` and `embed_tokens` are NOT tied (cosine=0.0198).
No free alias/dedup trick.

## How to Run

```bash
cd /root/autodl-tmp/lynn-engine
bash scripts/r6000_qwen35_9b_nvfp4_size_audit.sh

# Custom model dir:
MODEL=/path/to/other/nvfp4 bash scripts/r6000_qwen35_9b_nvfp4_size_audit.sh
```

If the model directory doesn't exist on the current host, the script writes
a PENDING status JSON without failing.

## R6000 Result

| Metric | Value |
|--------|-------|
| report JSON | `reports/qwen35_9b/p199_nvfp4_size_audit_20260519_live_size2.json` |
| total_gib | 8.248 GiB |
| delta_gib vs Q4_K_M | +2.948 GiB |
| quantized_total | 4.434 GiB |
| kept_bf16_total | 3.792 GiB |
| non_tensor_metadata | 0.022 GiB |
| embed_tokens | 1.895 GiB |
| lm_head | 1.895 GiB |
| kept other | 0.002 GiB |

Conclusion: the 8.3G package is not bloated by random metadata. The entire
gap versus Q4_K_M is almost exactly the two BF16 matrices `embed_tokens` and
`lm_head` plus normal NVFP4 scale overhead. A 6G-class NVFP4 package is
possible only by quantizing one or both of those matrices, which is not a
packaging-only change and needs a focused MMLU/GPQA gate.

Current recommendation: keep the 8.25G W4A16 artifact as the stable NVIDIA
release candidate. Treat a compact 5.3-5.6G variant as a new experimental
artifact, not a transparent repack.

---

## Relationship to Other Work

| Item | Relation |
|------|----------|
| P190 | P199 size audit informs P190's artifact packaging decisions |
| P197 | Independent — P197 is drift probe, P199 is size audit |
| P198 | Independent — P198 is FP4 MMA preflight, P199 is artifact size |
| keep_regex | P199 reports current keep_regex from manifest |
| Shrink plan | P199 produces shrink_options; execution is a separate task |
