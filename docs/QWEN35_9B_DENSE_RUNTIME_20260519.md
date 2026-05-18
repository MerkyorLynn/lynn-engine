# Qwen3.5-9B Dense Runtime Fix — Lynn Engine

**Date:** 2026-05-19
**Goal:** Make Lynn Engine load and serve Qwen3.5-9B (dense, non-MoE) without KeyError/AttributeError crashes.

## Problem

Lynn Engine hardcodes Qwen 3.6-35B-A3B MoE architecture assumptions in 8+ locations:
- `num_experts` accessed as `tc["num_experts"]` (KeyError for dense models)
- `_layer_forward()` unconditionally calls `_moe_forward()` (dense models have no expert FFN)
- Loader assumes `mlp.gate.weight` (MoE naming); dense uses `mlp.gate_proj.weight`
- `inference_state.py` hardcodes `HIDDEN=2048, NUM_KV_HEADS=2` (dense 9B: 3584/4)
- Triton kernels hardcode `NUM_EXPERTS=256, HIDDEN_SIZE=2048`

## Fixes

### `engine/inference_state.py`
- Added `from_config(tc)` factory: reads hidden_size, num_kv_heads, head_dim, num_layers from config dict
- Added `_infer_layer_types(tc)`: if `layer_types` in config, use it; else all `"full_attention"`
- Original module-level constants (HIDDEN=2048 etc.) still present for 3.6 backward compat

### `engine/resident_runner.py`
- `_runtime_config()`: replaced `tc["num_experts"]` → `tc.get("num_experts", 0)`, added `is_moe` flag
- `__init__()`: reads `self.is_moe`, skips `_prepare_triton_moe_layout()`, `_prepare_shared_expert_gate_up_fused()`, `_prepare_packed_nvfp4_moe_layout()`, `_prepare_packed_decode_aliases()` when `is_moe=False`
- Linear attention fused paths guarded by `has_linear_attn` check

### `engine/full_forward.py`
- All 4 config extraction sites: `cfg["num_experts"]` → `cfg.get("num_experts", 0)`, `cfg["is_moe"]` → `cfg.get("is_moe", True)`
- `_layer_forward()`: if `tc.get("is_moe", True)` is False, runs dense FFN (gate_proj/up_proj/down_proj with SiLU)
- `_moe_forward()`: early return `(residual, moe_debug)` if `num_experts == 0`

### `engine/loader.py`
- `load_qwen36_layer()`: auto-detects dense vs MoE via `_detect_expert_ffn_type()`
- Dense path: loads `mlp.gate_proj.weight`, `mlp.up_proj.weight`, `mlp.down_proj.weight`
- MoE path: loads `mlp.gate.weight`, `mlp.experts.{e}.up_proj.weight`, etc.

## Dense 9B Constants

| Parameter | Dense 9B | MoE 3.6 |
|-----------|----------|---------|
| hidden_size | 3584 | 2048 |
| num_kv_heads | 4 | 2 |
| head_dim | 128 | 256 |
| num_layers | 36 | 40 |
| layer_types | all full_attention | 10 linear + 30 full |
| num_experts | 0 (none) | 256 |
| FFN weights | gate_proj/up_proj/down_proj | gate/up (MoE routing) |

## Smoke Test

```bash
bash scripts/r6000_qwen35_9b_dense_runtime_smoke.sh --model-dir /path/to/Qwen3.5-9B
```

Five checks:
1. `_infer_layer_types()` parses config
2. `_runtime_config()` extracts dense dims without KeyError
3. Loader imports without error
4. `_layer_forward()` has dense FFN path
5. `__init__()` has `is_moe` guard

## Status

- **FIX:** Engine code changes (4 files)
- **PENDING:** Real model smoke test (needs Qwen3.5-9B weights)
- **PENDING:** Full runtime validation on R6000

## Notes

- Goal is NOT full Qwen3.5-9B optimization — just structural compatibility
- Dense FFN fallback in `_layer_forward()` uses plain PyTorch (no fused kernels)
- MTP sidecar still requires MoE weights (not guarded for dense — can be added later)
- `native_cuda.py` still compiles MoE CUDA sources unconditionally (cosmetic, not a crash)
