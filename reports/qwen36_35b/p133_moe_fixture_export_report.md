# P133 MoE Fixture Export — R6000 Report

Date: 2026-05-18 (pending R6000 execution)

Machine: RTX PRO 6000 Blackwell

## Purpose

Export real MoE intermediate activations from the full Qwen3.6-35B W4A16
(NVFP4 v8-RTN) model as fixture files for native kernel contract testing.

## Configuration

| Parameter | Value |
|-----------|-------|
| Model | Lynn-V4-Pro-Distill-Qwen-35B-A3B-NVFP4-v8-RTN |
| Layers | 0, 4, 8, 16, 20, 28, 32, 36, 39 (9 layers) |
| Prompts | "Hello", "The capital of France is" (2 prompts) |
| Expected fixtures | 18 (9 layers × 2 prompts) |
| Hidden size | 2048 |
| Top-K | 8 |
| Num experts | 256 |

## Expected Output

```
reports/qwen36_35b/p133_fixtures/
├── manifest.json
├── layer_00_prompt_00.safetensors
├── layer_00_prompt_01.safetensors
├── layer_04_prompt_00.safetensors
├── layer_04_prompt_01.safetensors
├── layer_08_prompt_00.safetensors
├── layer_08_prompt_01.safetensors
├── layer_16_prompt_00.safetensors
├── layer_16_prompt_01.safetensors
├── layer_20_prompt_00.safetensors
├── layer_20_prompt_01.safetensors
├── layer_28_prompt_00.safetensors
├── layer_28_prompt_01.safetensors
├── layer_32_prompt_00.safetensors
├── layer_32_prompt_01.safetensors
├── layer_36_prompt_00.safetensors
├── layer_36_prompt_01.safetensors
├── layer_39_prompt_00.safetensors
└── layer_39_prompt_01.safetensors
```

Each fixture safetensors file contains:
- `hidden_in`: [1, 2048] BF16
- `expert_ids`: [8] int32
- `routing_weights`: [8] float32
- `moe_output`: [1, 2048] BF16

Total size: ~18 × 16 KB ≈ 300 KB

## Execution

```bash
export MODEL_DIR=/root/autodl-tmp/models/Lynn-V4-Pro-Distill-Qwen-35B-A3B-NVFP4-v8-RTN
export REPO_DIR=/root/autodl-tmp/lynn-engine
bash scripts/r6000_export_qwen36_moe_fixtures.sh
```

## Status

**PENDING** — Awaiting R6000 execution.

Expected timing:
- Layer loading (40 layers): ~3-5 min
- Fixture export: ~30s
- Total: ~5 min

## Validation

After export, run p134 self-check to verify fixtures are reproducible:
- All fixtures must produce `max_abs=0` when recomputed from the same weights
- This confirms the fixtures can serve as Stream A native kernel admission gate
