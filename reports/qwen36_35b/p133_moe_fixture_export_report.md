# P133 MoE Fixture Export — R6000 Report

Date: 2026-05-18

Machine: RTX PRO 6000 Blackwell

## Purpose

Export real prompt-derived MoE intermediate activations from the official
Qwen3.6-35B-A3B W4A16 NVFP4 model as fixture files for native kernel contract
testing.  The exporter stream-loads layers instead of keeping all dequantized
layers resident, so it can run beside the larger 35B artifacts without becoming
a memory-pressure test.

## Configuration

| Parameter | Value |
|-----------|-------|
| Model | Qwen3.6-35B-A3B-lynn-native-w4a16-nvfp4-v0 |
| Folded sidecar | Qwen3.6-35B-A3B-lynn-native-w4a16-moe-repack-folded-scale-v0 |
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
- `routed_output`: [1, 2048] BF16
- `moe_output`: [1, 2048] BF16

The manifest records the folded sidecar path, layer id, prompt id, selected
experts, routing weights, token position, tensor shapes/dtypes, fixture sha256,
tensor norms, and model metadata.  Optional `--export-intermediates` adds router
logits and slot-level routed FFN debug tensors for native-kernel bring-up.

## Execution

```bash
export REPO_DIR=/root/autodl-tmp/lynn-engine
export MODEL_DIR=/root/autodl-tmp/models/Qwen3.6-35B-A3B-lynn-native-w4a16-nvfp4-v0
export SIDECAR_DIR=/root/autodl-tmp/models/Qwen3.6-35B-A3B-lynn-native-w4a16-moe-repack-folded-scale-v0
bash scripts/r6000_export_qwen36_moe_fixtures.sh
```

## Status

**GREEN** — R6000 execution complete.

| Metric | Result |
|---|---:|
| fixtures | 18 |
| manifest schema | lynn-moe-fixture-v2 |
| layer load time | 14.71 s |
| export wall time | 16.99 s |
| total fixture bytes | 268,061 |
| fixture size | 12.7 KiB each |
| prompt count | 2 |
| layer count | 9 |

## Validation

P134 self-check verifies the fixtures are reproducible:

| Check | Result |
|---|---:|
| self-check verdict | GREEN |
| passed / total | 18 / 18 |
| max abs | 0.0 |
| mean reference latency | 1.022 ms |
| max reference latency | 1.182 ms |
| candidate-output-dir self-check | GREEN, 18/18 |

This confirms the fixtures can serve as the Stream A native kernel admission
gate before a candidate is allowed to spend time on full P37/P25 service gates.

Artifacts:

- `reports/qwen36_35b/p133_fixtures_official_w4a16/manifest.json`
- `reports/qwen36_35b/p133_fixtures_official_w4a16/layer_*.safetensors`
- `reports/qwen36_35b/p134_triton_selfcheck_report.json`
