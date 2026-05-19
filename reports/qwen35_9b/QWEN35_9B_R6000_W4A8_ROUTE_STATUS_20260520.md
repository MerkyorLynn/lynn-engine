# Qwen3.5-9B R6000 W4A8 Route Status - 2026-05-20

## Verdict

Qwen3.5-9B W4A8 is quality-safe enough to keep as the NVIDIA speed branch, but the current R6000 result is still a fake-quant/emulation gate rather than native FP8-active serving speed.

The immediate interpretation is:

- W4A8 quality is not the blocker for 9B.
- The next real speed lever is a native FP8-active dense kernel path, not more prompt-level quality triage.
- The compact 6G-style artifact is not ready to ship yet; keep the safe 8.25 GiB NVFP4 artifact until embed/lm_head compaction has its own quality gate.

## R6000 W4A8 Structured Gate

Source artifact:

- `reports/qwen35_9b/remote_r6000_20260520/p196_qwen35_9b_w4a8_structured_content_gate_20260519_1718_p196_chat70.json`

Hard structured/content gate, 70 prompts, max_new 96:

| Variant | Pass | Mean Decode TPS | Notes |
|---|---:|---:|---|
| W4A16 reference | 63/70 | 60.49 | Reference itself is not fully GREEN on this hard set. |
| W4A8 gate/up fake-quant | 64/70 | 60.19 | Quality slightly better than W4A16 reference on this set. |
| W4A8 full fake-quant | 64/70 | 57.81 | Quality remains comparable; speed includes fake-quant overhead. |

This is a quality result, not a native W4A8 speed claim. The gate explicitly notes that fake-quant TPS includes FP8 round-trip emulation overhead.

## Artifact Size Gate

Source artifact:

- `reports/qwen35_9b/remote_r6000_20260520/compact_nvfp4_shrink_gate_20260519_live_compact.json`

Current safe NVFP4 artifact:

- 8.2477 GiB

The shrink gate recommends `SAFE_NO_CHANGE` because compact tiers need separate quality data for embedding and lm_head quantization. Do not shrink the release artifact just to chase Q4_K_M size until that quality gate exists.

## Thinking-On Q4_K_M Reference

Source artifact:

- `reports/qwen35_9b/remote_r6000_20260520/qwen35_9b_q4km_gpqa_thinking32_20260519_182901.summary.json`

Qwen3.5-9B Q4_K_M, thinking-on, 32K budget, GPQA Diamond full 198:

| Metric | Value |
|---|---:|
| Naive accuracy | 143/198 = 72.22% |
| Parse fail | 23/198 = 11.62% |
| Accuracy excluding parse fail | 143/175 = 81.71% |

This remains the strongest Mac/local reference path for first release: small artifact, mature llama.cpp runtime, and a proven thinking-on ceiling.

## Next Work

1. Keep Mac/local first-release on Q4_K_M through llama.cpp.
2. Keep NVIDIA release branch on Lynn-native NVFP4 W4A16/W4A8, with W4A8 now quality-cleared as a speed branch for 9B.
3. Implement or gate a true FP8-active dense kernel path before claiming Spark/R6000 W4A8 runtime gain.
4. Add a separate compact-artifact quality gate before shrinking NVFP4 from 8.25 GiB toward the Q4_K_M size class.
