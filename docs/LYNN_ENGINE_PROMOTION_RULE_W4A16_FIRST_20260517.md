# Promotion Rule: Official 35B W4A16 First

Date: 2026-05-17 23:04 CST

## Rule

Default promotion only considers:

```text
official Qwen/Qwen3.6-35B-A3B
  + Lynn-native W4A16 NVFP4
  + official 35B MTP sidecar
```

W4A8 is a speed experiment, not the default quality route.

## Rationale

Qwen3.6-35B-A3B is already a near-SOTA open model. The remaining product risk is
no longer broad model quality repair; it is whether Lynn-native quantization and
runtime can preserve that quality while cashing out speed.

W4A16 is the stable native counterpart to Q4_K_M:

- W4 weights deliver the size and bandwidth win;
- BF16 activations preserve margin on structured/code/tool-call prompts;
- native Lynn packaging keeps the MTP and runtime optimization path open.

W4A8 should still be measured in the matrix, but only as a later acceleration
branch. If W4A16 lands close to the Q4_K_M/FP8 quality band, do not trade that
stability away for W4A8.

## Tonight's Objective

The R6000 official 35B pipeline should answer:

1. Can official 35B BF16 download and validate cleanly?
2. Can Lynn-native W4A16 pack and load cleanly?
3. Does W4A16 stay close enough to BF16/Q4_K_M on generation gates?
4. Does the official 35B MTP sidecar attach and produce useful P107 credit?

If these are positive, A100 is no longer needed for open-ended 27B quality
recovery. The next workstream becomes R6000 efficiency: native kernels, MTP
runtime integration, and serving overhead.
