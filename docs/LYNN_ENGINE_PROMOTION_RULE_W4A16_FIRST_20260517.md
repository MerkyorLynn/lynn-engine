# Promotion Rule: Official 35B W4A16 First

Date: 2026-05-17 23:04 CST

## Rule

Default promotion only considers:

```text
official Qwen/Qwen3.6-35B-A3B
  + Lynn-native W4A16 NVFP4
  + quality-safe runtime profile
```

MTP is a gated acceleration add-on, not assumed default credit. A sidecar may be
attached only after it clears an iterative accept gate on the exact W4A16 runtime
profile. W4A8 is a speed experiment, not the default quality route.

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

The official Qwen3.6 MTP sidecar is useful as a compatibility probe, but it must
earn runtime credit empirically. On 2026-05-18 the official sidecar passed shape
and forward smoke on the Lynn-native W4A16 package, then failed iterative accept
at 0/24. Until that changes, 155 TPS planning should not count MTP as a free
multiplier.

External Atlas numbers should be read with the same caution. The pinned 131 TPS
benchmark is for Qwen3.5-35B-A3B MTP on Spark, while Qwen3.6-35B-A3B is a hybrid
SSM target and Atlas documentation warns that speculative decoding can be slower
on hybrid SSM models. Treat any Qwen3.6 MTP claim as a hypothesis until the local
accept-rate and end-to-end TPS gate proves it.

Spark quality on the official 35B package now supports the W4A16-first rule:
BF16 scored 86.40% MMLU / 45.45% GPQA, while Lynn-native W4A16 NVFP4 scored
84.40% MMLU / 49.49% GPQA. The MMLU delta is about -2pp and GPQA is within the
expected sample-noise band, so the next primary risk is runtime speed, not broad
quality rescue.

R6000 graph+in-place serving is the current speed baseline candidate. On
2026-05-18 it held 81-82 decode TPS through 128/256/512-token P25 server probes,
with 72.76 wall TPS at 512 tokens, and passed a 14-request OpenAI structured
gate covering JSON, tool-call JSON, Python, YAML, Chinese constraints, and a
numeric answer.

The 155 TPS gap is now a GPU-kernel problem first. P26 profiling measured about
8.68 ms/token in linear-attention graph blocks, 4.14 ms/token in full-attention
layers, 0.33 ms/token in norm + native FP4 lm_head, and only 0.14 ms/token host
gap. Prioritize linear-block replay/fusion and full-attention layer fusion before
large service-loop rewrites.

Full-attention graph slots are not yet a reusable cross-request solution. P9-V
passed strict parity when captured on the same prompt state, but P9-W failed
cross-prompt reuse. Keep the promoted serving profile on reusable linear-block
graphs plus eager full-attention until full-attention slots are state-safe across
requests or cheap enough to capture per request.

## Tonight's Objective

The R6000 official 35B pipeline should answer:

1. Can official 35B BF16 download and validate cleanly?
2. Can Lynn-native W4A16 pack and load cleanly?
3. Does W4A16 stay close enough to BF16/Q4_K_M on generation gates?
4. Does the official 35B MTP sidecar attach and produce useful accept credit?

If these are positive, A100 is no longer needed for open-ended 27B quality
recovery. The next workstream becomes R6000 efficiency: native kernels, MTP
runtime integration after accept is real, and serving overhead.
