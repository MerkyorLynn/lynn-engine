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

The key distinction from W4A8 is activation precision. Q4_K_M stays stable
because it is a weight-only 4-bit route: activations remain FP16/BF16. Lynn
W4A16 should follow that contract. Its CUDA path is not "FP8 activation x FP4
weight"; it is fused 4-bit weight load/dequant plus BF16/FP16 GEMV/GEMM or
grouped-MoE kernels. W4A8 has a real R6000 tensor-core atom path, but it changes
the activation contract and remains a separate acceleration experiment until
structured/code/tool-call quality proves it safe.

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

P28/P38/P39 sharpen the next target: the 10 linear graph blocks are uniform at
about 0.866 ms each, and sampled packed MoE costs about 0.205 ms per linear
layer. Across 30 linear layers this is roughly 6.15 ms/token, so MoE fusion and
shared-expert/native active-expert kernel work is the first kernel island to
attack before expecting 155 TPS from W4A16.

MoE optimization alone is not enough. On 2026-05-18 upper-bound ablations showed
skip-shared at 82.67 TPS, skip-active at 91.65 TPS, and skip-all-MoE at
102.49 TPS. This means 155 TPS requires combined gains: MoE fusion, attention
fusion, and/or a locally accepted speculation path. Also keep
`LYNN_PACKED_SHARED_EXPERT=0`: on 35B, packed shared scalar/native paths are
slower than BF16 shared expert and native fast 2D has worse local cosine.

The first Qwen3.6-specific speed win is now validated. `triton_fast_decode`
keeps the W4A16/BF16-activation contract and only simplifies the E2M1 decode
expression inside the existing Triton gate/up shape. On 2026-05-18 it passed the
P37 greedy parity gate on official 35B W4A16 with 3/3 exact prompts and improved
median runner TPS from 100.43 to 102.57. The service gate then passed 14/14
structured OpenAI requests with mean decode TPS 87.71, min 86.99, and P25
512-token wall/decode TPS of 76.58 / 86.60. This is now part of the promoted
fast W4A16 profile.

The second safe W4A16 speed win is `LYNN_QK_NORM_ROPE_BACKEND=triton_pair`.
It also keeps BF16 activations and only fuses the full-attention Q/K RMSNorm and
RoPE pair. P27 segment profiling dropped `attn.qk_norm_rope` from 0.361 to
0.129 ms on layer 3 and from 0.353 to 0.136 ms on layer 31. End-to-end P26
decode improved from 85.48 to 93.07 TPS, with full-attention layers dropping
from 4.08 to 3.11 ms/token. The OpenAI service gate then passed 14/14 structured
requests with mean decode TPS 96.43, min 96.03, and P25 512-token wall/decode
TPS of 83.38 / 95.88. This backend is now part of the promoted fast W4A16
profile.

Several tempting small fusions are closed for now. A Triton router top-k softmax
kernel is faster only after logits are already computed, but the full router path
regresses from 0.046 ms to 0.052 ms on sampled layers. A `tl.dot` gate/up rewrite
also regresses badly on official 35B layer 28: 0.080 ms versus the current 0.033
ms reference. Tile-only MoE sweeps did not beat the default gate/down shape.
The next real kernel island remains Lynn-owned W4A16 grouped active/shared MoE
and attention fusion, not more router micro-kernels.

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
