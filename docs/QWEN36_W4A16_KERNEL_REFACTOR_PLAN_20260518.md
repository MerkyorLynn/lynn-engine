# Qwen3.6 35B W4A16 Kernel Refactor Plan

Date: 2026-05-18

## Current Baseline

Promotion target:

```text
official Qwen/Qwen3.6-35B-A3B
  -> Lynn-native W4A16 NVFP4
  -> quality-safe graph+inplace serving profile
```

Measured state:

| Route | Status |
|---|---:|
| Safe default serving | ~107 decode TPS, structured 14/14 |
| Controlled AMBER structured serving | 114 decode TPS, hard structured 40/40 |
| Q4_K_M llama.cpp reference | ~71 wall TPS warm requests |
| W4A16 quality | MMLU 84.40%, GPQA 49.49% |
| BF16 quality | MMLU 86.40%, GPQA 45.45% |
| Official MTP sidecar | shape/forward ok, local accept 0; no TPS credit |

The current bottleneck is GPU kernel work, not Python HTTP/service overhead:

| Phase | AMBER+RoPE profile |
|---|---:|
| Linear-attention graph blocks | 5.80 ms/token |
| Full-attention layers | 2.72 ms/token |
| Norm + lm_head | 0.36 ms/token |
| Host gap | 0.16 ms/token |
| Wall | 9.07 ms/token |

MoE is uniform across linear layers:

| MoE segment | Mean ms/layer |
|---|---:|
| Router top-k | 0.036 |
| Active packed experts | 0.122 |
| Shared BF16 expert | 0.059 |
| Current full MoE | 0.214 |

Do not spend near-term engineering on a C++ HTTP/service-loop rewrite. The host
gap is too small. The next useful work is CUDA/Triton kernel boundaries and
graph-safe state ownership.

## Hard Constraints

1. Default promotion must preserve full top-8 active MoE plus shared expert.
2. Do not promote expert-dropping: top-k cuts and skip-shared fail hard
   structured parity and only reach ~117 TPS.
3. Do not count MTP until iterative accept and end-to-end TPS pass on the exact
   W4A16 runtime.
4. Atlas is AGPL. Use it only as high-level clean-room reference; do not copy
   source or port kernels line-by-line.
5. Every speed change needs P37 exact-greedy, P25 server, and hard structured
   gate before promotion.

## Parallel Workstreams

### Stream A: Native MoE Kernel Island

Suggested owner: DeepSeek.

Goal: preserve the current Triton numerical contract while removing launches
and intermediate boundaries in the routed/shared MoE path.

Primary files:

- `engine/moe_packed_nvfp4.py`
- `triton_kernels/nvfp4_moe.py`
- `triton_kernels/shared_expert_gate.py`
- `csrc/lynn_native/moe_scalar_kernel.cu`
- `engine/native_cuda.py`
- `benchmarks/p37_moe_config_generate_gate.py`
- `benchmarks/p38_moe_multilayer_profile.py`
- `benchmarks/p39_active_moe_inner_profile.py`

Tasks:

1. Add a strict contract probe for active MoE that reproduces the existing
   Triton path in the same operation order:
   `gate/up -> bf16 inter store -> down -> route weighted sum`.
2. Add a fail-loud backend name, for example
   `LYNN_NATIVE_ACTIVE_MOE_BACKEND=strict_fused_boundary`, and keep it opt-in.
3. First target launch/boundary reduction, not approximate math. The previous
   fast native candidates reached 108-124 TPS but failed greedy parity.
4. Preserve:
   - E2M1 decode table exactly `[0, .5, 1, 1.5, 2, 3, 4, 6]`
   - scale/global-scale order
   - top-k order and routing weights
   - BF16 inter rounding before down unless the probe explicitly proves exact
     greedy parity without it
5. Produce a new report family:
   - `p121_active_moe_strict_boundary_probe.py`
   - `p122_active_moe_strict_boundary_generate_gate.py`

Acceptance:

| Gate | Requirement |
|---|---:|
| Local active-MoE diff | cosine ~1.0, max abs no worse than Triton reference |
| P37 | 3/3 exact greedy |
| Hard structured gate | 40/40 format-clean |
| P25 server | must beat safe default by at least 5%, otherwise keep research-only |
| Failure mode | if first-token/min-prefix regression appears, close the branch |

Expected upside: 5-15% if boundary reduction is real. This alone will not close
155 TPS, but it is the highest-confidence kernel island.

### Stream B: Full-Attention and Linear-Core Fusion

Suggested owner: Claude.

Goal: reduce the 2.7 ms/token full-attention budget and the non-MoE portion of
linear blocks without changing greedy text.

Primary files:

- `engine/resident_runner.py`
- `engine/incremental_decode.py`
- `engine/qwen36_linear_attn_block.py`
- `triton_kernels/full_attn_rope_cache.py` if split out
- `benchmarks/p26_decode_phase_profile.py`
- `benchmarks/p27_full_layer_configd_segment_profile.py`
- `benchmarks/p28_hybrid_block_timing_profile.py`
- `benchmarks/p10c_linear_attn_core_segment_profile.py`

Tasks:

1. Keep `LYNN_FULL_ATTN_ROPE_CACHE=1` as default and refactor the cache into a
   small owned module with explicit prewarm and max-seq handling.
2. Do not retry naive QKV row concat. It was only 1.013x and failed P37 0/3.
3. Build a numerically strict full-attention layer probe that keeps q/k/v
   projection order intact but removes avoidable RoPE/cache/mask allocation.
4. For linear attention, target the boundary around:
   - native FP4 in-proj ~0.077 ms/layer
   - recurrent fused prepare ~0.036 ms/layer
   - conv update ~0.026-0.033 ms/layer
   - gated RMSNorm ~0.020 ms/layer
5. Produce a new report family:
   - `p123_full_attn_strict_cache_probe.py`
   - `p124_linear_core_boundary_probe.py`

Acceptance:

| Gate | Requirement |
|---|---:|
| P37 | 3/3 exact greedy |
| P25 server | positive on 256/512 token runs, not just microbench |
| P26/P28 | phase shift visible, not measurement noise |
| Hard structured gate | 40/40 if candidate is promoted beyond research |

Expected upside: 5-12%. This is the second kernel island after MoE.

### Stream C: Gate Harness and Promotion Discipline

Suggested owner: Codex or Claude sidecar.

Goal: make every agent use the same promotion ladder so good-looking but unsafe
speed branches do not pollute the default route.

Primary files:

- `scripts/qwen36_structured_hard_prompts.json`
- `scripts/openai_structured_tps_gate.py`
- `benchmarks/p25_server_decode_tps_probe.py`
- `benchmarks/p37_moe_config_generate_gate.py`
- `reports/ops/QWEN36_35B_W4A16_OVERNIGHT_STATUS_20260518.md`

Tasks:

1. Add a single wrapper script for candidate promotion:
   `scripts/r6000_qwen36_candidate_promotion_gate.sh`.
2. The wrapper should run:
   - P37 exact-greedy
   - P25 server 128/256/512
   - hard structured OpenAI gate
   - P26/P28 if speed is positive
3. Emit one summary JSON with:
   - candidate env
   - exact parity
   - format pass rate
   - P25 512 decode TPS
   - promote_default boolean
   - promote_amber boolean
4. Add clear thresholds:
   - default: P37 exact + hard structured clean + P25 >= safe default + 1%
   - AMBER: hard structured clean + P25 >= safe default + 5%, with documented
     exact-greedy drift
   - closed: first-token/min-prefix regression or P25 negative

Acceptance:

The wrapper can reproduce the current safe default and AMBER results using only
env vars and checked-in prompt files.

Expected upside: engineering velocity and fewer false positives.

### Stream D: MTP Reframing

Suggested owner: DeepSeek or paused until kernel streams move.

Goal: keep MTP as a measured branch, not a thesis.

Primary files:

- `benchmarks/p120_mtp_alignment_sweep.py`
- `scripts/a100_mtp_iterative_train.py`
- `scripts/a100_mtp_saved_sidecar_diagnostic.py`
- `reports/mtp/*`

Tasks:

1. Freeze official 35B MTP sidecar as warm-start/diagnostic only.
2. Do not use A100 for open-ended MTP until a concrete Qwen3.6 hybrid-SSM
   accept-rate plan exists.
3. If restarted, target iterative accept on W4A16 first, not BF16-only top-k.

Acceptance:

MTP can only count if local accept is non-zero, stable on structured prompts,
and P25 end-to-end TPS improves.

## Recommended Assignment

DeepSeek:

1. Stream A native MoE kernel island.
2. Only touch `csrc/lynn_native/*`, `engine/native_cuda.py`, and the opt-in
   active-MoE backend branch in `engine/moe_packed_nvfp4.py`.
3. Do not edit server or full-attention files.

Claude:

1. Stream B full-attention/linear-core fusion plus Stream C promotion wrapper.
2. Own `resident_runner.py`, attention/linear-core probes, and gate scripts.
3. Do not edit native CUDA MoE files.

Codex:

1. Integrate results, run R6000 gates, update reports, and push only candidates
   that pass the promotion ladder.
2. Keep default safe route stable while AMBER candidates stay opt-in.

## Near-Term Target

Next practical target is not 155 in one jump. It is:

```text
safe default: 107 -> 118-125 decode TPS
AMBER structured: 114 -> 125-130 decode TPS
```

After that, re-evaluate whether a trained MTP path or a larger native-kernel
rewrite is needed for 155.
