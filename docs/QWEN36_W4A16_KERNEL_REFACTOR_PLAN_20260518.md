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
| Q4_K_M llama.cpp reference | R6000 CUDA: 207 wall TPS single 512, 501 total TPS concurrent 8 |
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

## llama.cpp Reference Delta

The clean R6000 CUDA baseline for official Qwen3.6-35B-A3B Q4_K_M-imatrix is
now much stronger than the earlier Spark-only reference:

| llama.cpp Q4_K_M R6000 probe | Result |
|---|---:|
| Single 128 tokens | 154.39 wall TPS |
| Single 256 tokens | 202.47 wall TPS |
| Single 512 tokens | 207.29 wall TPS |
| Concurrent 2 total | 306.34 wall TPS |
| Concurrent 4 total | 400.39 wall TPS |
| Concurrent 8 total | 500.66 wall TPS |
| 5.8k prompt tokens + 128 decode | 91.13 wall TPS including prefill |
| 11.6k prompt tokens + 128 decode | 84.19 wall TPS including prefill |

This changes the competitive target. Q4_K_M is W4A16-class, not W4A8: it uses
4-bit weights with high-precision activations and accumulation. The useful
lesson is not "sacrifice quality for W4A8"; it is that a stable W4A16-class
runtime can be very fast if the weight layout, graph, and CUDA boundaries are
engineered correctly.

The llama.cpp CUDA files worth studying as MIT-licensed high-level references
are:

- `ggml/src/ggml-cuda/mmq.cu`, `mmq.cuh`, and `mmvq.cu` for Q4_K_M repacked
  matmul/matvec dispatch
- `ggml/src/ggml-cuda/topk-moe.cu` plus the graph-pattern detector in
  `ggml-cuda.cu` for softmax/top-k/get_rows fusion
- `ggml/src/ggml-cuda/gated_delta_net.cu` and `ssm-scan.cu` for GDN/SSM
  single-boundary kernels

The clean-room Lynn translation should be:

1. Add a Lynn-native offline repack for W4A16/NVFP4 decode-friendly expert
   tiles, separate from the safetensors manifest format.
2. Fuse top-k routing and row gather/scatter boundaries only after exact
   first-token probes pass.
3. Fuse GDN/SSM boundaries in Stream B before touching the HTTP service loop.
4. Keep every candidate behind the Stream C promotion ladder.

### Repack Inventory Result

`scripts/qwen36_w4a16_repack_inventory.py` is the read-only inventory pass for
the serving-layout route. It scans only `lynn_quant_manifest.json` plus
`model.safetensors.index.json` and does not change artifacts.

R6000 inventory for the current official 35B W4A16 NVFP4 artifact:

| Bucket | Records | Packed GiB | Shards | Meaning |
|---|---:|---:|---:|---|
| active MoE | 80 | 15.0000 | 7 | first offline repack target |
| shared MoE | 160 | 0.0586 | 7 | fuse with active MoE boundary after parity |
| linear attention | 150 | 0.4706 | 7 | boundary/layout target for 30 SSM layers |
| full attention | 40 | 0.1270 | 7 | cache/boundary target; weight repack is smaller ROI |
| MTP | 11 | 0.3931 | 1 | warm-start/diagnostic only; not counted in serving target |
| visual | 112 | 0.2078 | 1 | not on the text serving critical path |

The language stack has 40 layers: 30 linear-attention layers and 10
full-attention layers (`3, 7, 11, 15, 19, 23, 27, 31, 35, 39`). This makes
the next ordering concrete:

1. active-MoE gate/up + down decode-tile repack;
2. shared-expert and shared-gate co-location;
3. linear-attention projection pack and boundary fusion;
4. full-attention RoPE/cache/workspace cleanup.

Artifacts:

- `reports/qwen36_35b/qwen36_w4a16_repack_inventory_20260518.json`
- `reports/qwen36_35b/QWEN36_W4A16_REPACK_INVENTORY_20260518.md`

### MoE Repack V0 Result

The first MoE-only serving sidecar is complete on R6000:

| Item | Result |
|---|---:|
| Sidecar path | `/root/autodl-tmp/models/Qwen3.6-35B-A3B-lynn-native-w4a16-moe-repack-v0` |
| Layers | 40 |
| Size | 18.8624 GiB |
| Build time | 25.593 s |
| P127 all-layer contract | GREEN, 40/40 |

Each layer file co-locates router, active gate/up, active down, shared expert,
and shared-gate tensors. Active expert tensors are now stored in expert-major
3D layout:

- `active_gate_up.packed`: `[256, 1024, 1024]`
- `active_down.packed`: `[256, 2048, 256]`
- `router.weight`: `[256, 2048]`

This does not change math. P127 verifies the sidecar against the current
manifest path for all 40 layers. The next MoE kernel-boundary implementation
should consume this sidecar directly instead of chasing generic safetensors
manifest keys.

Artifacts:

- `scripts/qwen36_w4a16_moe_repack_sidecar.py`
- `engine/moe_repack_sidecar.py`
- `benchmarks/p127_moe_repack_sidecar_contract.py`
- `benchmarks/p128_moe_repack_triton_boundary_probe.py`
- `reports/qwen36_35b/qwen36_w4a16_moe_repack_manifest_20260518.json`
- `reports/qwen36_35b/p127_moe_repack_sidecar_contract_all40_20260518.json`
- `reports/qwen36_35b/p128_moe_repack_triton_boundary_all40_20260518.json`
- `reports/qwen36_35b/QWEN36_W4A16_MOE_REPACK_V0_20260518.md`

P128 then feeds the current Triton active-MoE boundary directly from the sidecar
for all 40 layers. It is GREEN with `max_abs=0.0`; mean active-MoE time is
`0.08257 ms` from sidecar tensors versus `0.08362 ms` from manifest-loaded
tensors. This confirms the sidecar is now a valid kernel-input ABI. It is not
the fused native boundary yet; it removes layout uncertainty before that work.

### Runtime Sidecar and Scratch Boundary

`LYNN_MOE_REPACK_SIDECAR_DIR` wires the sidecar into the resident runner without
changing the default path. R6000 gate `moe_repack_sidecar` attached all 40
layers from the sidecar and passed P37 exact plus 40/40 hard structured, but it
stayed at default-class speed:

| Candidate | P37 | P25 512 decode TPS | Structured | Decision |
|---|---:|---:|---:|---|
| sidecar runtime | exact | 107.39 | 40/40 | research-only, below margin |
| sidecar + active scratch | exact | 107.08 | 40/40 | research-only, below margin |

`LYNN_MOE_ACTIVE_SCRATCH=1` adds per-layer active-MoE intermediate/output
scratch tensors. This makes the runtime boundary fixed and strict, but the
profile is flat: allocation is not the 155 TPS blocker. The next MoE step must
replace the gate/up and down inner math with a real grouped native-FP4 kernel;
more Python/env switch hunting around this boundary is low ROI.

Artifacts:

- `scripts/qwen36_candidate_env_moe_repack_sidecar.env`
- `scripts/qwen36_candidate_env_moe_repack_scratch.env`
- `reports/qwen36_35b/QWEN36_W4A16_MOE_REPACK_RUNTIME_P129_20260518.md`

### Effective-Scale Repack Probe

P130 adds the first memory-neutral scale repack candidate:
`LYNN_MOE_EFFECTIVE_SCALE=1`.  At load time the active-MoE scale aliases are
replaced with `scale / global_scale`, and the global scale aliases become one.
This avoids doubling resident scale memory; the first attempted "attach another
copy" version OOMed the 35B runner on R6000.

Layer-local active-MoE probe:

| Metric | Result |
|---|---:|
| exact output vs current active-MoE | 9/9 |
| max abs / max rel L2 | 0.0 / 0.0 |
| mean active boundary | 0.05594 ms -> 0.05147 ms |
| active boundary speedup | 1.087x |

Full R6000 gate stayed quality-safe but below promotion margin:

| Gate | Result |
|---|---:|
| P37 exact | true |
| P25 512 decode TPS | 107.98 |
| hard structured | 40/40, mean 108.40 decode TPS |
| decision | research-only |

Decision: keep P130 as an opt-in native-MoE building block.  It is a strict
repack win, not the 122/155 TPS breakthrough; the next step remains true
grouped native-FP4 gate/up/down math behind the same active-MoE boundary.

Artifacts:

- `benchmarks/p130_moe_effective_scale_probe.py`
- `scripts/qwen36_candidate_env_moe_effective_scale.env`
- `reports/qwen36_35b/QWEN36_W4A16_MOE_EFFECTIVE_SCALE_P130_20260518.md`

### Repack Stack Closure

P131 combines the strict sidecar, active scratch, and effective-scale paths:

```text
LYNN_MOE_REPACK_SIDECAR_DIR=/root/autodl-tmp/models/Qwen3.6-35B-A3B-lynn-native-w4a16-moe-repack-v0
LYNN_MOE_ACTIVE_SCRATCH=1
LYNN_MOE_EFFECTIVE_SCALE=1
```

R6000 result:

| Gate | Result |
|---|---:|
| P37 exact | true |
| P37 median speedup | 0.998x |
| P25 512 decode TPS | 107.95 |
| hard structured | 40/40, mean 107.72 decode TPS |
| decision | research-only |

This closes the non-math MoE repack stack: sidecar lookup, scratch allocation,
and global-scale division are not the remaining 122/155 TPS blockers.  The next
MoE step must be a true grouped native gate/up and down kernel behind the same
strict active-MoE boundary; more env stacking around the current Triton inner
math is low ROI.

Artifacts:

- `scripts/qwen36_candidate_env_moe_repack_scratch_effective.env`
- `reports/qwen36_35b/QWEN36_W4A16_MOE_REPACK_SCRATCH_EFFECTIVE_P131_20260518.md`

### Folded-Scale Sidecar Input Contract

P132 adds an offline sidecar option for native MoE kernels:

```text
scripts/qwen36_w4a16_moe_repack_sidecar.py --fold-active-global-scale
```

For active experts only, this stores `scale / global_scale` directly and writes
`global_scale=1`.  Shared expert tensors keep the original scale contract.  The
purpose is to give the native grouped kernel a direct effective-scale input
without runner-time replacement.

Layer-0 R6000 validation:

| Check | Result |
|---|---:|
| folded sidecar build | GREEN |
| P127 contract | GREEN |
| P128 Triton boundary | GREEN |
| max abs / mean abs | 0.0 / 0.0 |
| folded effective aliases | gate/up true, down true |
| folded sidecar active-MoE timing | 0.08210 ms |
| manifest / sidecar timing ratio | 1.062x |

Full all-layer R6000 validation:

| Check | Result |
|---|---:|
| sidecar path | `/root/autodl-tmp/models/Qwen3.6-35B-A3B-lynn-native-w4a16-moe-repack-folded-scale-v0` |
| sidecar size | 23 GiB on disk |
| files | 40 layer files + manifest |
| P127 contract | GREEN, 40/40 |
| P128 Triton boundary | GREEN, 40/40 |
| max abs / mean abs | 0.0 / 0.0 |
| manifest active-MoE mean | 0.08317 ms/layer |
| folded sidecar active-MoE mean | 0.08244 ms/layer |
| manifest / sidecar mean ratio | 1.009x |

Decision: keep this as the next native-kernel input format.  The full 40-layer
folded sidecar is now built and validated, so Stream A can consume one stable
ABI instead of re-opening manifest/global-scale layout questions.  The sidecar
loader exposes folded active scales as effective-scale aliases, and the
resident runner skips runner-time replacement when those aliases are already
present.  This is not expected to move TPS alone; it is the repack foundation
for replacing the inner active-MoE math.

Artifacts:

- `reports/qwen36_35b/QWEN36_W4A16_MOE_FOLDED_SCALE_SIDECAR_P132_20260518.md`
- `reports/qwen36_35b/p132_moe_folded_scale_sidecar_contract_layer0_20260518.json`
- `reports/qwen36_35b/p132_moe_folded_scale_sidecar_triton_boundary_layer0_20260518.json`
- `reports/qwen36_35b/p132_moe_folded_scale_sidecar_contract_all40_20260518.json`
- `reports/qwen36_35b/p132_moe_folded_scale_sidecar_triton_boundary_all40_20260518.json`
- `scripts/qwen36_candidate_env_moe_folded_sidecar.env`

### Active-MoE Fixture Contract Gate

P133/P134 add a fast target for native active-MoE development.  P133 stream-loads
the official 35B W4A16 model layer by layer, runs real prompt prefill, splits
each target block at the attention/MoE boundary, and writes the final-token MoE
input plus routing/output tensors into tiny safetensors fixtures.  P134 reloads
those fixtures and verifies the reference MoE path reproduces the stored output.
The v2 fixture schema records tensor shapes/dtypes, token position, fixture
sha256, full MoE output, and routed-only output.  Optional debug export can also
include router logits and slot-level routed FFN intermediates.

R6000 result:

| Check | Result |
|---|---:|
| fixture set | `reports/qwen36_35b/p133_fixtures_official_w4a16/` |
| layers | 0, 4, 8, 16, 20, 28, 32, 36, 39 |
| prompts | 2 |
| fixtures | 18 |
| fixture schema | lynn-moe-fixture-v2 |
| export time | 16.99 s |
| layer load time | 14.71 s |
| fixture bytes | 268,061 |
| P134 self-check | GREEN, 18/18 |
| P134 candidate-output-dir self-check | GREEN, 18/18 |
| P134 routed-only self-check | GREEN, 18/18 |
| max abs / mean abs | 0.0 / 0.0 |
| mean reference latency | 1.022 ms |
| routed-only mean reference latency | 0.914 ms |

Decision: make P134 the first admission gate for Stream A native MoE kernel
candidates.  A candidate that cannot pass the 18-fixture fast target should not
spend R6000 time on full P37/P25/structured gates.  This keeps the mainline
quality bar strict while making kernel iteration cheaper.  Candidate developers
can either plug in a Python backend or write precomputed safetensors into a
candidate-output directory and let P134 compare them without loading layer
weights.  Output-owned/non-atomic routed-MoE candidates should start with
`--routed-only` against `routed_output`, then graduate to full MoE output after
the routed path is strict.  `scripts/r6000_qwen36_moe_fixture_gate.sh` wraps this
into the standard first gate and writes a `.summary.json` traffic light:
`CLOSED_NUMERIC`, `PASS_NUMERIC_ONLY`, `PASS_SLOW`, or `FAST_CANDIDATE`.

Artifacts:

- `benchmarks/p133_export_active_moe_fixtures.py`
- `benchmarks/p134_active_moe_fixture_contract.py`
- `scripts/r6000_export_qwen36_moe_fixtures.sh`
- `scripts/r6000_qwen36_moe_fixture_gate.sh`
- `scripts/summarize_qwen36_moe_fixture_gate.py`
- `reports/qwen36_35b/p133_fixtures_official_w4a16/manifest.json`
- `reports/qwen36_35b/p134_triton_selfcheck_report.json`
- `reports/qwen36_35b/p134_candidate_output_selfcheck_report.json`
- `reports/qwen36_35b/p134_routed_only_selfcheck_report.json`
- `reports/qwen36_35b/p134_routed_candidate_output_selfcheck_report.json`

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
| P25 server | minimum useful result is 512-token decode >=115 TPS |
| Failure mode | if first-token/min-prefix regression appears, close the branch |

Stretch target: 512-token decode >=120 TPS while preserving P37 exact-greedy.
Expected upside: 5-15% if boundary reduction is real. This alone may not close
155 TPS, but it is the highest-confidence kernel island.

2026-05-18 P125 update: current strict-boundary allowlist probes are still
closed. Full-attention layer candidates reached at most `1.087x` median speedup
but `0/3` exact with min prefix `2-3`; linear-attention layer candidates reached
at most `1.067x` median speedup but also `0/3` exact. Do not promote these
native candidates. The next MoE island needs tighter Triton-contract parity
before speed work resumes.

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
| P25 server | minimum useful result is 512-token decode >=113 TPS |
| P26/P28 | phase shift visible, not measurement noise |
| Hard structured gate | 40/40 if candidate is promoted beyond research |

Stretch target: 512-token decode >=118 TPS while preserving P37 exact-greedy.
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

2026-05-18 status: Stream C is implemented and validated on R6000. The safe
default is a `DEFAULT_CANDIDATE` under the full 40-request hard structured gate
(`P37` exact, P25 512 decode TPS `107.43`, hard structured `40/40`, mean decode
TPS `107.86`), while the fastest shared-gate + conv-inplace profile is
correctly classified as `AMBER_CANDIDATE` (`P37` drift, P25 512 decode TPS
`114.04`, hard structured `40/40`, mean decode TPS `114.30`). Future agents
should use `scripts/r6000_qwen36_candidate_promotion_gate.sh` before asking for
default promotion.

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
2. Own `resident_runner.py` and attention/linear-core probes. Stream C wrapper
   is already implemented, so reuse `scripts/r6000_qwen36_candidate_promotion_gate.sh`
   instead of creating a parallel gate.
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
