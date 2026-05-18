# Qwen3.6 W4A16 Parallel Stream Handoff

Date: 2026-05-18

## Current Goal

Primary runtime target:

```text
official Qwen/Qwen3.6-35B-A3B
  -> Lynn-native W4A16 NVFP4
  -> R6000 safe default 107 TPS
  -> strict kernel work toward 118-125 TPS safe / 125-130 TPS AMBER
```

Spark serving is now a separate product route: Q4_K_M-imatrix GGUF through
llama.cpp is the current Spark default at about `70 TPS`. Do not spend the main
Lynn-native kernel loop trying to force Spark `sm_121` to match R6000.

## Ownership

### Stream A: Native MoE Kernel Island

Owner: Codex CLI / DeepSeek branch.

Branch:

```text
codex/native-moe-kernel-island
```

Scope:

- `engine/moe_packed_nvfp4.py`
- `engine/native_cuda.py`
- `csrc/lynn_native/bindings.cpp`
- `csrc/lynn_native/moe_scalar_kernel.cu`
- optional `triton_kernels/nvfp4_moe.py`
- `benchmarks/p121_active_moe_strict_boundary_probe.py`
- `benchmarks/p122_active_moe_strict_boundary_generate_gate.py`

Contract:

```text
gate/up -> bf16 inter store -> down -> route weighted sum
```

Do not change service-loop files. Do not promote native shortcuts that change
exact greedy text. Candidate backend name should stay fail-loud, for example
`LYNN_NATIVE_ACTIVE_MOE_BACKEND=strict_fused_boundary`.

### Stream B: Full-Attention and Linear-Core Fusion

Owner: Claude branch.

Suggested branch:

```text
claude/phase-a-foundation-20260517
```

Scope:

- `engine/resident_runner.py`
- `engine/incremental_decode.py`
- `engine/qwen36_linear_attn_block.py`
- optional `triton_kernels/full_attn_rope_cache.py`
- `benchmarks/p123_full_attn_strict_cache_probe.py`
- `benchmarks/p124_linear_core_boundary_probe.py`

Tasks:

1. Turn the full-attention RoPE cache into an owned module with explicit
   prewarm and max-seq handling.
2. Keep q/k/v projection order intact; do not revive naive qkv row concat.
3. Remove avoidable RoPE/cache/mask allocation while preserving exact greedy
   output.
4. Probe linear-attention boundaries around:
   - native FP4 in-proj, about `0.077 ms/layer`;
   - recurrent fused prepare, about `0.036 ms/layer`;
   - conv update, about `0.026-0.033 ms/layer`;
   - gated RMSNorm, about `0.020 ms/layer`.

Do not touch native MoE files owned by Stream A.

### Stream C: Promotion Gate

Owner: Codex integration; Claude can consume and extend.

Status: implemented and validated.

Main wrapper:

```text
scripts/r6000_qwen36_candidate_promotion_gate.sh
```

Candidate env files:

```text
scripts/qwen36_candidate_env_amber_sharedgate_convinplace.env
scripts/qwen36_candidate_env_template.env
```

Validated gates:

| Route | Decision | Key result |
|---|---|---|
| Safe default | `DEFAULT_CANDIDATE` | P37 exact, P25 512 decode `107.43`, hard structured `40/40` |
| shared-gate Triton + conv Triton inplace | `AMBER_CANDIDATE` | P37 drift, P25 512 decode `114.04`, hard structured `40/40` |

Use this wrapper for every candidate before requesting promotion.

R6000 note: remote `git fetch` is intermittently failing with TLS/RPC errors, so
the latest wrapper/env/prompt files were also copied directly to:

```text
/root/autodl-tmp/lynn-engine/scripts/
```

## Acceptance Ladder

Default promotion:

1. P37 exact greedy `3/3`.
2. P25 server positive on 256/512, with 512-token decode above the safe default
   threshold.
3. hard structured OpenAI gate `40/40`.
4. no first-token or min-prefix regression.

AMBER promotion:

1. hard structured OpenAI gate `40/40`.
2. P25 512-token decode at least 5% above safe default.
3. exact-greedy drift is documented and normal-looking.
4. remains opt-in.

Closed:

1. structured/code/tool-call format failure;
2. repeated-token collapse;
3. first-token route drift;
4. P25 server regression even if microbench or P37 speed looks positive.

## Current Baseline

Safe default:

```text
P37 exact: true
P25 512 decode TPS: 107.43
hard structured: 40/40
hard structured mean decode TPS: 107.86
```

AMBER fast:

```text
P37 exact: false
P25 512 decode TPS: 114.04
hard structured: 40/40
hard structured mean decode TPS: 114.30
```

Near-term target:

```text
safe default: 107 -> 118-125 TPS
AMBER structured: 114 -> 125-130 TPS
```

## 2026-05-18 14:00 R6000 Fast122 Sweep

R6000 re-swept the existing switch space before asking the kernel branches to
chase 122 TPS. Result: the old toggle pool is exhausted. The current AMBER
shared-gate + conv-inplace profile remains the fastest known candidate, and it
still has documented exact-greedy drift.

| Candidate | P37 exact | Candidate median decode TPS | Decision |
|---|---:|---:|---|
| `LYNN_MOE_ADD_SHARED_INPLACE=1` | yes | 107.86 | small safe signal, full gate running |
| current AMBER shared-gate + conv-inplace | no | 114.26 | still best AMBER baseline |
| AMBER + `LYNN_MOE_FAST_FIXED=0` | no | 113.09 | closed, slower |
| AMBER + packed linear decode | no | 69.12 | closed, severe regression |
| AMBER + sorted router / non-fixed MoE | no | 111.36 | closed, slower |
| AMBER + top-k 7 | no | 112.75 | closed, expert dropping drifts and does not beat AMBER |
| shared scalar-add Triton variants | no | 109-110 | closed, greedy drift |
| top-k 6/7 only | no | 106-107 | closed, no useful speed |
| `cuda_tile` down on full/layer0 | no | 107.86 | closed, drift/repeated-token risk |
| `cuda_tile_inter` gate/up on full layers | no | 106.31 | closed, slower |

Implication for external branches:

- Stream A must deliver a real strict MoE fused boundary. Single-kernel native
  substitutions and top-k shortcuts are not enough.
- Stream B must deliver a real attention/linear workspace or boundary change.
  Reusing old packed-linear/full-attn switches is not enough.
- Stream C should reject average-TPS-only reports. Each candidate needs P37,
  P25 512-token service TPS, and hard structured results.

Concrete branch goals:

| Stream | Minimum useful result | Stretch result |
|---|---:|---:|
| A Native MoE | P37 exact, hard structured 40/40, P25 512 decode >=115 TPS | >=120 TPS |
| B Attn/linear core | P37 exact, P25 512 decode >=113 TPS, visible P26/P28 phase win | >=118 TPS |
| A+B combined | hard structured 40/40 or 70/70 AMBER, P25 512 decode >=122 TPS | >=125 TPS |
