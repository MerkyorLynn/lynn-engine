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

