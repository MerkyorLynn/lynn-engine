# DeepSeek Task - Native MoE Drift Reduction

## Context

Default serving is still official Qwen3.6-35B-A3B W4A16 NVFP4 with Triton active MoE. The packed native MoE line has strong speed signal but fails exact-greedy generation.

Current evidence on R6000:

| Candidate | Gate | Result |
|---|---|---|
| KIMI2.6 `native_output_owned_bf16` | p134 routed-only relaxed | 18/18 GREEN, `candidate_ms_mean=0.05047ms`, `max_abs_max=3.90625e-3`, `cosine_min=0.999980211`, exact 0/18 |
| Codex `grouped_per16_nonatomic_out` | p134 routed-only relaxed | 18/18 GREEN, `candidate_ms_mean=0.05317ms`, `max_abs_max=3.90625e-3`, `cosine_min=0.999975026`, exact 0/18 |
| Codex `grouped_per16_nonatomic_out` | P37 graph-on | median 129.74 TPS, 1.242x speedup, exact RED |
| Selected-layer native subsets | P37 graph-on | all exact RED |

Conclusion: the scheduling direction is correct; the blocker is numeric drift amplification, not Python overhead or CUDA graph capture.

## Ownership

Work on a new branch:

```bash
git checkout -b deepseek/native-moe-drift-reduction-20260518
```

Allowed write scope:

- `csrc/lynn_native/moe_scalar_kernel.cu`
- `csrc/lynn_native/bindings.cpp`
- `benchmarks/candidates/native_grouped_per16_nonatomic*.py`
- new `benchmarks/p13*_moe_*` probes
- new reports under `reports/qwen36_35b/`

Do not touch:

- `server/*`
- `engine/resident_runner.py`
- `engine/incremental_decode.py`
- `triton_kernels/*`
- Stream B full-attention / linear-attention files

## Goal

Reduce packed NVFP4 active-MoE drift while preserving output-owned / non-atomic speed structure.

Primary acceptance:

1. p134 routed-only strict improves materially:
   - current strict exact is 0/18
   - target: exact_count > 0/18, or `max_abs_max <= 1e-3` with `cosine_min >= 0.99999`
2. p134 relaxed remains FAST_CANDIDATE:
   - `candidate_ms_mean <= 0.065ms`
3. P37 on graph-on does not collapse and reports either:
   - exact GREEN, or
   - a smaller first divergence than current candidate with documented first-token/top-k margin evidence

Stretch:

- P37 exact GREEN with candidate median TPS >= 115.

## Suggested Experiments

### A. Match Triton Rounding More Closely

The existing packed path likely fails from tiny but systematic accumulation/rounding differences.

Try variants behind new explicit functions or env-gated candidate modules:

- explicit `__float2bfloat16_rn`
- route-weight multiply order: `(down_acc * route)` vs accumulating `route * term`
- per-slot partial sum then route accumulation, instead of route inside every inner term
- slot order exactly matching `expert_ids` and routing weights with no reordering
- optional Kahan-style FP32 accumulation in down stage only

Do not change default existing functions; create new candidate names so old reports remain reproducible.

### B. Down-Only and Gate-Up-Only Attribution

Use p134 or a small new fixture probe to compare:

- native gate/up + Triton/PyTorch down
- Triton/PyTorch gate/up + native down
- native gate/up out ABI only
- native down out ABI only

Report which stage contributes most to p134 `max_abs_max` and P37 top-k drift.

### C. Output-Owned Packed NVFP4 Variant

Port the output-owned scheduling idea from `moe_output_owned_bf16.cu` to packed NVFP4 weights:

- keep output-owned down blocks
- avoid atomics
- keep graph-safe caller-owned scratch
- do not use dequantized BF16 full expert weights in production candidate

## Required Commands on R6000

Use the existing fixture gate first:

```bash
cd /root/autodl-tmp/lynn-engine
ROUTED_ONLY=1 \
CANDIDATE_BACKEND=<candidate_name> \
WARMUP=3 \
ITERS=10 \
LYNN_NATIVE_CUDA_BUILD_DIR=/tmp/lynn_engine_native_build/<candidate_stamp> \
bash scripts/r6000_qwen36_moe_fixture_gate.sh
```

For relaxed speed admission:

```bash
ROUTED_ONLY=1 \
CANDIDATE_BACKEND=<candidate_name> \
MAX_ABS_THRESHOLD=0.004 \
COSINE_THRESHOLD=0.9999 \
ALLOW_NONEXACT=1 \
WARMUP=5 \
ITERS=20 \
LYNN_NATIVE_CUDA_BUILD_DIR=/tmp/lynn_engine_native_build/<candidate_stamp> \
bash scripts/r6000_qwen36_moe_fixture_gate.sh
```

Only run P37 if p134 relaxed is FAST_CANDIDATE and strict is improved:

```bash
LYNN_NATIVE_CUDA_BUILD_DIR=/tmp/lynn_engine_native_build/<candidate_stamp> \
/root/autodl-tmp/conda-envs/r6000-eval/bin/python benchmarks/p37_moe_config_generate_gate.py \
  --model /root/autodl-tmp/models/Qwen3.6-35B-A3B-lynn-native-w4a16-nvfp4-v0 \
  --out /root/autodl-tmp/reports/qwen36_35b/p37_<candidate_stamp>.json \
  --max-new 64 \
  --candidate LYNN_NATIVE_ACTIVE_MOE_BACKEND=<runtime_backend_if_any> \
  --candidate LYNN_MOE_ACTIVE_SCRATCH=1 \
  --candidate LYNN_MOE_FAST_FIXED=0
```

## Deliverable

Push the branch and provide:

- changed file list
- p134 strict JSON
- p134 relaxed summary JSON
- P37 JSON if applicable
- one concise verdict: `CLOSED_NUMERIC`, `FAST_BUT_DRIFT`, or `PROMOTION_CANDIDATE`
