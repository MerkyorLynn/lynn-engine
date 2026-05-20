# Qwen3.5-9B R6000 FP4xFP8 Stage Profile

Date: 2026-05-20

## Context

Qwen3.5-9B W4A8 quality is no longer the blocker. The open question is whether
R6000 can close the gap from the Lynn-native W4A16 baseline (~61.7 decode TPS)
toward the llama.cpp Q4_K_M class by replacing BF16/W4A16 dense FFN pieces with
true `E4M3 activation x E2M1 weight` native FP4 MMA.

P200 profiles the current FP4xFP8 dense FFN fixture path and a new dual gate/up
native ABI.

Artifacts:

- First profile: `reports/qwen35_9b/p200_dense_fp4xfp8_stage_profile_20260520_141854.json`
- Dual gate/up profile: `reports/qwen35_9b/p200_dense_fp4xfp8_stage_profile_20260520_142421.json`

## Build Finding

The first R6000 run failed when the native extension was built for plain
`sm_120`; ptxas rejected `.kind::f8f6f4`. The R6000 FP4 path must compile with:

- `LYNN_ENABLE_SM120A_FP4_MMA=1`
- `LYNN_NATIVE_CUDA_ARCH_AUTO=1` or an explicit `LYNN_NATIVE_CUDA_ARCH=sm_120a`

The P200 runner now sets these by default.

## Stage Breakdown

Mean over 8 fixtures (layers 0/8/16/31, two prompts):

| Metric | Mean |
|---|---:|
| Current full FFN island | 0.3825 ms |
| gate MMA | 0.0707 ms |
| up MMA | 0.0700 ms |
| down MMA | 0.1830 ms |
| projection MMA sum | 0.3237 ms |
| Torch/quant/intermediate boundary sum | 0.1504 ms |
| duplicated input quant | 0.0485 ms |

Current path is dominated by projection work plus boundary overhead. Removing
only the duplicated input quant gives about `1.05x`, so it is not enough to
close the llama.cpp gap.

## Dual Gate/Up Result

New native symbol:

- `dense_fp4xfp8_mma_scaled_dual_probe`

It consumes one shared quantized activation and computes gate/up in one native
kernel launch.

| Metric | Mean |
|---|---:|
| gate + up as two launches | 0.1407 ms |
| dual gate/up launch | 0.1085 ms |
| local speedup | 1.30x |
| estimated FFN stage speedup | 1.21x |
| gate output mismatch vs single launch | 0.0 |
| up output mismatch vs single launch | 0.0 |

## Interpretation

Dual gate/up is real and numerically identical to the existing single-projection
path, but it is not enough by itself. If promoted into resident decode without
further fusion, the likely end-to-end lift is closer to the 10-20% band than a
2x breakthrough.

To chase llama.cpp-class TPS on R6000, the next ROI path is:

1. Wire dual gate/up into the resident W4A8 FFN path behind an opt-in env.
2. Add a resident P25 + structured/content gate, not exact-greedy only.
3. Then fuse `SiLU * up + down` or make a larger native-owned FFN boundary so
   intermediate tensors do not round-trip through Torch.

MTP is not the current R6000 9B speed lever: the official 9B MTP head works, but
the speculative path falls out of graph/native decode and loses to the 61.7 TPS
baseline. The highest ROI remains true W4A8 FP4xFP8 boundary work.
