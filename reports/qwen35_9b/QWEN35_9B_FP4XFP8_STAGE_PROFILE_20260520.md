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

## Resident Gate Follow-Up

P190/P197 checked whether the dual gate/up path can be used directly in resident
decode. It cannot yet.

Artifacts:

- `reports/qwen35_9b/p190_qwen35_9b_true_fp8_resident_gate_dual_gateup_20260520_143236.json`
- `reports/qwen35_9b/p190_qwen35_9b_true_fp8_resident_gate_gateup_dual_20260520_143548.json`
- `reports/qwen35_9b/p190_qwen35_9b_true_fp8_resident_gate_down_dual_20260520_143709.json`
- `reports/qwen35_9b/p197_gateup_dual_token_drift_20260520_144116.json`

| Candidate | Exact | Decode TPS | Speedup | Verdict |
|---|---:|---:|---:|---|
| full FP4xFP8, dual gate/up | 0/6 | 71.74 | 1.20x | RED |
| gate/up-only FP4xFP8, dual gate/up | 0/6 | 70.01 | 1.17x | RED |
| down-only FP4xFP8 | 0/6 | 58.05 | 0.97x | RED |

P197 top-k drift for gate/up-only:

| Metric | Value |
|---|---:|
| decision | CLOSED |
| exact top-k steps | 5/40 |
| first drift step | 1 |
| drift ratio | 87.5% |
| top-5 jaccard mean | 0.1486 |
| shared cosine mean | 0.2737 |
| combined score | 0.2112 |

Important detail: step 0 top-k is identical, then step 1 collapses. The resident
problem is not just exact-greedy sensitivity and not a local lm_head artifact.
The FP8 gate/up hidden-state delta is being amplified by the next decode step.

To chase llama.cpp-class TPS on R6000, the next ROI path is:

1. Keep dual gate/up as an opt-in research backend only.
2. Run per-layer hidden/logit drift isolation for gate/up-only to find whether
   this is an activation-scale convention issue or unavoidable W4A8 hidden drift.
3. Only after top-k recovers should we run resident P25 + structured/content
   gates. Exact-greedy alone is too strict, but P197 proves current top-k is too
   far off for AMBER.
4. Then fuse `SiLU * up + down` or make a larger native-owned FFN boundary so
   intermediate tensors do not round-trip through Torch.

MTP is not the current R6000 9B speed lever: the official 9B MTP head works, but
the speculative path falls out of graph/native decode and loses to the 61.7 TPS
baseline. The highest ROI remains true W4A8 FP4xFP8 boundary work.
