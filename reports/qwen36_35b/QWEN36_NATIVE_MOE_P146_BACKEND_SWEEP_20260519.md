# Qwen3.6-35B W4A16 Native MoE P146 Backend Sweep

**Date:** 2026-05-19  
**Model:** `/root/autodl-tmp/models/Qwen3.6-35B-A3B-lynn-native-w4a16-nvfp4-v0`  
**Graph mode:** linear-block graph disabled  
**Purpose:** close or admit resident Native MoE backends before any P25 or structured gate.

## Verdict

**No resident Native MoE backend in this sweep passed P37.**

Keep the default serving path on Triton active MoE. Do not escalate any of these
backends to P25 or structured gates.

## Result Matrix

| Backend | Status | P37 exact | First failure | Candidate TPS signal |
|---|---|---:|---|---|
| `grouped_per16_nonatomic` | `CLOSED_P37_DRIFT` | 2/3 | prompt 2 token 2 drift | 10.47 / 33.11 / 33.14 |
| `cuda_scalar_contract` | `CLOSED_P37_DRIFT` | 1/3 | prompt 0 token 2 drift | 11.03 / 34.62 / 34.57 |
| `cuda_scalar` | `CLOSED_P37_DRIFT` | 1/3 | prompt 0 token 2 drift | 10.32 / 32.52 / 32.77 |
| `grouped_per16` | `ERROR_NO_REPORT` | - | fail-loud ABI, kernel not implemented | - |
| `grouped_per16_fused` | `ERROR_NO_REPORT` | - | fail-loud ABI, fused kernel not implemented | - |

The three runnable backends do not collapse, but all drift in true resident
decode. The first-token sequence pattern is consistent with earlier native MoE
drift reports: the candidate often diverges into the `[198, 8160, 579, 264,
7047, 1817]` continuation after token 2.

## Interpretation

This sweep closes the current stock backend pool as promotion candidates. The
remaining useful Native MoE work is not another env sweep; it needs a new
implementation that preserves the Triton active path's numerical contract while
reducing a real boundary. The fail-loud `grouped_per16` and
`grouped_per16_fused` ABI slots remain valid replacement points for that new
kernel, but they are not implemented production candidates today.

## Next Engineering Direction

1. Keep `LYNN_NATIVE_ACTIVE_MOE_BACKEND=triton` as the only default backend.
2. Treat the P146 sweep as the first admission gate for future resident MoE
   work.
3. Build the next candidate against a Triton-contract-preserving boundary:
   either fuse the existing Triton-compatible gate/up and down boundary, or
   implement a one-boundary grouped-per16 kernel that matches Triton rounding
   closely enough to pass P37 3/3 before speed testing.

## Artifacts

- `scripts/r6000_qwen36_moe_p146_backend_sweep.sh`
- `reports/qwen36_35b/p146_backend_sweep_20260519_025347_summary.json`
- `reports/qwen36_35b/p146_backend_sweep_20260519_025347/p146_grouped_per16_nonatomic.json`
- `reports/qwen36_35b/p146_backend_sweep_20260519_025347/p146_cuda_scalar_contract.json`
- `reports/qwen36_35b/p146_backend_sweep_20260519_025347/p146_cuda_scalar.json`
