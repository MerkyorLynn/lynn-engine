# P190 Qwen3.5-9B FP4xFP8 Resident Findings

Date: 2026-05-19

## Summary

The R6000 true `E4M3 x E2M1` dense-FFN bridge is now wired into the resident
decode path behind opt-in envs:

- `LYNN_DENSE_FFN_TRUE_FP8=1`
- `LYNN_DENSE_FFN_TRUE_FP8_SIDECAR_DIR=/root/autodl-tmp/reports/qwen35_9b/p192_dense_fp4x_fp8_sidecar`
- optional `LYNN_DENSE_FFN_TRUE_FP8_LAYERS=...`

Default serving is unchanged. The resident hook only takes over single-token
decode (`B*T == 1`); prefill and warmup stay on the BF16/W4A16 path.

## Gate Result

The fixture path is promising, but full resident promotion is blocked:

- P195 full dense FFN fixture gate: `AMBER_NUMERIC`
- P195 8-fixture mean native time: `0.384 ms/layer`
- P195 mean speedup vs scalar reference: `7.77x`
- P190 resident full-32-layer gate: `TRUE_FP8_RESIDENT_RED`
- P190 full-32-layer decode TPS: `65.70` vs `60.11` reference (`1.093x`)
- P190 full-32-layer exact: `0/6`, early special-token drift

This means the math island is real, but the current all-layer W4A8 resident
path is not structured-safe yet.

## Layer Sweep

| Layer spec | Verdict | Exact | Decode TPS | Speedup |
|---|---:|---:|---:|---:|
| all | RED | 0/6 | 65.72 | 1.102x |
| prewarm_all | RED | 0/6 | 65.70 | 1.093x |
| 0-30 | RED | 0/6 | 67.14 | 1.121x |
| 8-31 | RED | 0/6 | 64.30 | 1.071x |
| 0-23 | RED | 0/6 | 63.70 | 1.061x |
| 0-15 | RED | 0/6 | 62.59 | 1.043x |
| 16-31 | RED | 0/6 | 62.13 | 1.036x |
| 24-31 | RED | 0/6 | 61.03 | 1.024x |
| 31 | EXACT | 6/6 | 58.43 | 0.980x |

Single-layer probes mostly drift on 1/3 or 0/3 prompts while giving only a
small apparent speedup. Layer 31 can be exact, but it is slower after native
extension prewarm, so it is not a useful promotion candidate.

## Decision

Do not promote `LYNN_DENSE_FFN_TRUE_FP8` into any default or AMBER serving
profile yet.

The next useful engineering step is not more layer-mask sweeps. It is a tighter
resident boundary:

1. Fuse gate/up into one native projection launch.
2. Keep the intermediate in a native-owned buffer rather than returning through
   Python/Torch between stages.
3. Add a W4A8 structured/content gate instead of using exact-greedy parity as
   the only safety signal.

