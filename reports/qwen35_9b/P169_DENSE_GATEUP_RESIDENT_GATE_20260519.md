# P169 Qwen3.5-9B Dense Gate/Up Resident Gate

Date: 2026-05-19

## Purpose

P168 showed that a fused dense FFN gate/up projection is exact and about 2%
faster on isolated fixtures.  P169 wires the same idea into the resident path
behind `LYNN_DENSE_FFN_GATE_UP_FUSED=1` and compares it against the current
safe 9B `linear_graph_only` profile.

## Direct Resident Gate

| Metric | Result |
|---|---:|
| Greedy exact | 6/6 |
| Mean decode TPS, baseline | 59.29 |
| Mean decode TPS, fused gate/up | 61.14 |
| Mean speedup | 1.0313x |
| Minimum speedup | 1.0288x |
| Decision | resident candidate |

## Service Gate

| Metric | Baseline P150 | Fused gate/up P150 |
|---|---:|---:|
| 128 decode TPS | 60.80 | 61.41 |
| 256 decode TPS | 61.47 | 62.01 |
| 512 decode TPS | 61.69 | 61.85 |
| 512 speedup | - | 1.0026x |

## Decision

Keep the implementation as an opt-in exact path, but do not make it the default
9B profile yet.  The direct resident replay sees a clean 3% gain against the
safe `linear_graph_only` profile, while the OpenAI service gate is effectively
flat at 512 tokens.  This confirms the boundary is safe and exact, but the
current service bottleneck is no longer only the dense gate/up split.

Note: an earlier P169 local run used the broader P148 full-fast environment,
which is not the safe 9B profile.  The corrected artifact is the
`0822_densegateup_linearonly` report below.

## Next Step

Use this as a stable building block for a larger dense FFN boundary: fused
gate/up plus activation plus down, or an offline repacked TensorCore path.  The
promotion bar remains exact greedy parity plus a clear service TPS win.

## Artifacts

- `benchmarks/p169_qwen35_9b_dense_gateup_resident_gate.py`
- `scripts/r6000_qwen35_9b_dense_gateup_resident_gate.sh`
- `reports/qwen35_9b/p169_dense_gateup_resident_gate_20260519_0822_densegateup_linearonly.json`
- `reports/qwen35_9b/p150_qwen35_9b_nvfp4_linear_graph_p25_20260519_0800_densegateup_p150.json`
- `reports/qwen35_9b/p150_qwen35_9b_nvfp4_linear_graph_summary_20260519_0800_densegateup_p150.json`
