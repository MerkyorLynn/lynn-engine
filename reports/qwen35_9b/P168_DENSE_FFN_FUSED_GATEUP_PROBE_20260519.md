# P168 Qwen3.5-9B Dense FFN Fused Gate/Up Probe

Date: 2026-05-19

## Purpose

P164/P165 closed the existing packed NVFP4 dense FFN wrappers, and P167 showed
that caller-owned Torch `mm(out=...)` buffers are exact but flat.  P168 tests
the smallest exact-first FFN boundary with real work removed: concatenate
`gate_proj` and `up_proj` weights once, run a single `F.linear`, chunk the
result into gate/up, then keep SiLU and down projection unchanged.

## R6000 Result

| Metric | Result |
|---|---:|
| Exact | 8/8 |
| Max abs | 0.0 |
| Min cosine | 1.0 |
| Reference FFN | 0.21582 ms |
| Fused gate/up FFN | 0.21163 ms |
| Mean speedup | 1.01979x |
| Minimum speedup | 1.01898x |
| Decision | fused gate/up candidate |

## Decision

Keep as a resident opt-in candidate for Qwen3.5-9B dense.  The gain is small
per fixture, but it is exact and applies to all 32 dense FFN layers, so it is
worth a focused resident-path gate.  It should not be treated as a 35B MoE
solution; it is the dense-model analogue of reducing launch/matmul boundaries.

## Next Probe

Wire a guarded load-time dense `gate/up` fusion behind
`LYNN_DENSE_FFN_GATE_UP_FUSED=1`, then run the 9B NVFP4 service gate.  Promote
only if greedy parity remains exact and 512-token decode TPS improves over the
current linear-graph default.

## Artifacts

- `benchmarks/p168_qwen35_9b_dense_gateup_fused_probe.py`
- `scripts/r6000_qwen35_9b_dense_ffn_p168_fused_gateup_probe.sh`
- `reports/qwen35_9b/p168_dense_ffn_fused_gateup_probe_20260519_0732_fused_gateup.json`
