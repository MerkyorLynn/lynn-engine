# Qwen3.6 35B W4A16 P130 MoE Effective-Scale Repack

Date: 2026-05-18

## Candidate

`LYNN_MOE_EFFECTIVE_SCALE=1` is an opt-in MoE repack/kernel-boundary probe.
At load time it replaces each active-MoE scale tensor with
`scale / global_scale` and sets the corresponding global scale to `1`.
This is intentionally memory-neutral: keeping both original and effective
scales resident adds several GiB and OOMs the 35B runner on R6000.

The runtime path still preserves the strict operation order:

```text
gate/up -> bf16 inter store -> down -> route weighted sum
```

## Local Active-MoE Probe

`benchmarks/p130_moe_effective_scale_probe.py` compared the current sidecar-fed
Triton active-MoE boundary against the effective-scale kernels on 9
representative layers.

| Metric | Result |
|---|---:|
| Exact output | 9/9 |
| Max abs | 0.0 |
| Max rel L2 | 0.0 |
| Min cosine | 0.99999994 |
| Mean current active-MoE | 0.05594 ms |
| Mean effective-scale active-MoE | 0.05147 ms |
| Boundary speedup | 1.087x |

## R6000 Promotion Gate

| Gate | Result |
|---|---:|
| P37 exact greedy | true |
| P37 median speedup | 1.009x |
| P25 512 decode TPS | 107.98 |
| Hard structured | 40/40 |
| Structured mean decode TPS | 108.40 |
| Decision | research-only, below promotion margin |

## Decision

This is a real, strict MoE repack win, but not a 122 TPS-class breakthrough.
Keep it as an opt-in building block for the next native MoE island. It proves
that scale/global division can be removed without drift, and it slightly lowers
the active-MoE floor before the real grouped native-FP4 gate/up/down kernel.

Artifacts:

- `benchmarks/p130_moe_effective_scale_probe.py`
- `scripts/qwen36_candidate_env_moe_effective_scale.env`
- `reports/qwen36_35b/p130_moe_effective_scale_probe_20260518.json`
- `reports/qwen36_35b/r6000_qwen36_w4a16_moe_effective_scale_20260518_171827_moe_eff_replace_promotion_summary.json`
