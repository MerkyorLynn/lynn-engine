# P178 Full-Layer Native MoE P37 Probe

Date: 2026-05-19

## Purpose

Test whether the existing packed-NVFP4 native active-MoE candidates can be used
only on full-attention layers. This avoids the linear-block CUDA graph capture
hazard while still touching ten MoE layers in the 35B resident decode path.

## Candidates

Both probes used:

```text
LYNN_NATIVE_ACTIVE_MOE_LAYERS=full
LYNN_MOE_ACTIVE_SCRATCH=1
--graph-on
--max-new 32
```

| Candidate backend | P37 exact | Collapse | Candidate TPS readout | Decision |
|---|---:|---:|---|---|
| `grouped_per16_nonatomic_out` | 0/3 | false | prompt 1/2 around 105 TPS, prompt 0 stalled at 0.75 TPS | closed |
| `grouped_per16_nonatomic` | 0/3 | false | prompt 1/2 around 105 TPS, prompt 0 around 42 TPS | closed |

## Drift Pattern

The failures are early-token route drift, not token-0 graph collapse.

| Candidate | Prompt | First drift index | Baseline prefix | Candidate prefix |
|---|---:|---:|---|---|
| `grouped_per16_nonatomic_out` | 0 | 2 | `[271, 248068, 271, 248069, ...]` | `[271, 248068, 198, 8160, ...]` |
| `grouped_per16_nonatomic_out` | 1 | 3 | `[198, 727, 51184, 318, ...]` | `[198, 727, 51184, 1393, ...]` |
| `grouped_per16_nonatomic` | 0 | 2 | `[271, 248068, 271, 248069, ...]` | `[271, 248068, 198, 8160, ...]` |
| `grouped_per16_nonatomic` | 1 | 3 | `[198, 727, 51184, 318, ...]` | `[198, 727, 51184, 1393, ...]` |

## Decision

Do not run P25 or structured gates for these candidates. Full-layer-only native
MoE avoids the graph-collapse failure mode, but it does not preserve the P37
greedy token contract. The next 35B MoE work should either reproduce the Triton
active-MoE numerical contract exactly or move to a larger boundary where Triton
remains the exact authority.

## Artifacts

- `reports/qwen36_35b/p178_full_layers_grouped_nonatomic_out_20260519_090116.json`
- `reports/qwen36_35b/p178_full_layers_grouped_nonatomic_20260519_090347.json`
