# P174 Linear Boundary Stage Timing

Date: 2026-05-19

## Purpose

P174 profiles the exact P173 boundary on P169 fixtures:

```text
in_proj -> conv -> reshape/gate prep -> recurrent/GDN
```

It keeps the same exactness target as P173 and does not change serving defaults.

## Artifacts

- Probe: `benchmarks/p174_qwen36_linear_boundary_stage_timing.py`
- R6000 wrapper: `scripts/r6000_qwen36_linear_boundary_stage_timing.sh`
- JSON report: `reports/qwen36_35b/p174_linear_boundary_stage_timing_20260519_0830_stage.json`
- Serving-like state-scratch report: `reports/qwen36_35b/p174_linear_boundary_stage_timing_20260519_0839_stage_outside.json`

## R6000 Result

All 20 fixtures passed exact output checks.

| Stage | Median ms | Mean ms | Min ms | Max ms |
|---|---:|---:|---:|---:|
| `in_proj` | 0.105551 | 0.106826 | 0.099338 | 0.125247 |
| `split` | 0.016291 | 0.016391 | 0.015349 | 0.019095 |
| `conv` | 0.071244 | 0.072078 | 0.067958 | 0.085799 |
| `reshape_gate` | 0.060903 | 0.061256 | 0.059357 | 0.067171 |
| `recurrent_gdn` | 0.069037 | 0.069745 | 0.066711 | 0.078110 |
| `total` | 0.324505 | 0.327424 | 0.311388 | 0.371170 |

## Implication

The first boundary is not dominated by a single massive stage.  The biggest individual stage is `in_proj` at roughly `0.106 ms`, but the more attractive fused target is:

```text
conv + reshape_gate + recurrent_gdn
```

Those three together are roughly `0.201 ms` per sampled fixture and include multiple Python/Triton tensor boundaries.  A fused candidate here can avoid:

- conv output materialization followed by q/k/v split
- q/k/v/z reshape glue
- separate beta/g prep tensors
- separate recurrent kernel launch after conv

The admission rule remains strict:

1. Emit `core_attn_out`, `conv_state_out`, and `recurrent_state_out`.
2. Pass P169 candidate-output-dir 20/20 exact.
3. Pass P172 diagnostics if exact hashes are expected.
4. Only then run P37/P25/structured.

## State Scratch Check

A follow-up run used `STATE_COPY_MODE=outside`, preparing state scratch before timing to better approximate the serving in-place state path. It also passed 20/20 exact:

| Stage | Median ms |
|---|---:|
| `in_proj` | 0.115953 |
| `split` | 0.017939 |
| `conv` | 0.066647 |
| `reshape_gate` | 0.067724 |
| `recurrent_gdn` | 0.066492 |
| `total` | 0.336122 |

This is the same class as the original `0.324505 ms` total, so the boundary is not hiding a large state-clone tax. The next fused candidate should focus on removing inter-stage materialization and launches, not on fixture clone mechanics.
