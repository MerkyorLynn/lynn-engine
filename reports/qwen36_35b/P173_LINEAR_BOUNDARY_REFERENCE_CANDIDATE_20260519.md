# P173 Linear Boundary Reference Candidate

Date: 2026-05-19

## Purpose

P173 narrows the first fused linear-core target to the exact boundary planned for native work:

```text
in_proj -> conv -> recurrent/GDN
```

It deliberately excludes `gated_norm` and `out_proj`. The candidate output directory writes:

- `core_attn_out`
- `recurrent_state_out`
- `conv_state_out`
- `z`

Only the first three are required by the P169/P172 candidate checks. `z` is included for the next downstream boundary.

## Artifacts

- Producer: `benchmarks/p173_qwen36_linear_boundary_reference_candidate.py`
- R6000 wrapper: `scripts/r6000_qwen36_linear_boundary_reference_candidate.sh`
- Candidate report: `reports/qwen36_35b/p173_linear_boundary_reference_candidate_20260519_0824_boundary.json`
- P169 gate report: `reports/qwen36_35b/p173_linear_boundary_reference_candidate_20260519_0824_boundary_p169_check.json`
- Diagnostics report: `reports/qwen36_35b/p172_linear_core_candidate_diagnostics_20260519_0824_boundary_diag.json`

## R6000 Result

| Check | Result |
|---|---:|
| Fixtures | 20 |
| Tensors written | 80 |
| P169 candidate gate | 20/20 GREEN |
| max_abs_max | 0.0 |
| cosine_min | 0.9999998211860657 |
| compute_ms_mean | 0.37005546813209855 |
| compute_ms_median_of_medians | 0.3576003946363926 |
| compute_ms_min | 0.3428952768445015 |
| compute_ms_max | 0.4046754911541939 |

Diagnostics with hash matching also passed:

| Tensor | Exact Hash Matches |
|---|---:|
| `core_attn_out` | 20/20 |
| `conv_state_out` | 20/20 |
| `recurrent_state_out` | 20/20 |

## Implication

The first native fused-boundary candidate should beat roughly `0.36 ms` per sampled linear-attention fixture while preserving byte-exact output for:

- `core_attn_out`
- `conv_state_out`
- `recurrent_state_out`

This is a tighter and more honest target than the full linear-core P172 baseline (`~0.56 ms`) because it excludes `gated_norm/out_proj`, which are intentionally deferred to a second boundary.

Escalation remains:

1. P173/P169 fixture gate 20/20 exact.
2. P172 diagnostics hash preflight, if exact output is expected.
3. P37 exact with graph-on serving.
4. P25 512 and structured gates.

