# P172 Linear Core Reference Candidate

Date: 2026-05-19

## Purpose

P171 proved that P169 candidate-output-dir plumbing works. P172 adds a computed reference candidate: load the model once, recompute the linear core from each P169 fixture input, emit only the minimum output tensors, and run the same P169 candidate gate.

This gives future fused `in_proj -> conv -> recurrent/GDN` kernels a clean fixture baseline before any P37/P25/structured service gate.

## Artifacts

- Producer: `benchmarks/p172_qwen36_linear_core_reference_candidate.py`
- Producer wrapper: `scripts/r6000_qwen36_linear_core_reference_candidate.sh`
- Diagnostics: `benchmarks/p172_qwen36_linear_core_candidate_diagnostics.py`
- Diagnostics wrapper: `scripts/r6000_qwen36_linear_core_candidate_diagnostics.sh`
- Reference candidate report: `reports/qwen36_35b/p172_linear_core_reference_candidate_20260519_0816_refwarm.json`
- P169 gate report: `reports/qwen36_35b/p172_linear_core_reference_candidate_20260519_0816_refwarm_p169_check.json`
- Fixture diagnostics: `reports/qwen36_35b/p172_linear_core_candidate_diagnostics_20260519_0816_fixtures.json`
- Candidate diagnostics: `reports/qwen36_35b/p172_linear_core_candidate_diagnostics_20260519_0816_refwarm_diag.json`

## R6000 Result

| Check | Result |
|---|---:|
| P172 fixtures | 20 |
| Minimum tensors written | 60 |
| Output mode | `only-final` |
| P169 candidate gate | 20/20 GREEN |
| max_abs_max | 0.0 |
| cosine_min | 0.9999998211860657 |
| compute_ms_mean | 0.575262416775028 |
| compute_ms_median_of_medians | 0.5613099783658981 |
| compute_ms_min | 0.5338927730917931 |
| compute_ms_max | 0.6343740969896317 |

The first un-warmed run was polluted by Triton/model warmup. The committed report uses one warmup run and three timed runs per fixture.

## Diagnostics Result

| Check | Result |
|---|---:|
| Fixture ABI | 20/20 GREEN |
| Optional `z` present | 20/20 |
| Optional `core_attn_out` present | 20/20 |
| Candidate preflight | 20/20 GREEN |
| `linear_core_out` byte hash matches | 20/20 |
| `conv_state_out` byte hash matches | 20/20 |
| `recurrent_state_out` byte hash matches | 20/20 |

## Admission Rule

A future fused linear-core candidate should first emit a candidate output directory containing at least:

- `linear_core_out`
- `conv_state_out`
- `recurrent_state_out`

It must pass:

```bash
python benchmarks/p169_qwen36_linear_core_fixture_contract.py \
  --fixtures /root/autodl-tmp/reports/qwen36_35b/p169_linear_core_fixtures_official_w4a16_20260519_0750 \
  --candidate-output-dir <candidate-output-dir> \
  --out <p169-candidate-report.json> \
  --check
```

Then run P172 diagnostics with `REQUIRE_HASH_MATCH=1` if the candidate is supposed to be byte-exact against the reference output. Only after this gate is GREEN should a candidate move to P37/P25/structured.

