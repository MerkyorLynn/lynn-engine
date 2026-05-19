# P176 Recurrent From OutConv + A/B Candidate

Date: 2026-05-19

## Purpose

P176 tests a larger boundary than P175. It reads q/k/v directly from `out_conv` and moves beta/g computation into the Triton recurrent kernel:

```text
out_conv + a_raw + b_raw + A/dt -> recurrent/GDN
```

The goal was to remove the separate Python `sigmoid` / `softplus` gate-prep stage.

## Artifacts

- Triton candidate: `triton_kernels/gated_delta.py`
- Probe: `benchmarks/p176_qwen36_recurrent_from_outconv_ab_candidate.py`
- Wrapper: `scripts/r6000_qwen36_recurrent_from_outconv_ab_candidate.sh`
- JSON report: `reports/qwen36_35b/p176_recurrent_from_outconv_ab_candidate_20260519_0918_outconv_ab.json`
- P169 report: `reports/qwen36_35b/p176_recurrent_from_outconv_ab_candidate_20260519_0918_outconv_ab_p169_check.json`

## R6000 Fixture Result

| Metric | Result |
|---|---:|
| P169 passed | 0/20 |
| max_abs_max | 0.015341758728027344 |
| cosine_min | 0.9999949932098389 |

Timing was faster than P175:

| Stage | Median ms |
|---|---:|
| `in_proj` | 0.113695 |
| `split_zab` | 0.019126 |
| `conv` | 0.069393 |
| `recurrent_from_outconv_ab` | 0.083838 |
| `total` | 0.285646 |

## Conclusion

P176 is closed for promotion. It proves there is fixture-level speed available if beta/g prep moves into Triton, but the PyTorch-to-Triton sigmoid/softplus transfer is not bit-exact under the current contract.

Do not escalate to P37/P25. A future larger boundary must either:

- preserve PyTorch-produced `beta` and `g` as explicit inputs, or
- intentionally relax the exactness contract, which is not acceptable for default promotion.

