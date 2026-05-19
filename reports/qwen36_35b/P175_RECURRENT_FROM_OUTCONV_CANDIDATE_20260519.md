# P175 Recurrent From OutConv Candidate

Date: 2026-05-19

## Purpose

P175 is the first real opt-in code candidate from the P173/P174 linear-boundary work. It adds a Triton recurrent/GDN kernel that reads q/k/v directly from `out_conv`, avoiding Python q/k/v split, reshape, and contiguous materialization.

The candidate is controlled by:

```bash
LYNN_LINEAR_ATTN_RECURRENT_FROM_OUTCONV=1
```

Default serving remains unchanged.

## Fixture Result

Artifacts:

- Candidate: `benchmarks/p175_qwen36_recurrent_from_outconv_candidate.py`
- Wrapper: `scripts/r6000_qwen36_recurrent_from_outconv_candidate.sh`
- Candidate env: `scripts/qwen36_candidate_env_recurrent_from_outconv.env`
- JSON report: `reports/qwen36_35b/p175_recurrent_from_outconv_candidate_20260519_0856_outconv_clean.json`
- P169 report: `reports/qwen36_35b/p175_recurrent_from_outconv_candidate_20260519_0856_outconv_clean_p169_check.json`
- Diagnostics: `reports/qwen36_35b/p172_linear_core_candidate_diagnostics_20260519_0856_outconv_clean_diag.json`

P169 fixture gate:

| Metric | Result |
|---|---:|
| Passed | 20/20 |
| max_abs_max | 0.0 |
| cosine_min | 0.9999998211860657 |

Diagnostics hash match:

| Tensor | Exact Hash Matches |
|---|---:|
| `core_attn_out` | 20/20 |
| `conv_state_out` | 20/20 |
| `recurrent_state_out` | 20/20 |

Fixture timing:

| Stage | Median ms |
|---|---:|
| `in_proj` | 0.109049 |
| `split_z` | 0.017901 |
| `conv` | 0.064916 |
| `gate` | 0.050148 |
| `recurrent_from_outconv` | 0.065481 |
| `total` | 0.309200 |

Compared with P174 serving-like total `0.336122 ms`, this is about 8% faster at the fixture boundary.

## Resident Gate

Artifacts:

- P37: `reports/qwen36_35b/r6000_qwen36_w4a16_recurrent_from_outconv_20260519_0902_outconv_resident_p37.json`
- P25: `reports/qwen36_35b/r6000_qwen36_w4a16_recurrent_from_outconv_20260519_0902_outconv_resident_p25.json`
- Structured: `reports/qwen36_35b/r6000_qwen36_w4a16_recurrent_from_outconv_20260519_0902_outconv_resident_hard_structured.json`
- Summary: `reports/qwen36_35b/r6000_qwen36_w4a16_recurrent_from_outconv_20260519_0902_outconv_resident_promotion_summary.json`

| Gate | Result |
|---|---:|
| P37 exact | true |
| P37 median speedup | 1.004233 |
| P25 512 decode TPS | 107.995933 |
| Structured | 40/40 |
| Structured mean decode TPS | 107.864847 |
| Decision | CLOSED |

## Conclusion

The candidate is numerically safe and graph-compatible, but service gain is too small to promote. It remains a useful exact opt-in/research line and confirms that q/k/v split elimination alone is not enough for the 122 TPS target.

Next fused work should combine a larger boundary, likely `conv + gate + recurrent`, or move to a true allocation-free graph-safe boundary that removes more than just q/k/v views.

