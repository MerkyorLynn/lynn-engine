# P165 Qwen3.5-9B Dense FFN Stage Drift Probe

Date: 2026-05-19

## Purpose

P165 is a reporting-only probe over existing P159 dense FFN fixtures.  It
compares packed NVFP4 `scalar_bridge` and `native_fast_2d` stage outputs against
fixture tensors:

- `gate_output`
- `up_output`
- `intermediate`
- `ffn_output`

The probe reports `max_abs`, `mean_abs`, `rel_l2`, and `cosine` for gate, up,
intermediate, and output, plus per-stage timings for gate, up, intermediate,
down, and full FFN.

## R6000 Command

```bash
bash scripts/r6000_qwen35_9b_dense_ffn_p165_stage_drift_probe.sh
```

Optional overrides:

```bash
FIXTURE_DIR=/root/autodl-tmp/reports/qwen35_9b/p159_dense_ffn_fixtures_20260519_0458 \
P165_JSON=/root/autodl-tmp/reports/qwen35_9b/p165_dense_ffn_stage_drift_$(date +%Y%m%d_%H%M%S).json \
bash scripts/r6000_qwen35_9b_dense_ffn_p165_stage_drift_probe.sh
```

## Notes

Fixtures must have been exported with P159 intermediates.  If `gate_output`,
`up_output`, or `intermediate` is missing, P165 exits instead of producing a
partial stage report.

## R6000 Result

Artifact:

- `reports/qwen35_9b/p165_dense_ffn_stage_drift_20260519_0648_stage_drift.json`

| Backend | Gate max abs / cosine | Up max abs / cosine | Intermediate max abs / cosine | Output max abs / cosine | Total ms | Readout |
|---|---:|---:|---:|---:|---:|---|
| `scalar_bridge` | 0.0625 / 0.999998 | 0.0625 / 0.999998 | 0.25 / 0.999992 | 0.0625 / 0.999990 | 0.5393 | close numerically, too slow |
| `native_fast_2d` | 0.6094 / 0.989131 | 0.6563 / 0.984861 | 5.1094 / 0.975264 | 4.875 / 0.895435 | 0.2012 | faster, numeric contract broken |

`native_fast_2d` starts drifting at the first gate/up projections; the down
stage then amplifies that drift.  This closes the existing packed wrapper as a
9B resident candidate.  The next useful speed path is not a wrapper toggle, but
a dedicated dense FFN kernel or a corrected native-scaled-mm activation/scale
contract that can pass P160/P165 before serving escalation.
