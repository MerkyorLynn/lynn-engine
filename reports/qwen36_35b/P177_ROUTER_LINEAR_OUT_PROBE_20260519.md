# Qwen3.6 35B W4A16 P177 Router Linear Out-Buffer Probe

Date: 2026-05-19

## Purpose

P163 proved the router top-k out-buffer is exact and saves a small but stable
boundary cost. P177 tests the preceding router projection:

```text
reference: F.linear(hidden, gate_weight)
candidate: torch.mm(hidden, gate_weight.t(), out=preallocated_logits)
```

This is a fixture-only probe. It does not touch resident serving defaults.

## Admission Rule

Only consider a resident opt-in if all P138 fixtures are bit-exact:

- expert ids match;
- route weights match exactly;
- router logits max_abs is zero;
- mean full-router timing improves by at least 0.001 ms/layer.

Even if accepted, this is a small MoE boundary cleanup and must pass the normal
P37/P25/structured gate before any promotion.

## R6000 Fixture Result

Artifact:
`reports/qwen36_35b/p177_qwen36_router_linear_out_probe_20260519_084632.json`

| Metric | Result |
|---|---:|
| exact fixtures | 18/18 |
| logits max_abs | 0 |
| route max_abs | 0 |
| linear projection | 0.00983 -> 0.00752 ms |
| full router boundary | 0.04434 -> 0.03896 ms |
| router delta | -0.00538 ms/layer |

Decision: `ROUTER_LINEAR_OUT_CANDIDATE`.

## Resident Candidate

Opt-in env:

```text
LYNN_ROUTER_LINEAR_OUT_BUFFER=1
```

The resident runner attaches `mlp.gate.weight_t` and
`mlp.gate._logits_scratch` at load time. The MoE decode path uses
`torch.mm(..., out=...)` only when those tensors are present and the decode
shape is `[1, hidden]`; otherwise it falls back to `F.linear`.

## R6000 Promotion Gate

Artifacts:

- `reports/qwen36_35b/r6000_qwen36_w4a16_router_linear_out_20260519_084930_p37.json`
- `reports/qwen36_35b/r6000_qwen36_w4a16_router_linear_out_20260519_084930_p25.json`
- `reports/qwen36_35b/r6000_qwen36_w4a16_router_linear_out_20260519_084930_hard_structured.json`
- `reports/qwen36_35b/r6000_qwen36_w4a16_router_linear_out_20260519_084930_promotion_summary.json`

| Gate | Result |
|---|---:|
| P37 exact | true |
| P37 median speedup | 0.9954x |
| P25 512 decode TPS | 106.77 |
| hard structured | 40/40 |
| hard structured mean decode TPS | 106.51 |

Decision: `CLOSED`. The fixture win is real, but the extra resident memory
pressure from the transposed router weights and logits scratch does not improve
service TPS. Keep `LYNN_ROUTER_LINEAR_OUT_BUFFER=1` as an opt-in diagnostic
only.
