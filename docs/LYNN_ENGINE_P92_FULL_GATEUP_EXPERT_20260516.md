# P92 Full Gate/Up Expert Probe

Date: 2026-05-16

P92 expands the P91 row-tile result to a complete gate/up projection for one
active expert:

- one real Lynn 27B routed expert;
- 512 gate rows and 512 up rows;
- full hidden dimension K=2048;
- current Lynn-native per-16 activation/weight scales;
- SM120a blockscaled FP4 MMA using the P89/P90 split16 scale contract.

This is the first proof that the current Lynn-native artifact can drive a full
native-FP4 gate/up expert sub-operator without official/vendor re-quantization.

## Result

Report:

```text
reports/p16_155/p92_sm120a_split16_gateup_full_expert.json
```

| Field | Value |
|---|---:|
| Layer | 28 |
| Expert id | 116 |
| Rows | 512 gate + 512 up |
| Blocks | 64 |
| Median ms | 0.0502 |
| Mean ms | 0.0507 |
| Rows/ms median | 20401.7 |

Accuracy:

| Metric | Value |
|---|---:|
| max_abs_err | `4.77e-07` |
| mean_abs_err | `8.02e-08` |
| rel_l2 | `1.53e-07` |
| tolerance gate | PASS at `1e-5` |

## Decision

P92 passes as a full gate/up expert contract.

The remaining work is no longer "can the artifact feed SM120 native FP4?"
That question is answered. The next work is kernel engineering:

- reduce launch/atomic overhead;
- remove unnecessary global atomics for row ownership where possible;
- wire this as an opt-in gate/up backend;
- compare full active-MoE gate/up+down against the current Triton path;
- only consider runtime promotion after strict full-generate parity.

P93 should either build a production-shaped gate/up backend from this contract
or start the matching down/fused path so active expert FFN can become one
coherent native-FP4 backend.
