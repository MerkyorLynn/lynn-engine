# P93 Split16 Gate/Up Top-K Backend Probe

Date: 2026-05-16

P93 turns the P92 single-expert gate/up proof into a production-shaped top-k
gate/up backend probe:

- one CUDA launch;
- `grid.x = top_k = 8` active expert slots;
- `grid.y = 64` row tiles per expert;
- each block owns 8 intermediate rows;
- shared-memory accumulation avoids the P92 global atomic pattern;
- output shape is `[top_k, 512]` BF16, ready for the down projection.

This still does not touch runtime defaults. It is a contract and shape probe
for the current Lynn-native NVFP4 artifact.

## Result

Report:

```text
reports/p16_155/p93_sm120a_split16_gateup_topk_backend_probe.json
```

| Field | Value |
|---|---:|
| Layer | 28 |
| Top-k | 8 |
| Expert ids | 12, 49, 63, 116, 159, 192, 204, 155 |
| Native median | 0.0602 ms |
| Triton median | 0.0565 ms |
| Native / Triton median | 0.938x |
| Reference build | quantized-activation CPU reference |

Accuracy against the quantized-activation reference:

| Metric | Value |
|---|---:|
| max_abs | `0.00683` |
| mean_abs | `0.000171` |
| rel_l2 | `0.00167` |
| cosine | `0.9999986` |
| contract gate | PASS |

Observed difference against the current Triton BF16-activation path:

| Metric | Value |
|---|---:|
| max_abs | `0.242` |
| rel_l2 | `0.116` |
| cosine | `0.9933` |

That BF16-activation delta is expected: P93 uses FP4-quantized activations so
the comparison is not an exact same-math parity target. The authoritative
contract for P93 is the quantized-activation reference.

## Decision

P93 passes as a production-shaped top-k gate/up backend contract.

It is **not promoted** because the first implementation is still slightly
slower than the current Triton gate/up path. The win is architectural rather
than immediate runtime speed: the current Lynn-native artifact can now feed a
single-launch top-k gate/up backend without official/vendor re-quantization.

Next gate:

1. compose P93 gate/up with the active down projection;
2. compare full active-MoE output against the quantized-activation reference;
3. measure whether down fusion or a non-atomic schedule can recover the speed;
4. run strict full-generate parity before any runtime promotion.

