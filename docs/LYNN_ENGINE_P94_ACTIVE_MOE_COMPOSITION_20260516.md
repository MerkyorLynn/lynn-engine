# P94 Active MoE Composition Probe

Date: 2026-05-16

P94 composes the P93 production-shaped native gate/up backend with the current
packed down weighted-sum path:

```text
quantized hidden
  -> P93 split16 native-FP4 top-k gate/up
  -> inter[top_k, 512]
  -> packed down weighted-sum
  -> active MoE output[2048]
```

This is deliberately a two-stage composition probe. It proves the full active
MoE numerical contract before attempting a fused or persistent grouped kernel.

## Result

Report:

```text
reports/p16_155/p94_sm120a_split16_active_moe_composition_probe.json
```

| Field | Value |
|---|---:|
| Layer | 28 |
| Top-k | 8 |
| Expert ids | 12, 49, 63, 116, 159, 192, 204, 155 |
| Native composed median | 0.0823 ms |
| Current Triton median | 0.0818 ms |
| Native / Triton median | 0.994x |

Accuracy against the quantized-activation active-MoE reference:

| Metric | Value |
|---|---:|
| max_abs | `0.000243` |
| mean_abs | `0.0000203` |
| rel_l2 | `0.00171` |
| cosine | `0.9999986` |
| contract gate | PASS |

Observed difference against the current Triton BF16-activation active path:

| Metric | Value |
|---|---:|
| max_abs | `0.00696` |
| rel_l2 | `0.111` |
| cosine | `0.9938` |

That BF16-activation delta is expected: P94 uses FP4-quantized activations.
The authoritative contract is the quantized-activation reference.

## Decision

P94 passes as a full active-MoE composition contract.

The current two-stage implementation is almost tied with the production Triton
path, so it is still **not promoted**. The important result is that the active
expert path is now numerically proven end to end on the current Lynn-native
artifact:

- top-k gate/up backend: proven by P93;
- down composition: proven by P94;
- no official/vendor re-quantization required for the R6000 path.

Next work should focus on speed rather than correctness:

1. fuse gate/up and down scheduling to reduce launch/intermediate overhead;
2. use non-atomic ownership where possible;
3. evaluate persistent grouped expert kernels;
4. only promote after strict full-generate parity and server TPS gates.

