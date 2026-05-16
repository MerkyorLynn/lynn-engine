# P96 Native-Down Composition Probe

Date: 2026-05-16

P96 composes the best P95 down backend with the P93 native gate/up backend:

```text
quantized hidden
  -> P93 split16 native-FP4 top-k gate/up
  -> inter[top_k, 512]
  -> native_down_tile1
  -> active MoE output[2048]
```

The goal is to check whether the local 2.14x down win from P95 survives when it
is used in the full active expert composition.

## Result

Report:

```text
reports/p16_155/p96_sm120a_split16_native_down_composition_probe.json
```

| Field | Value |
|---|---:|
| Layer | 28 |
| Top-k | 8 |
| Native composed median | 0.0830 ms |
| Current Triton active median | 0.0810 ms |
| Native / Triton median | 0.976x |

Accuracy against the quantized-activation active-MoE reference:

| Metric | Value |
|---|---:|
| max_abs | `0.000243` |
| mean_abs | `0.0000203` |
| rel_l2 | `0.00171` |
| cosine | `0.9999986` |
| contract gate | PASS |

## Decision

P96 passes numerically but is **not promoted**.

The result is instructive: P95 proved that native_down_tile1 is much faster
than Triton down in isolation, but the full composition still loses slightly
because the current P93 gate/up backend and two-stage scheduling overhead absorb
that gain.

Next work should not be another down-only tweak. The likely paths are:

1. reduce P93 gate/up overhead;
2. fuse gate/up and down scheduling to remove intermediate/launch overhead;
3. build a persistent grouped active expert kernel;
4. keep full-generate parity as the promotion gate.

