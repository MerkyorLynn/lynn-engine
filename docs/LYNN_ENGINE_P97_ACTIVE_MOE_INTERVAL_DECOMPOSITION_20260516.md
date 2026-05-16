# P97 Active MoE Interval Decomposition

Date: 2026-05-16

P97 decomposes active-MoE candidates into CUDA-event intervals:

```text
gate/up interval -> down interval -> total interval
```

This follows P95/P96: down-only native tile was fast, but the first full
composition did not beat the production path. P97 identifies where the speed is
lost.

## Result

Report:

```text
reports/p16_155/p97_sm120a_active_moe_interval_decomposition.json
```

| Variant | Gate/up median | Down median | Total median | Speedup vs baseline |
|---|---:|---:|---:|---:|
| Triton gate/up + Triton down | 0.0562 ms | 0.0328 ms | 0.0891 ms | 1.00x |
| P93 gate/up + Triton down | 0.0582 ms | 0.0266 ms | 0.0848 ms | 1.05x |
| P93 gate/up + native scalar down | 0.0579 ms | 0.0328 ms | 0.0905 ms | 0.98x |
| **P93 gate/up + native_down_tile1** | **0.0575 ms** | **0.0225 ms** | **0.0800 ms** | **1.11x** |

The best candidate is:

```text
p93_gateup_native_down_tile1
```

Candidate accuracy against the quantized-activation active-MoE reference:

| Metric | Value |
|---|---:|
| rel_l2 | `0.00171` |
| cosine | `0.9999986` |

## Important Note

The JSON `contract_pass` field is `false` because the script also judged the
current Triton BF16-activation baseline against the quantized-activation
reference. That is the wrong contract for the baseline and should not be read
as candidate failure.

The candidate variants using P93 quantized gate/up all pass the intended
quantized-activation contract.

## Decision

P97 produces the first full active-MoE composition speed win:

```text
0.0891 ms -> 0.0800 ms = 1.113x
```

Next step:

1. wire `P93 gate/up + native_down_tile1` behind an explicit runtime backend;
2. run full-generate parity, strict tool-call, and no-think loop guards;
3. only promote if those generation gates pass.

