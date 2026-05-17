# R6000 Active-MoE Profile for W4A8 v2 - 2026-05-17

Report:

```text
reports/a100/r6000_p23_active_moe_profile_v2_20260517_120647.json
```

## Result

The profile runs all 40 layers against the R6000 v2 package with the safe
Config D-style runtime:

```text
LYNN_MOE_FAST_FIXED=1
LYNN_NATIVE_DOWN_BACKEND=triton
LYNN_ROUTER_TOPK_SORTED=0
```

Total measured MoE time across 40 layers:

| Component | Total ms/token | Share of full MoE |
|---|---:|---:|
| Router/top-k/softmax | 1.908 | 23.5% |
| Gate/up | 1.403 | 17.3% |
| Down | 1.297 | 16.0% |
| Active routed experts end-to-end | 2.582 | 31.8% |
| Shared expert | 2.286 | 28.2% |
| Full MoE path | 8.120 | 100.0% |
| Residual composition/dispatch | 1.344 | 16.5% |

Layer timing is nearly flat: the slowest layers are all about
`0.203-0.205 ms` for the full MoE path. This is not a single bad layer problem.

## Interpretation

The R6000 155 TPS target needs roughly this step-time movement:

```text
current safe server decode: ~100.45 tok/s -> ~9.95 ms/token
target:                    155 tok/s    -> ~6.45 ms/token
gap:                                      ~3.50 ms/token
```

Active routed experts alone are not enough. Even a perfect active-expert
replacement only accounts for about `2.58 ms/token`; the remaining gap also
requires router/top-k, shared expert, and Python/composition overhead work.

This explains why the `cudaTileDown` service sweep reached `~108.9 tok/s` but
was not promotable: it improves one component, while the full decode stack still
needs broader correctness-preserving changes.

## Next Speed Work

The next runtime milestones should be:

1. Keep Config D as the serving fallback.
2. Treat MTP as a required multiplier, not a cosmetic add-on.
3. Start a native active-MoE boundary that owns router output, gate/up, down,
   and shared expert accounting together, instead of only swapping the down
   projection.
4. If doing a C++/CUDA refactor, target a single per-layer active-MoE call that
   can later sit behind a reusable decode loop; optimizing one down kernel is
   confirmed insufficient.

The useful engineering target is no longer just "faster active experts"; it is
"reduce the whole 8.12 ms/token MoE budget while preserving structured gates."

## Budget Ladder Check

Follow-up one-run service-budget probe:

```text
reports/a100/r6000_moe_budget_one_runner_v2_20260517_121109.json
```

| Candidate | Median TPS | Speedup | Exact vs baseline | Min prefix |
|---|---:|---:|---:|---:|
| baseline_top8_shared | 100.77 | 1.000x | 3/3 | 96 |
| top6_shared | 100.95 | 1.002x | 1/3 | 10 |
| top4_shared | 101.98 | 1.012x | 0/3 | 1 |
| top2_shared | 102.98 | 1.022x | 0/3 | 1 |
| top1_shared | 103.04 | 1.023x | 0/3 | 1 |
| top8_skip_shared | 105.26 | 1.045x | 0/3 | 1 |
| top1_skip_shared | 107.90 | 1.071x | 0/3 | 1 |

This closes the "just reduce active experts" shortcut. Top-k reduction gives
almost no speed and quickly diverges. Skipping shared expert gives a larger
speed signal, but all tested outputs diverge near the first token. The 155 TPS
route should therefore keep exact top-k/shared semantics and focus on real
kernel/runtime replacement plus MTP, not approximation by dropping experts.

## P97 v2 Layer-28 Rerun

Follow-up serial P97 report:

```text
reports/p16_155/p97_r6000_v2_layer28_serial_20260517_121948.json
```

| Variant | Gate ms | Down ms | Total ms | Speedup | Contract |
|---|---:|---:|---:|---:|---|
| triton_gateup_triton_down | 0.05589 | 0.03174 | 0.08774 | 1.000x | timing ref |
| p93_gateup_triton_down | 0.05838 | 0.02662 | 0.08501 | 1.032x | pass |
| p93_gateup_native_down_scalar | 0.05776 | 0.03174 | 0.08925 | 0.983x | pass |
| p93_gateup_native_down_tile1 | 0.05774 | 0.02253 | 0.08022 | 1.094x | pass |

This confirms the v2 local active-MoE micro-gain is real, but it is not large
enough to explain a server jump from ~100 TPS to 155 TPS. R6000 is now running a
serial 5-layer P97 summary so the next kernel decision is based on layer spread,
not a single favorable layer.

## P97 v2 5-Layer Summary

Follow-up summary:

```text
reports/p16_155/p97_r6000_v2_5layer_summary_local_20260517_122158.json
```

| Metric | Value |
|---|---:|
| Layers | 4, 12, 20, 28, 36 |
| Contract pass | 5/5 |
| Best variant set | p93_gateup_native_down_tile1 |
| Speedup min | 1.094x |
| Speedup mean | 1.105x |
| Speedup max | 1.120x |
| Best gate median mean | 0.05797 ms |
| Best down median mean | 0.02253 ms |

Decision: the native-down tile path is consistently the local winner, but the
gate/up interval is now the larger remaining active-MoE cost. This shifts the
next speed work from "prove native down" to "reduce or fuse gate/up scheduling
without triggering split16 activation-quant drift in generation."

## Split16 + Tile1 Generate Gate

Follow-up service-shaped gate:

```text
reports/p16_155/p37_r6000_v2_split16_tile1_generate_gate_20260517_122904.json
```

| Mode | Median decode TPS | Exact IDs |
|---|---:|---:|
| Config D baseline | 101.13 | reference |
| split16 gate/up + native down tile1 | 25.15 | 0/3 |

The local P97 winner is therefore not a promotable serving switch. It requires
graph-off activation quantization and changes greedy IDs. Treat split16 as an
offline kernel research artifact, not the next runtime bridge.
