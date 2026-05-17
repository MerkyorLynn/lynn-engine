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

## Full-Token Diagnostic Profile

Follow-up diagnostic profile:

```text
reports/p16_155/p6_full_token_profile_r6000_v2_configd_20260517_123301.json
```

This profile uses the older `_decode_layer` benchmark path, so its absolute
`27.33 TPS` estimate is not the service number. The relative layer shape is
still useful:

| Type | Count | Avg ms/layer | Sum ms |
|---|---:|---:|---:|
| linear_attention | 30 | 0.905 | 27.154 |
| full_attention | 10 | 0.791 | 7.908 |

The next R6000 profile is therefore layer-34 linear-attention segmentation, not
another active-MoE approximation.

## Linear-Attention Segment Profiles

Follow-up reports:

```text
reports/p16_155/p10c_linear_attn_layer0_r6000_v2_configd_20260517_123614.json
reports/p16_155/p10c_linear_attn_layer24_r6000_v2_configd_20260517_123614.json
reports/p16_155/p10c_linear_attn_layer34_r6000_v2_configd_20260517_123509.json
```

| Layer | Full core ms | Fused in-proj | Recurrent | Conv | Split | Norm | Out |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.341 | 0.078 | 0.036 | 0.032 | 0.026 | 0.019 | 0.015 |
| 24 | 0.333 | 0.075 | 0.036 | 0.033 | 0.026 | 0.020 | 0.014 |
| 34 | 0.330 | 0.079 | 0.036 | 0.033 | 0.026 | 0.020 | 0.016 |

The segment shape is stable across early/mid/late linear-attention layers. The
largest individual linear-attention target is the fused native-FP4 in-proj, but
the whole linear core is only about `0.33-0.34 ms`; per-layer MoE/router/shared
and Python/runtime scheduling still need to be part of the 155 TPS plan.

## Config D Service Anchor

Follow-up service probe:

```text
reports/p16_155/r6000_p25_server_decode_sweep_20260517_123947.json
reports/p16_155/r6000_p25_server_decode_sweep_20260517_124155.json
```

| Max Tokens | Decode TPS |
|---:|---:|
| 128 | 99.42 |
| 256 | 100.01 |
| 512 | 100.35 |
| 1024 | 98.85 |

This reconfirms the safe serving line is still ~100 TPS. The preview is normal
text, not the rejected exclamation-loop failure mode.

The 1024-token run also reports wall `88.70 tok/s`, median decode step
`9.97 ms`, linear block graph reuse enabled, and native FP4 lm_head enabled.
Longer generation does not expose a hidden throughput jump; the remaining 155
TPS gap still requires a runtime multiplier, most likely MTP plus a native
decode-loop boundary.

## Down Backend Service Sweep

Service-path report:

```text
reports/p16_155/r6000_p25_server_decode_sweep_20260517_124605.json
reports/p16_155/r6000_server_decode_sweep_summary_20260517_124605.json
```

| Config | 256 decode TPS | 512 decode TPS | Preview Gate |
|---|---:|---:|---|
| configD / triton down | 99.78 | 99.05 | pass |
| cudaTileDown | 108.17 | 108.84 | fail: exclamation loop |

The CUDA tile down backend has a real service-speed signal (`1.08-1.10x`), but
it is not usable: both sampled generations collapse into an exclamation loop.
This rules out down-backend swapping as an immediate promotion path. The useful
next step is a numerical/semantic gate for the CUDA tile kernel itself, not a
larger server benchmark.

Follow-up first-divergence probes:

```text
reports/p16_155/p50_down_tile_first_divergence_r6000_v2_service_20260517_124952.json
reports/p16_155/p50_down_tile_first_divergence_r6000_v2_p25prompt_20260517_125245.json
```

The short diagnostic prompt stays top-1 matched for 8 steps, but hidden drift is
visible from step 2/layer 21 and logits rel_l2 reaches roughly `0.10`. The P25
service prompt diverges at step 28: Triton chooses `主流`, while cuda_tile
chooses `核心`; the Triton margin is only `0.4297`, and the cuda_tile margin is
`0.0547`. The first layer below the hidden cosine threshold appears at
step 0/layer 11.

This explains the service failure without contradicting the local speed win:
the down tile kernel is numerically close enough for local contracts, but its
accumulation-order drift survives through the decode stack and flips low-margin
semantic tokens. Promotion requires a drift-reduced kernel or a full decode
quality gate, not just faster down projection timing.
