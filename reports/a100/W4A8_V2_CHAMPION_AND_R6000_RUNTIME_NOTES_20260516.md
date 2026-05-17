# Lynn 27B A3B W4A8 v2 Champion and R6000 Runtime Notes - 2026-05-16

Naming note: new handoff/package aliases should use **Lynn 27B A3B** so this
MoE artifact is not confused with dense Qwen3.6 27B-family checkpoints.

## Current Champion

`structured_v10_top6` is the current W4A8 Recovery handoff candidate.

2026-05-17 update: `structured_v16_top6_damped075` is now the best exact-count
research candidate, but it is still RED and does not replace the production
promotion bar.

| Candidate | Exact | Min Prefix | Mean Prefix | High-Risk Divergences | Decision |
|---|---:|---:|---:|---:|---|
| structured_v16_top6_damped075 | **9/12** | 12 | **35.92** | 3 | best exact-count research candidate, still RED |
| structured_v17_top6_damped050 | 6/12 | 7 | 29.58 | 6 | damping too weak, regression |
| structured_v19_prefix_margin_top6_expert | 6/12 | 7 | 31.50 | 6 | local prefix-margin GREEN, generation regression |
| structured_v20_v16_plus_prefix025 | 7/12 | 7 | 30.67 | 5 | v16 + small prefix alpha regresses |
| structured_v21_v16_plus_prefix050 | 5/12 | 1 | 23.75 | 7 | v16 + larger prefix alpha severe regression |
| structured_v22_guarded12_prefix_margin_expert | 5/12 | 1 | 23.75 | 7 | conservative guarded12 local AMBER, generation regression |
| **structured_v10_top6** | **8/12** | **13** | 33.33 | 4 | current v2 handoff |
| structured_v9_top5 | 8/12 | 7 | **34.25** | 4 | previous v1 handoff |
| structured_v11_top4_alt30 | 7/12 | 12 | 32.42 | 5 | not promoted |
| structured_v13_top7_add17 | 7/12 | 7 | 36.08 | 5 | improves some long prefixes, hurts exact |
| structured_v14_top6_add17_no26 | 7/12 | 7 | 31.83 | 5 | not promoted |
| structured_v15_top6_add18_no26 | 5/12 | 1 | 27.25 | 7 | regression |

`structured_v10_top6` is still **not production GREEN**. It is the best
runtime-research package because it matches v9 on exact count and improves the
worst-case prefix from 7 to 13. The remaining failures are still high-risk
tool/JSON/format prompts, so production promotion still requires more Recovery
or QAT-lite.

## v2 NVFP4 Package

Source folded model:

```text
/mnt/data2/lynn-a100/models/lynn-27b-variable-recovery-step5000-bf16-w4a8-alpha-overlay-structured_v10_top6
```

NVFP4 package:

```text
/mnt/data2/lynn-a100/nvfp4/lynn-27b-w4a8-structured-v10-top6-nvfp4-native
```

Symlink:

```text
/mnt/data2/lynn-a100/nvfp4/lynn-27b-w4a8-nvfp4-v2
/mnt/data2/lynn-a100/nvfp4/lynn-27b-a3b-w4a8-nvfp4-v2
```

Package properties:

| Field | Value |
|---|---:|
| shards | 7 |
| quantized tensors | 542 |
| kept tensors | 484 |
| size | ~20 GiB |
| checksum | SHA256 pass on A100 |

Transfer target:

```text
/root/autodl-tmp/models/lynn-27b-w4a8-nvfp4-v2
/root/autodl-tmp/models/lynn-27b-a3b-w4a8-nvfp4-v2
```

`v1` partial on R6000 is intentionally left in place as a resumable fallback,
but the active transfer bandwidth is now assigned to `v2`.

## R6000 v0 Runtime Signal

The 96-token P105 generation gate confirms the two-stage runtime plan:

| Mode | Exact | Min Prefix | Mean Prefix | Mean Decode TPS |
|---|---:|---:|---:|---:|
| gate/up W4A8 | **10/12** | **12** | **48.25** | 21.22 |
| full active W4A8 | 8/12 | 5 | 43.92 | 19.90 |

Interpretation:

- Gate/up W4A8 remains the safer first runtime bridge.
- Full active-MoE W4A8 still introduces early drift and should wait for
  stronger Recovery or QAT-lite.
- JSON/tool-call exactness remains the hard production gate, even when ordinary
  chat quality looks acceptable.

## R6000 v2 P97 Runtime Signal

Serial P97 five-layer rerun on the `v2` R6000 package:

```text
reports/a100/r6000_p97_v2_multilayer_summary_20260517_092812.json
```

| Field | Value |
|---|---:|
| layers | 4, 12, 20, 28, 36 |
| all contract pass | true |
| best variant | p93_gateup_native_down_tile1 |
| speedup min | 1.1001x |
| speedup mean | 1.1177x |
| speedup max | 1.1263x |
| gate median mean | 0.05777 ms |
| down median mean | 0.02253 ms |

Interpretation: `native_down_tile1` gives a real local active-MoE interval
micro gain, but it is still only around ~1.12x for this interval and does not
close the 155 TPS gap by itself. Gate/up remains the dominant budget, so the
next runtime step should target gate/up scheduling/fusion/graph work.

## Structured Format Guard Signal

The structured-anchor gate separates prompt cleanliness and W4A8 parity from
raw first-token drift. On `structured_v16_top6_damped075`, the unguarded anchor
gate remains RED, while generic forced format prefixes turn the 6-prompt mini
gate GREEN:

```text
reports/a100/a100_w4a8_format_anchor_gate_structured_v16_guard_v3_6prompt_32tok.json
```

| Gate | Exact | Min Prefix | Mean Prefix | Reference Format | Candidate Format | Raw Prefix Match | Decision |
|---|---:|---:|---:|---:|---:|---:|---|
| no guard | 2/6 | 0 | 11.17 | 1/6 | 1/6 | 6/6 | RED |
| guard v3 | 6/6 | 22 | 27.00 | 6/6 | 6/6 | 0/6 | GREEN |

Interpretation: the current high-ROI path is first-token/few-token structured
domain correction plus Prefix-Margin Recovery. This is a serving/recovery
research bridge, not permission to promote full-active W4A8 by default.

The first Prefix-Margin Recovery implementation (`structured_v19`) confirms
that expert alpha can reduce local first-8-token structured drift from 4.03% to
2.76%, but it overfits the guarded format-anchor domain and regresses the
ungarded 12-prompt generation gate to 6/12. Keep v16 as the better general
Recovery baseline and use v19 only as a local repair signal.

2026-05-17 follow-up: v20/v21 blended the v19 prefix alpha back into v16 at
0.25/0.50 and both regressed. v22 trained a more conservative expert alpha on
the original 12 prompts with serving guard specs and improved local drift from
3.92% to 3.15%, but still regressed generation to 5/12. Stop sweeping post-hoc
prefix alpha overlays for now; the signal needs a broader teacher-cleanup or
QAT objective.

The useful serving-side result is `serving_guard12` on v16:

```text
reports/a100/a100_w4a8_serving_guard12_gate_structured_v16_12prompt_48tok.json
```

It reaches 8/12 served-text exact with 12/12 format-clean outputs after
balanced JSON/code/bullet stops. That makes format guard a practical structured
mode fallback, but not a full generation parity fix.

## Whole-Decode Graph Slot Boundary

`LYNN_FULL_TOKEN_GRAPH_SLOT=1` triggers the existing current-position
whole-decode CUDA graph body covering all 40 layers plus `lm_head`. It also
requires `LYNN_ROUTER_TOPK_SORTED=1`, unless the diagnostic-only
`LYNN_ALLOW_UNSORTED_FULL_TOKEN_GRAPH_SLOT=1` override is set.

The current R6000 fast MoE path conflicts with that contract:
`LYNN_MOE_FAST_FIXED=1` requires `LYNN_ROUTER_TOPK_SORTED=0`. With
`LYNN_MOE_FAST_FIXED=0`, the v2 package passes a short strict-slot parity gate,
but capture-per-token is still a runtime loss:

| Metric | Value |
|---|---:|
| parity | true |
| avg capture | 81.31 ms/token |
| avg replay | 9.60 ms/token |
| replay-only TPS | 104.15 tok/s |
| graph-slot decode TPS | 10.99 tok/s |

Conclusion: keep whole-decode graph as a correctness primitive. It needs a
reusable current-position slot lifecycle and a graph-safe MoE route before it
is a 155 TPS lever.

## P93 Gate/Up Split16 Sweep

Five-layer P93 sweep on R6000 v0:

| Layer | Native Median ms | Triton Median ms | Native/Triton Speed |
|---:|---:|---:|---:|
| 4 | 0.05755 | 0.05626 | 0.977x |
| 12 | 0.06403 | 0.05664 | 0.885x |
| 20 | 0.06270 | 0.05518 | 0.880x |
| 28 | 0.06347 | 0.05680 | 0.895x |
| 36 | 0.06290 | 0.05606 | 0.891x |

All layers pass the quantized-activation numerical contract, but native split16
gate/up is slower than Triton in isolation. Do **not** promote P93 gate/up as a
standalone runtime replacement. It remains useful only as a building block for a
larger fused active-MoE path that can amortize scheduling and intermediate
traffic.

## Next Actions

1. Keep full-active W4A8 out of default promotion until structured/tool-call
   generation gates move to AMBER/GREEN.
2. Use `structured_v16_top6_damped075` as the next A100 Recovery research
   baseline.
3. Build Prefix-Margin Recovery around the first 1-8 decode tokens for JSON,
   code, YAML, tool-call, and bullet-format prompts.
4. Convert format-start guard into a serving-side option for structured modes,
   ideally with balanced JSON stop for object-only outputs.
5. Continue R6000 runtime work on gate/up scheduling/fusion/graph capture; P97
   v2 down-path micro gain alone is not enough for 155 TPS.
