# Lynn 27B A3B W4A8 v2 Champion and R6000 Runtime Notes - 2026-05-16

Naming note: new handoff/package aliases should use **Lynn 27B A3B** so this
MoE artifact is not confused with dense Qwen3.6 27B-family checkpoints.

## Current Champion

`structured_v10_top6` is the current W4A8 Recovery handoff candidate.

| Candidate | Exact | Min Prefix | Mean Prefix | High-Risk Divergences | Decision |
|---|---:|---:|---:|---:|---|
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

1. Finish `v2` transfer to R6000.
2. Run checksum on R6000.
3. Run loader smoke on `v2`.
4. Run P105 gate/up and full 12-prompt gates on `v2`.
5. Rerun P97 active-MoE interval decomposition on `v2`.
6. Continue A100 Recovery/QAT-lite against the four remaining high-risk
   divergences from `structured_v10_top6`.
