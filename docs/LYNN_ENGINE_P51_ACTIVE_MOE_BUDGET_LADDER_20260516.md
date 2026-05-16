# Lynn Engine P51: active-MoE budget ladder

Date: 2026-05-16
Branch: `codex/p16-r6000-155-tps`

## Goal

P48-P50 showed that local native down kernels can be faster but are not
greedy-exact safe. P51 asks a separate product question:

> Can we get near 155 TPS by spending fewer MoE experts or skipping the shared
> expert, then quality-gating that profile separately?

This is not a default promotion gate. It is a speed/quality budget map.

## Implementation

`engine/moe_packed_nvfp4.py` now supports opt-in research knobs in the fast MoE
path:

```bash
LYNN_MOE_TOPK_LIMIT=1..8
LYNN_MOE_SKIP_SHARED=0|1
LYNN_MOE_TOPK_RENORMALIZE=1
```

The default path remains unchanged.

Benchmark:

```text
reports/p16_155/p51_active_moe_budget_ladder.json
```

## Result

| Candidate | Median TPS | Speedup | Quality sample |
|---|---:|---:|---|
| top8 + shared | 100.52 | 1.000x | coherent |
| top6 + shared | 102.17 | 1.016x | coherent |
| top4 + shared | 105.00 | 1.045x | starts `<think>` pollution |
| top2 + shared | 109.40 | 1.088x | repetition / contamination |
| top1 + shared | 111.37 | 1.108x | broken |
| top8 + skip shared | 115.26 | 1.147x | degraded |
| top6 + skip shared | 116.20 | 1.156x | repetition |
| top4 + skip shared | 121.61 | 1.210x | broken |
| top2 + skip shared | 124.39 | 1.237x | broken |
| top1 + skip shared | 122.69 | 1.221x | broken |

## Decision

This route does not reach 155 TPS and it damages quality before it gets close.

The best-looking reduced-compute point is `top6 + shared`, but it only gains
about **1.6%**. `top4 + shared` gains **4.5%** but already shows visible
format/prompt contamination. Skipping the shared expert produces larger speed
numbers but breaks output quality.

P51 therefore closes the "just compute fewer experts" shortcut. The 155 TPS
route remains:

1. exact-safe orchestration/graph improvements, or
2. a real grouped native-FP4 active expert FFN with explicit quality gates.

No P51 profile should become the production default without a separate V8/V9 /
tool-call / long-context quality gate.
