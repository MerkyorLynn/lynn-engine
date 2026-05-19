# Qwen3.6-35B MoE W4A8 Route-Flip Gate

**Date:** 2026-05-19
**Branch:** `qwen/qwen36-35b-w4a8-route-flip-gate-20260519`
**Status:** SCAFFOLD (harness ready; execution requires p133 fixtures + model on R6000)

## Purpose

9B dense W4A8 looks quality-safe (P185/P186 GREEN). 35B A3B MoE is riskier
because W4A8 activation noise can **flip router top-K expert selections**.
If the router picks different experts, model behavior changes fundamentally
-- this is not just numerical drift.

This gate checks whether FP8 fake-quant of `hidden_in` changes which experts
the router selects in Qwen3.6-35B MoE layers. Must pass before any 35B W4A8
resident work begins.

## What It Does

For each p133 MoE fixture:

1. Load `hidden_in` [1, 2048] and ground-truth `expert_ids` [8]
2. Load router weight `mlp.gate.weight` [256, 2048]
3. Apply FP8 fake-quant to `hidden_in`
4. Run router on both original and quantized hidden_in
5. Compare expert selections and routing weights

## Metrics

| Metric | Description |
|--------|-------------|
| `topk_exact` | 1 if expert_ids identical, 0 otherwise |
| `topk_jaccard` | intersection / union of expert sets |
| `route_flip_count` | number of expert slots that differ |
| `routing_weight_max_abs` | max absolute delta on routing weights |
| `routing_weight_cosine` | cosine similarity of routing weight vectors |

## Verdicts

| Verdict | Condition | Action |
|---------|-----------|--------|
| `MOE_W4A8_ROUTE_GREEN` | all fixtures exact, cosine >= 0.9999 | Safe to proceed with 35B W4A8 |
| `MOE_W4A8_ROUTE_AMBER` | some flips but jaccard >= 0.75, cosine >= 0.999 | Investigate; may be acceptable |
| `CLOSED_ROUTE_FLIP` | significant expert changes | Do NOT proceed with 35B W4A8 |

## Files

| File | Purpose |
|------|---------|
| `benchmarks/p189_qwen36_35b_moe_w4a8_route_flip_gate.py` | Main gate logic |
| `scripts/r6000_qwen36_35b_w4a8_route_flip_gate.sh` | R6000 runner |
| `reports/qwen36_35b/QWEN36_35B_W4A8_ROUTE_FLIP_GATE_20260519.md` | This document |

## Prerequisites

1. **p133 MoE fixtures** on R6000:
   ```
   /root/autodl-tmp/reports/qwen36_35b/p133_fixtures_official_w4a16/
   ```
   If missing, run:
   ```bash
   python benchmarks/p133_export_active_moe_fixtures.py \
     --model /root/autodl-tmp/models/Qwen3.6-35B-A3B-lynn-native-w4a16-nvfp4-v0 \
     --layers 0,4,8,16,20,28,32,36,39 \
     --out /root/autodl-tmp/reports/qwen36_35b/p133_moe_fixtures_official_w4a16
   ```

2. **Model weights** on R6000:
   ```
   /root/autodl-tmp/models/Qwen3.6-35B-A3B-lynn-native-w4a16-nvfp4-v0
   ```

## Quick Start (R6000)

```bash
# Full run
bash scripts/r6000_qwen36_35b_w4a8_route_flip_gate.sh

# Custom fixture dir
FIXTURE_DIR=/path/to/fixtures bash scripts/r6000_qwen36_35b_w4a8_route_flip_gate.sh

# Different FP8 format
FP8_FORMAT=e5m2 bash scripts/r6000_qwen36_35b_w4a8_route_flip_gate.sh

# Results
ls -la /root/autodl-tmp/reports/qwen36_35b/p189_moe_w4a8_route_flip_gate_*.json
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ROOT` | `/root/autodl-tmp/lynn-engine` | Engine root |
| `PYTHON_BIN` | `/root/autodl-tmp/conda-envs/r6000-eval/bin/python` | Python |
| `MODEL` | `/root/autodl-tmp/models/Qwen3.6-35B-A3B-lynn-native-w4a16-nvfp4-v0` | Model path |
| `FIXTURE_DIR` | (auto-discovered) | p133 fixture directory |
| `REPORT_DIR` | `/root/autodl-tmp/reports/qwen36_35b` | Report output |
| `FP8_FORMAT` | `e4m3` | FP8 format (e4m3 or e5m2) |
| `GRANULARITY` | `per16` | Quantization granularity |
| `DEVICE` | `cuda` | Torch device |

## JSON Report Schema

```json
{
  "schema": "lynn-qwen36-35b-moe-w4a8-route-flip-gate-v1",
  "verdict": "MOE_W4A8_ROUTE_GREEN | MOE_W4A8_ROUTE_AMBER | CLOSED_ROUTE_FLIP",
  "summary": {
    "total": 18,
    "exact": 18,
    "any_flip": false,
    "jaccard_min": 1.0,
    "jaccard_mean": 1.0,
    "flip_count_total": 0,
    "routing_weight_max_abs": 0.0,
    "routing_weight_cosine_min": 1.0
  },
  "per_layer": { ... },
  "results": [ ... ]
}
```

## Relationship to Other Gates

| Gate | Scope | Status |
|------|-------|--------|
| P185 | 9B dense W4A8 fixture | GREEN (lossless) |
| P186 | 9B dense W4A8 resident | GREEN (exact generation) |
| **P189** | **35B MoE W4A8 route-flip** | **This gate** |
| P133 | MoE fixture export | Done (18 fixtures) |
| P134 | MoE fixture contract | 18/18 GREEN |

## Why Route-Flip Matters

In dense models, W4A8 activation noise only affects output magnitude
(cosine drift). In MoE models, the router computes:

```
router_logits = hidden_in @ gate_weight.T
expert_ids = topk(router_logits, K=8)
```

If FP8 noise changes `router_logits` enough to swap any of the top-8
experts, the model routes tokens to **completely different experts**.
This is a discrete, catastrophic change -- not smooth degradation.

Even a single expert swap across 256 candidates means the W4A8 activation
noise has crossed the router decision boundary.
