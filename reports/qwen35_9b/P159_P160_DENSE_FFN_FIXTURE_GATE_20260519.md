# Qwen3.5-9B Dense FFN Fixture Gate

**Date:** 2026-05-19  
**Model:** official Qwen3.5-9B Lynn-native W4A16 NVFP4  
**Purpose:** create a fast, strict target for dense-FFN kernel work.

## Why This Exists

P155 showed the 9B dense FFN costs about **7.37 ms/token** in the default
manual decode profile, with gate/up/down projections dominating that block.
The 35B MoE line already has p133/p134 fixture contracts; the 9B dense line now
gets the equivalent admission gate before anyone writes or promotes a fused FFN
kernel.

## New Artifacts

| Artifact | Role |
|---|---|
| `benchmarks/p159_qwen35_9b_dense_ffn_fixture_export.py` | Exports prompt-derived dense FFN inputs and reference outputs. |
| `benchmarks/p160_qwen35_9b_dense_ffn_fixture_contract.py` | Recomputes dense FFN outputs and validates exactness/timing. |
| `scripts/r6000_qwen35_9b_dense_ffn_fixture_gate.sh` | R6000 one-command p159 -> p160 gate. |

## R6000 Command

```bash
cd /root/autodl-tmp/lynn-engine
bash scripts/r6000_qwen35_9b_dense_ffn_fixture_gate.sh
```

Optional:

```bash
LAYERS=0,4,8,12,16,20,-1 EXPORT_WEIGHTS=1 bash scripts/r6000_qwen35_9b_dense_ffn_fixture_gate.sh
```

`EXPORT_WEIGHTS=1` writes per-layer dense FFN weight shards for standalone
kernel development. The default avoids duplicating large weights and lets p160
reload only the needed layers from the model artifact.

## R6000 Validation

Artifact:
`reports/qwen35_9b/p160_dense_ffn_fixture_contract_20260519_0445_fixed.json`

| Metric | Value |
|---|---:|
| Fixtures | 8 |
| Layers | 0 / 8 / 16 / 31 |
| Passed | 8 / 8 |
| Exact | 8 / 8 |
| max_abs_max | 0 |
| cosine_min | 1.0 |
| dense FFN ref_ms_mean | 0.21585 ms |
| Decision | `DENSE_FFN_FIXTURE_GREEN` |

This gives the 9B dense line a strict fixture target before any fused
gate/up/down kernel enters serving gates.

## Promotion Rule

Any 9B dense FFN native/repacked candidate must first pass p160 against the
p159 fixtures. Only then should it move to the 9B P37/P25/structured serving
gates.

Exact tensor equality is treated as cosine `1.0` in p160 so the gate is not
falsely failed by FP32 cosine reduction precision when `max_abs == 0`.
