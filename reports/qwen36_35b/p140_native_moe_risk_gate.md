# P140 Native MoE Candidate Risk Gate — 🟡 AMBER

**Generated:** 2026-05-18T22:28:08

## Verdict

| Field | Value |
|-------|-------|
| Tier | **AMBER** |
| Recommend P37 exploratory | **True** |

## Reasons

- slot_max_abs 2.929688e-03 > 0.001
- cosine_min 0.9999796748 < 0.999999

## Annotations

- NO default promote — P37 exploratory only
- slot_max_abs=2.929688e-03 exceeds DEFAULT threshold 0.001
- unique_max_abs=1.953125e-03 (ref, not blocking)
- strict cuBLAS oracle latency=0.4667ms (too slow for serving)

## Input Metrics

| Source | Metric | Value |
|--------|--------|-------|
| p136 contract | verdict | GREEN (18/18) |
| p136 contract | max_abs_max | 0.0 |
| candidate (native_slot_output_owned_bf16) | slot_max_abs | 2.929688e-03 |
| candidate | unique_max_abs | 1.953125e-03 |
| candidate | cosine_min | 0.9999796748 |
| candidate | avg_latency_ms | 0.0523 ms |
| strict oracle | avg_latency_ms | 0.4667 ms |
| strict oracle | all_exact | True |
| p137 diagnostics | native_full_vs_torch_slot_max_abs | 2.929688e-03 |
| p137 diagnostics | native_full_ms_mean | 0.0517 ms |

## Thresholds

| Tier | slot_max_abs | cosine_min | unique_max_abs | latency_ms |
|------|-------------|------------|----------------|------------|
| DEFAULT | ≤0.001 | ≥0.999999 | — | ≤0.059 |
| AMBER | ≤0.003 | — | ≤0.002 | ≤0.055 |
