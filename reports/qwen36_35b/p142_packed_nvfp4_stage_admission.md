# P142 · Packed NVFP4 Stage Admission Gate

**Generated:** 2026-05-18T23:27:09

## Admission Decision

**Verdict: 🟡 DEFAULT_BLOCKED**

> p141 verdict AMBER_STAGE — fixture-stage AMBER prevents default/resident promote

| Flag | Value |
|------|-------|
| default_promote_blocked | ✅ |
| p37_exploratory_allowed | ✅ |

## Inputs

### p141 Stage Diagnostics (V2)

| Field | Value |
|-------|-------|
| path | `p141_v2_report.json` |
| verdict | 🟡 AMBER_STAGE |
| max_max_abs | 0.001953125 |
| avg_latency_ms | 0.0517 |
| min_cosine | 0.9999890924 |

### p140 Packed NVFP4 Probe

| Field | Value |
|-------|-------|
| path | `p140_packed_nvfp4_probe_report.json` |
| verdict | 🔴 CLOSED |
| max_max_abs | 0.00390625 |
| avg_latency_ms | 0.0903 |
| min_cosine | 0.9999736547 |

### Candidate Summary

| Field | Value |
|-------|-------|
| path | `native_moe_candidate_summary.json` |
| best_verdict | AMBER_FAST_PRETRANSPOSED |
| has_default_candidate | ❌ |
| has_amber_candidate | ✅ |

## Decision Matrix

| p141 verdict | packed verdict | → Admission |
|-------------|---------------|-------------|
| AMBER_STAGE within P37 bounds | any | 🟡 DEFAULT_BLOCKED + P37_ALLOWED |
| AMBER_STAGE outside P37 bounds | any | 🟡 DEFAULT_BLOCKED |
| GREEN | CLOSED | 🟡 DEFAULT_BLOCKED |
| GREEN | GREEN/AMBER (within P37 bounds) | 🔵 P37_ALLOWED |
| any | hard limit exceeded | 🔴 CLOSED |
