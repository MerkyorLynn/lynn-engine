# P143 · Resident P37 Admission Gate

**Generated:** 2026-05-19T00:08:03

## Admission Decision

**Verdict: ⏳ WAITING_FOR_P37_REPORT**

> P37 report not found

| Flag | Value |
|------|-------|
| default_promote_allowed | ❌ (never allowed) |

## Inputs

### Stage Report (p142 graphsafe)

| Field | Value |
|-------|-------|
| path | `p142_graphsafe_v31_fixture_report.json` |
| candidate | moe_packed_pretransposed_graphsafe_v3 |
| verdict | 🟡 AMBER_GRAPHSAFE |
| max_max_abs | 0.001953125 |
| avg_latency_ms | 0.0440 |
| min_cosine | 0.9999876022 |

### P37 Report

| Field | Value |
|-------|-------|
| path | ⚪ NOT FOUND |

## Decision Matrix

| Stage verdict | P37 state | → Admission |
|---------------|-----------|-------------|
| not AMBER_GRAPHSAFE/DEFAULT_STAGE | — | 🔴 CLOSED_STAGE_BLOCK |
| ok, but max_abs/latency exceeded | — | 🔴 CLOSED_STAGE_BLOCK |
| ok | missing | ⏳ WAITING_FOR_P37_REPORT |
| ok | collapse=true | 🔴 CLOSED_GRAPH_COLLAPSE |
| ok | exact=true OR passed=total | 🟢 P25_ALLOWED |
| ok | exact=false, no collapse | 🔴 CLOSED_P37_DRIFT |

> **Note:** DEFAULT_PROMOTE is never output. Maximum admission is P25_ALLOWED.
