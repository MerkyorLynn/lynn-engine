# Stage 6 Phase 3-B — selected-layer prefill gate runbook

Date: 2026-06-04

Verdict: **RUNBOOK ONLY; no P3-B result is banked yet.**

P3-B is the first multi-layer gate after the P3-A grouped active-MoE contract
probe. Its purpose is to verify that the P3-A contract can be composed into a
selected prefill stack without rebuilding active BF16 expert shadows.

## Required Predecessors

Do not run or report P3-B as a bankable gate unless these artifacts exist and
are PASS:

- P2-O `basic` report from `scripts/write_stage6_p2o_report.py`;
- P2-O `rc-mini` report from `scripts/write_stage6_p2o_report.py`;
- P3-A report from `scripts/write_stage6_p3a_report.py`;
- suite-level report from `scripts/write_stage6_gpu_gate_suite_report.py`.

If any predecessor is missing or failed, P3-B work may continue as local
engineering, but README/release notes must still say P3-B is pending.

## Scope

P3-B must test selected real layers, not only a single isolated active-MoE
function:

- selected layers: start with `0-3`; widen to `0-7` only after the first pass;
- input: deterministic synthetic hidden or a frozen prompt hidden trace;
- path: P3-A active-MoE contract for active experts, existing BF16 shared expert
  and router unless a separate P3-B contract explicitly replaces them;
- comparison: BF16 or current P2N/P2E selected-layer reference;
- forbidden: active BF16 expert shadow reload/rebuild, hidden calls to
  `reload_decode_bf16_shadows()`, or claiming P3-C/server readiness.

## Required Evidence

A P3-B artifact must include:

- exact commit and remote provenance manifest;
- selected layer list;
- model path, container image, launch command, and effective env;
- numeric table per selected layer and for the final selected stack output;
- `no_active_bf16_shadow=true` at candidate start;
- memory before/after deleting active BF16 shadows;
- peak memory for candidate path;
- latency versus P2-N/P2E reference;
- failure/caveat section even when PASS.

## PASS Gates

P3-B PASS requires all of:

- predecessor gates are PASS;
- final selected-stack cosine >= 0.999;
- final selected-stack argmax match;
- every selected layer has active BF16 expert shadows absent before candidate
  execution;
- no reload call is observed;
- candidate latency is not slower than the P2-N selected-layer reference for the
  same layer set and token count;
- artifact has `result.json`, `summary.md`, `report.md`, `run.log`, GPU snapshots,
  remote HEAD/provenance files, and exact command record.

Speed regression may be recorded as engineering evidence, but it is a P3-B FAIL
and cannot be promoted.

## Suggested Command Shape

The eventual runner should look like:

```bash
scripts/run_spark_stage6_p3b_selected_prefill_gate.sh \
  --layers 0-3 \
  --tokens 16,64
```

Until that runner exists, P3-B remains runbook-only. The current executable next
step is still the suite:

```bash
scripts/run_stage6_gpu_gate_suite.sh
```

## Relationship To P3-C/P3-D

P3-B is not a server gate. A PASS only proves selected-layer composition.
Resident-runner real prompt behavior remains P3-C, and promotion/RC quality
remains P3-D.
