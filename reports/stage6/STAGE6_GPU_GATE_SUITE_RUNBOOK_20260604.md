# Stage 6 GPU Gate Suite Runbook

Date: 2026-06-04

Verdict: **RUNBOOK ONLY; no new GPU result is banked by this document.**

This runbook defines the current headless GPU gate suite for the restarted Lynn
engine Stage 6 path. It groups the next required Spark gates into one
artifact-producing command:

- P2-O packed-prefill RC smoke, `basic` preset;
- P2-O packed-prefill RC smoke, `rc-mini` preset;
- P3-A grouped active-MoE contract probe.

## Command

```bash
scripts/run_stage6_gpu_gate_suite.sh
```

Common override for the fallback tunnel:

```bash
scripts/run_stage6_gpu_gate_suite.sh --host dgx-via-ssh
```

Dry-run without touching Spark:

```bash
scripts/run_stage6_gpu_gate_suite.sh --dry-run --local-root /tmp/stage6-suite-dry
```

## Artifact Contract

The suite writes a parent artifact directory:

```text
reports/stage6/stage6_gpu_gate_suite_YYYYmmdd_HHMMSS/
```

Required suite-level files:

- `suite_meta.env` — local HEAD, expected Spark HEAD, host, model, image, mode;
- `local_git_status.txt` — local worktree status before launch;
- `commands.sh` — exact child wrapper commands;
- `suite_status.tsv` — child step status and exit code;
- `summary.md` — suite-level human summary.

Child wrappers write their own artifact subdirectories under the suite root.
They remain authoritative for each gate:

- P2-O uses `scripts/run_spark_stage6_p2o_rc_smoke.sh`, `result.json`, and
  `summary.md` from `scripts/summarize_stage6_p2o_rc_smoke.py`.
- P3-A uses `scripts/run_spark_stage6_p3a_contract_probe.sh`, `result.json`, and
  `summary.md` from `scripts/summarize_stage6_p3a_contract_probe.py`.

## Strict Verdict Rule

Default mode is strict. The suite exits non-zero if any child gate exits
non-zero, but it still attempts every enabled child gate so failure evidence is
preserved.

A suite pass does not automatically promote P2-O or P3. Promotion still requires
reading the child summaries and updating the relevant Stage 6 report with exact
commands, commit, artifact path, and caveats.

## Current Next Use

When Spark SSH is reachable, run:

```bash
scripts/run_stage6_gpu_gate_suite.sh
```

If the primary alias is still unhealthy but the tunnel works:

```bash
scripts/run_stage6_gpu_gate_suite.sh --host dgx-via-ssh
```

If a remote checkout is one commit behind but the files match, the child wrappers
can prove provenance via manifest. Only use this when the generated artifacts
show `remote manifest ok`:

```bash
scripts/run_stage6_gpu_gate_suite.sh --allow-provenance-mismatch
```

## Local CI

GPU-free tooling check:

```bash
scripts/run_stage6_evidence_ci.sh
```

This validates the P2-O evidence tools, P3 contract static checks, P3-A evidence
summarizer, and this suite wrapper's dry-run path.
