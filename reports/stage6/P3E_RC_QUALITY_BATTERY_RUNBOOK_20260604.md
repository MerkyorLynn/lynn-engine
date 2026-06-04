# Stage 6 Phase 3-E — RC quality-battery smoke runbook

Date: 2026-06-04

Verdict: **RUNBOOK/TOOLING ONLY; no P3-E result is banked yet.**

P3-E is the first quality battery after P3-D's OpenAI server smoke. It keeps the
candidate opt-in server path alive with `LYNN_SKIP_RELOAD_IF_PACKED_PREFILL=1`
so packed prefill must run without rebuilding released BF16 shadows. It checks:

- MMLU sample through `scripts/openai_mmlu_500_5shot_eval.py`;
- GPQA Diamond sample through `scripts/openai_gpqa_diamond_eval.py`;
- structured JSON smoke;
- tool-call smoke;
- repo-local V8/V9-shaped prompt-format smoke;
- long-context needle smoke.

## Boundary

A P3-E PASS banks only **RC quality smoke** for the opt-in zero-shadow prefill
path:

```text
banked_rc_quality_smoke=true
banked_default_promotion=false
banked_full_leaderboard_quality=false
```

It is not a full MMLU/GPQA leaderboard run and not default promotion. Full
release promotion still needs the larger release battery, README/release-matrix
update, and explicit default-switch policy.

## Preconditions

Run P3-E only after:

- P2-O `basic` PASS;
- P2-O `rc-mini` non-long shard PASS;
- P3-A PASS;
- P3-B PASS;
- P3-C PASS;
- P3-D PASS.

The full P2-O `rc-mini` long-context prompt remains a separate slow-mode scale
gate. P3-E has its own long-context needle smoke, so any P3-E long-context claim
must come from the P3-E artifact itself, not from P2-O.

The wrapper requires `--p3d-pass` so a standalone run cannot accidentally bank a
quality smoke without the server-smoke predecessor.

## Command

Default Spark smoke:

```bash
scripts/run_spark_stage6_p3e_rc_quality_battery.sh --p3d-pass
```

Smaller triage sample:

```bash
scripts/run_spark_stage6_p3e_rc_quality_battery.sh \
  --p3d-pass \
  --mmlu-sample 25 \
  --gpqa-sample 20 \
  --longctx-target-tokens 4096
```

Default thresholds:

| Gate | Floor |
|---|---:|
| MMLU sample accuracy | `0.70` |
| GPQA sample accuracy | `0.30` |
| Parse-fail rate | `<= 0.10` |
| Long-context needle | exact needle present |

These are smoke floors, not final benchmark claims.

## Artifacts

Each run pulls:

```text
reports/stage6/p3e_rc_quality_battery_<timestamp>/
  expected_git_head.txt
  expected_provenance_manifest.txt
  git_head.txt
  git_status.txt
  head_check.txt
  nvidia_smi_before.txt
  nvidia_smi_after.txt
  provenance_manifest.txt
  docker_exit_code.txt
  run.log
  candidate_server.log
  mmlu_sample.jsonl
  mmlu_sample.summary.json
  gpqa_sample.jsonl
  gpqa_sample.summary.json
  result.json
  summary.md
  report.md
```

## Hard Gates

- P3-D predecessor explicitly asserted;
- candidate server reaches `/health`;
- `/v1/models` lists the served candidate;
- release-after-prefill is enabled in `/health`;
- skip-reload-for-packed-prefill is enabled in `/health`;
- release/reload count stays zero across the quality battery;
- structured JSON smoke parses and contains required keys;
- tool-call smoke yields an OpenAI `tool_calls` entry or raw function name;
- V8/V9-shaped prompt smoke is non-degenerate;
- long-context needle is recovered;
- MMLU data exists, sample count is met, accuracy floor is met, parse/errors pass;
- GPQA data exists, sample count is met, accuracy floor is met, parse/errors pass;
- `banked_default_promotion=false`;
- `banked_full_leaderboard_quality=false`.

Use `scripts/summarize_stage6_p3e_rc_quality_battery.py --strict-exit` as the
authority for the smoke verdict.

## Local Tooling Check

GPU-free evidence-tooling self-test:

```bash
python3 scripts/test_stage6_p3e_evidence_tools.py
```

Full local evidence CI:

```bash
scripts/run_stage6_evidence_ci.sh
```

## Report

Formal report writer:

```bash
python3 scripts/write_stage6_p3e_report.py \
  reports/stage6/p3e_rc_quality_battery_<timestamp> \
  --report-out reports/stage6/P3E_RC_QUALITY_BATTERY_20260604.md
```

The report may bank only RC quality smoke. Promotion to default requires a
separate full release battery and explicit default-switch decision.
