# Stage 6 Phase 3-D — OpenAI server RC smoke gate runbook

Date: 2026-06-04

Verdict: **RUNBOOK/TOOLING ONLY; no P3-D result is banked yet.**

P3-D is the first service-level gate after P3-C. It launches the OpenAI-compatible
server twice on Spark:

- baseline server: normal BF16 prefill path;
- candidate server: `LYNN_PACKED_PREFILL_SLOW_MODE=p3a_grouped`,
  `LYNN_LINEAR_ATTN_PREFILL_BLOCK_GQA=1`, and
  `LYNN_RELEASE_DECODE_SHADOWS_AFTER_PREFILL=1`.

The gate compares greedy server responses, checks `/v1/models`,
`/v1/completions`, `/v1/chat/completions`, and validates candidate `/health`
release/reload counters.

## Boundary

A P3-D PASS banks only an **opt-in server smoke** for the zero-shadow prefill
path. It is not default promotion and not full RC quality.

Full publication still requires the separate MMLU/GPQA/tool/long-context quality
battery and an explicit default-switch decision.

## Preconditions

Run P3-D only after:

- P2-O `basic` PASS;
- P2-O `rc-mini` PASS;
- P3-A PASS;
- P3-B PASS;
- P3-C PASS.

The wrapper requires `--p3c-pass` so a standalone run cannot accidentally bank a
server smoke without the resident-runner predecessor.

## Command

Basic service smoke:

```bash
scripts/run_spark_stage6_p3d_server_rc_gate.sh \
  --preset basic \
  --p3c-pass
```

Wider prompt smoke after `basic` passes:

```bash
scripts/run_spark_stage6_p3d_server_rc_gate.sh \
  --preset rc-mini \
  --p3c-pass
```

Suite orchestration after all earlier gates:

```bash
scripts/run_stage6_gpu_gate_suite.sh --p3d-preset basic
```

## Artifacts

Each run pulls:

```text
reports/stage6/p3d_<preset>_server_rc_gate_<timestamp>/
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
  baseline_server.log
  candidate_server.log
  result.json
  summary.md
  report.md
```

## Hard Gates

- P3-C predecessor explicitly asserted;
- `/v1/models` works for both baseline and candidate servers;
- `/v1/completions` works on every prompt;
- `/v1/chat/completions` works on the selected chat subset;
- baseline and candidate greedy server text match;
- baseline/candidate outputs are not degenerate;
- candidate `/health` reports release-after-prefill enabled;
- candidate `/health` reports decode shadows released after requests;
- candidate release amount is meaningful;
- candidate reload count is at least the expected per-request reload count after
  the first request;
- `banked_default_promotion=false`;
- `banked_full_rc_quality=false`.

Use `scripts/summarize_stage6_p3d_server_rc_gate.py --strict-exit` as the
authority for the smoke verdict.

## Local Tooling Check

GPU-free evidence-tooling self-test:

```bash
python3 scripts/test_stage6_p3d_evidence_tools.py
```

Full local evidence CI:

```bash
scripts/run_stage6_evidence_ci.sh
```

## Report

Formal report writer:

```bash
python3 scripts/write_stage6_p3d_report.py \
  reports/stage6/p3d_basic_server_rc_gate_<timestamp> \
  --report-out reports/stage6/P3D_SERVER_RC_SMOKE_20260604.md
```

The report may bank only the opt-in server smoke. README/default promotion is
closed until full RC quality and default-switch policy are recorded separately.
