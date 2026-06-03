# Stage 6 P2-O Packed-Prefill RC Gate Runbook

Date: 2026-06-04

## Purpose

P2-N proved selected-layer synthetic-hidden coverage for the combined path:

- active MoE packed prefill via `LYNN_PACKED_PREFILL_SLOW_MODE=p2e_hybrid`
- linear-attention prefill via `LYNN_LINEAR_ATTN_PREFILL_BLOCK_GQA=1`

P2-O is the first resident-runner real-prompt gate for the same direction. It
does not promote the path by itself. It verifies whether the runner can:

1. generate baseline prompts through the normal BF16 prefill path;
2. release active MoE BF16 expert shadows;
3. run the same prompts through packed MoE prefill + block linear-attn without
   calling `reload_decode_bf16_shadows()`;
4. preserve functional, non-degenerate output and token-exact agreement on the
   smoke prompts.

Projection shadows intentionally remain resident in this gate:
`include_projection_aliases=False`. P2-O is therefore an active-MoE no-reload
gate, not a full all-shadow-free serving promotion.

## Commands

Local evidence-tooling self-test (GPU-free):

```bash
python3 scripts/test_stage6_p2o_evidence_tools.py
```

Primary Spark path:

```bash
scripts/run_spark_stage6_p2o_rc_smoke.sh --preset basic
```

Fallback SSH alias if the primary FRP path is down:

```bash
scripts/run_spark_stage6_p2o_rc_smoke.sh --preset basic --host dgx-via-ssh
```

If `basic` passes, run the wider prompt smoke:

```bash
scripts/run_spark_stage6_p2o_rc_smoke.sh --preset rc-mini
```

The runner creates and pulls an artifact directory under:

```text
reports/stage6/p2o_<preset>_packed_prefill_rc_smoke_<timestamp>/
```

Expected files:

- `expected_git_head.txt`
- `expected_provenance_manifest.txt`
- `git_head.txt`
- `git_status.txt` (captured before creating the artifact directory)
- `head_check.txt`
- `nvidia_smi_before.txt`
- `nvidia_smi_after.txt`
- `provenance_manifest.txt`
- `run.log`
- `result.json`
- `summary.md`

## Verdict Rules

Use `scripts/summarize_stage6_p2o_rc_smoke.py --strict-exit` as the authority
for the smoke verdict.

The Spark runner also gates evidence provenance before Docker starts. A run may
proceed when either the remote repo `HEAD` matches the expected local `HEAD`, or
the P2-O evidence-file manifest matches exactly. This allows clean feature/main
cherry-picks with different commit hashes while still proving the gate scripts
and runbook content are identical.

If this is intentionally not desired, pass `--allow-provenance-mismatch` and
treat the artifact as a non-canonical diagnostic run.

`PASS` requires:

- `passes.functional_non_degenerate == true`
- `passes.generated_token_exact == true`
- no degenerate baseline or opt-in completions
- non-empty prompt/comparison count
- meaningful active-MoE shadow release and memory drop
- no `reload_decode_bf16_shadows()` call during the opt-in no-reload phase
- artifact notes still match the intended scope

`WARN` means functional/text-prefix survived but generated-token exactness did
not. This must not be promoted as RC-equivalent.

`FAIL` means the packed-prefill no-reload path is not banked for this preset.

## Report Promotion

Only after a real Spark artifact exists:

1. write a dated report from the artifact:

   ```bash
   python3 scripts/write_stage6_p2o_report.py \
     reports/stage6/p2o_basic_packed_prefill_rc_smoke_<timestamp> \
     --report-out reports/stage6/P2O_PACKED_PREFILL_RC_SMOKE_20260604.md
   ```

2. update README/README_EN/RELEASE_NOTES only with the exact verdict;
3. keep the caveat explicit: active MoE shadows only, projection shadows remain
   resident;
4. push both the feature branch and `main`.

## Current State

As of this runbook commit, the P2-O harness, runner, and summarizer are ready on
both the feature branch and `main`. No P2-O benchmark result is banked yet.
