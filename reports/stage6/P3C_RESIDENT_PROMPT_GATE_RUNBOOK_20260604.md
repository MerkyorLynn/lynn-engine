# Stage 6 Phase 3-C — resident-runner real-prompt gate runbook

Date: 2026-06-04

Verdict: **PASS artifact banked for `basic`; runbook remains the gate contract.**

P3-C is the first real-prompt resident-runner gate after P3-B selected-prefill
composition. It tests whether the P3-A grouped active-MoE contract can survive
tokenized resident generation after active BF16 expert shadows are released.

Tooling status: Spark artifact
`reports/stage6/p3c_basic_resident_prompt_gate_20260604_145940`
banked the basic resident-prompt no-reload gate. This does not bank server
default, RC quality, full `rc-mini`, or long-context behavior.

## Banked Basic Artifact

- Preset: `basic`
- Prompts: `3`
- Generated-ID exact prompts: `3/3`
- Text-prefix prompts: `3/3`
- Loaded memory: `88.161 GiB`
- After release memory: `28.178 GiB`
- Released memory: `60.000 GiB`
- Reload not called: `true`
- Baseline prefill average: `1.228 s`
- Candidate prefill average: `151.861 s`
- Prefill speed ratio: `0.008x`

Use [the artifact report](p3c_basic_resident_prompt_gate_20260604_145940/report.md)
as the authoritative P3-C `basic` result. The slow candidate prefill is a
shipping blocker for high-throughput multi-request serving; this gate banks
correctness/memory only.

## Required Predecessors

Do not run or report P3-C as bankable unless these artifacts exist and are PASS:

- P2-O `basic` report;
- P2-O `rc-mini` non-long shard report (`prompt_indices=0-4`);
- P3-A grouped active-MoE contract report;
- P3-B selected-prefill report;
- suite-level report showing the predecessor chain.

The full P2-O `rc-mini` long-context prompt remains a separate slow-mode scale
gate. P3-C must state its own prompt preset and long-context scope instead of
inheriting a full long-context claim from P2-O.

If P3-B is missing or failed, P3-C may be used only as an engineering diagnostic.

## Scope

P3-C uses real prompts in the resident runner:

- baseline: normal BF16 prefill path;
- candidate: active MoE BF16 shadows released, then
  `LYNN_PACKED_PREFILL_SLOW_MODE=p3a_grouped` plus
  `LYNN_LINEAR_ATTN_PREFILL_BLOCK_GQA=1`;
- router and shared expert remain on the existing BF16 paths;
- projection shadows remain resident in this gate;
- `reload_decode_bf16_shadows()` is forbidden during the candidate no-reload
  phase.

P3-C is not a server/default/RC promotion. P3-D owns promotion and quality.

## Commands

Standalone Spark gate:

```bash
scripts/run_spark_stage6_p3c_resident_prompt_gate.sh \
  --preset basic \
  --p3b-pass
```

Wider prompt smoke after `basic` passes:

```bash
scripts/run_spark_stage6_p3c_resident_prompt_gate.sh \
  --preset rc-mini \
  --p3b-pass
```

The Stage 6 suite runs P3-C only after P3-B PASS:

```bash
scripts/run_stage6_gpu_gate_suite.sh
```

## Required Evidence

Artifact directory:

```text
reports/stage6/p3c_<preset>_resident_prompt_gate_<timestamp>/
```

Required files:

- `result.json`
- `summary.md`
- `report.md`
- `run.log`
- `docker_exit_code.txt`
- `nvidia_smi_before.txt`
- `nvidia_smi_after.txt`
- `git_head.txt`
- `expected_git_head.txt`
- `provenance_manifest.txt`
- `expected_provenance_manifest.txt`
- `head_check.txt`

## PASS Gates

P3-C PASS requires:

- P3-B predecessor explicitly asserted and proven by the suite/child report;
- non-empty prompt/comparison count;
- functional, non-degenerate baseline and candidate outputs;
- generated `new_ids` token-exact on the smoke prompts;
- meaningful active-MoE shadow release and memory drop;
- no `reload_decode_bf16_shadows()` call during candidate phase;
- `banked_server_path=false` and `banked_rc_quality=false`.

Failure is useful evidence, but it cannot be promoted.
