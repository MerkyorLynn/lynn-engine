# Stage 6 Phase 2-O — packed-prefill RC smoke

Date: 2026-06-04

Verdict: **PASS** (functional + generated-token exact smoke).

P2-O is the first resident-runner real-prompt smoke for the combined
packed-prefill direction after P2-N. It tests the active-MoE no-reload
path with `LYNN_PACKED_PREFILL_SLOW_MODE=p2e_hybrid` plus
`LYNN_LINEAR_ATTN_PREFILL_BLOCK_GQA=1`.

Projection shadows intentionally remain resident in this gate
(`include_projection_aliases=False`). This is therefore an active-MoE
no-reload smoke, not a full all-shadow-free serving promotion.

## Artifact

Artifact directory: `reports/stage6/p2o_basic_packed_prefill_rc_smoke_20260604_132946`

| File | Present |
|---|---|
| `expected_git_head.txt` | `True` |
| `expected_provenance_manifest.txt` | `True` |
| `git_head.txt` | `True` |
| `git_status.txt` | `True` |
| `head_check.txt` | `True` |
| `nvidia_smi_before.txt` | `True` |
| `nvidia_smi_after.txt` | `True` |
| `provenance_manifest.txt` | `True` |
| `docker_exit_code.txt` | `True` |
| `run.log` | `True` |
| `result.json` | `True` |
| `summary.md` | `True` |

## Provenance

| Field | Value |
|---|---|
| Expected HEAD | `ed60c672bc57869f96a918c815e6270fdec8d39f` |
| Remote HEAD | `ed60c672bc57869f96a918c815e6270fdec8d39f` |
| Head check | `remote HEAD ok` |
| Manifest matches | `True` |
| Docker exit code | `0` |
| Git status dirty | `True` |
| Preset | `basic` |
| Model | `/home/merkyor/models/Qwen3.6-35B-A3B-lynn-native-w4a16-nvfp4-from-bf16-20260526` |
| Max new tokens | `8` |

GPU before:

```text
NVIDIA GB10, 0 %, [N/A], [N/A]
```

GPU after:

```text
NVIDIA GB10, 0 %, [N/A], [N/A]
```

Provenance manifest:

```text
e59f80900d40e57ddb618adfea5d8bfeba17f870346bec79fc70515904350782 scripts/run_spark_stage6_p2o_rc_smoke.sh
3e6a9a957255abaf92025ae93e9f66e2b8e6d0bc82d3e12f72d88e88dcec384e scripts/spark_stage6_p2o_packed_prefill_rc_smoke.py
0ac998904b29443371397437fb63aeb1c6442f7eb348cc44857ea7e07532a571 scripts/summarize_stage6_p2o_rc_smoke.py
de7bc2191c3177d287d96557de34dc046efb5f2c735ebc18733ea2467bdb9f9f scripts/write_stage6_p2o_report.py
772deb8564ec9095a3752a4054ae2118295df9026dd9e25896cf7ae7769c71dc reports/stage6/P2O_PACKED_PREFILL_RC_GATE_RUNBOOK_20260604.md
```

## Gate Summary

# Stage 6 P2-O RC Smoke Summary

| Field | Value |
|---|---|
| Verdict | **PASS** (functional + generated-token exact smoke) |
| Preset | `basic` |
| Model | `/home/merkyor/models/Qwen3.6-35B-A3B-lynn-native-w4a16-nvfp4-from-bf16-20260526` |
| Max new tokens | `8` |
| Functional non-degenerate | `True` |
| Prompt count gate | `True` |
| Generated-ID exact prompts | `3/3` |
| Text-prefix prompts | `3/3` |
| Release meaningful | `True` |
| Memory drop meaningful | `True` |
| Reload not called | `True` |
| Baseline degenerate | `0/3` |
| Opt-in degenerate | `0/3` |
| Loaded memory | `88.161 GiB` |
| After release memory | `28.178 GiB` |
| Memory drop | `59.983 GiB` |
| Released memory | `60.000 GiB` |
| Baseline prefill avg | `1.225 s` |
| Opt-in prefill avg | `149.622 s` |
| Prefill speed ratio | `0.008x` |
| Baseline decode TPS avg | `30.706` |
| Opt-in decode TPS avg | `43.748` |
| Decode TPS ratio | `1.425x` |

## Per Prompt

| # | Token Exact | Prefix | Baseline IDs | Opt-in IDs |
|---|---|---:|---|---|
| 0 | True | 8/8 | `[16, 22, 478, 220, 17, 20, 16327, 220]` | `[16, 22, 478, 220, 17, 20, 16327, 220]` |
| 1 | True | 8/8 | `[90, 198, 220, 328, 8944, 763, 328, 2665]` | `[90, 198, 220, 328, 8944, 763, 328, 2665]` |
| 2 | True | 8/8 | `[71093, 12305, 198, 727, 884, 11331, 2007, 1590]` | `[71093, 12305, 198, 727, 884, 11331, 2007, 1590]` |

## Caveats

- This releases only active MoE BF16 expert shadows; projection shadows remain resident.
- The gate is a real-prompt resident-runner smoke, not a full RC quality battery.
- Token-exact means generated new_ids match on this smoke prompt set; it is not a logits/hidden-state/full-RC proof.
- Default path remains unchanged when opt-in flags are unset.

## Hard Gates

| Gate | Value |
|---|---|
| Prompt count | `true` |
| Functional non-degenerate | `true` |
| Generated-token exact | `true` |
| Release meaningful | `true` |
| Memory drop meaningful | `true` |
| Reload not called | `true` |
| Released tensors | `80` |
| Released GiB | `60.0` |
| Memory drop GiB | `59.98265838623047` |

## Decision

Bank P2-O for this preset as a resident-runner smoke.

Do not treat this smoke as logits, hidden-state, or full RC quality proof.
A PASS only means generated `new_ids` matched on this prompt preset while
the active-MoE shadow-release/no-reload evidence gates also passed.

## Run Log Tail

```text
    },
    {
      "index": 1,
      "token_exact": true,
      "token_prefix_match": 8,
      "token_prefix_n": 8,
      "text_prefix_200_match": true,
      "baseline_ids": [
        90,
        198,
        220,
        328,
        8944,
        763,
        328,
        2665
      ],
      "optin_ids": [
        90,
        198,
        220,
        328,
        8944,
        763,
        328,
        2665
      ],
      "baseline_text": "{\n  \"answer\": \"No",
      "optin_text": "{\n  \"answer\": \"No"
    },
    {
      "index": 2,
      "token_exact": true,
      "token_prefix_match": 8,
      "token_prefix_n": 8,
      "text_prefix_200_match": true,
      "baseline_ids": [
        71093,
        12305,
        198,
        727,
        884,
        11331,
        2007,
        1590
      ],
      "optin_ids": [
        71093,
        12305,
        198,
        727,
        884,
        11331,
        2007,
        1590
      ],
      "baseline_text": "```python\ndef add_one(x):",
      "optin_text": "```python\ndef add_one(x):"
    }
  ],
  "passes": {
    "prompt_count": true,
    "functional_non_degenerate": true,
    "generated_token_exact": true,
    "token_exact": true,
    "text_prefix_200_match": true,
    "release_meaningful": true,
    "memory_drop_meaningful": true,
    "reload_not_called": true,
    "all": true
  },
  "reload_calls": [],
  "notes": [
    "This releases only active MoE BF16 expert shadows; projection shadows remain resident.",
    "The gate is a real-prompt resident-runner smoke, not a full RC quality battery.",
    "Token-exact means generated new_ids match on this smoke prompt set; it is not a logits/hidden-state/full-RC proof.",
    "Default path remains unchanged when opt-in flags are unset."
  ]
}
[W604 05:42:00.365550080 AllocatorConfig.cpp:28] Warning: PYTORCH_CUDA_ALLOC_CONF is deprecated, use PYTORCH_ALLOC_CONF instead (function operator())
```
