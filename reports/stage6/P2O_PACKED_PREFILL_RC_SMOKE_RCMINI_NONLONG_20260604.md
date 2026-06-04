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

Artifact directory: `reports/stage6/p2o_rc-mini_idx0-4_packed_prefill_rc_smoke_20260604_141640`

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
| Expected HEAD | `85121fde04a00847cc0efa24d98c87dbc54a1451` |
| Remote HEAD | `85121fde04a00847cc0efa24d98c87dbc54a1451` |
| Head check | `remote HEAD ok` |
| Manifest matches | `True` |
| Docker exit code | `0` |
| Git status dirty | `True` |
| Preset | `rc-mini` |
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
a9ffd8dd5a6d22c4662e460432a78ee9798a98b69849b75ab9fa614492ed7a34 scripts/run_spark_stage6_p2o_rc_smoke.sh
b43206e748962c78d9771ecda0879a800f1600e4fef0be8b7dfc77d83a22005b scripts/spark_stage6_p2o_packed_prefill_rc_smoke.py
d0abee38a297a5247cacea4edb3407642ee0e63879678399bba8475b8403894a scripts/summarize_stage6_p2o_rc_smoke.py
de7bc2191c3177d287d96557de34dc046efb5f2c735ebc18733ea2467bdb9f9f scripts/write_stage6_p2o_report.py
33a4c294aa673bdcf6c2b9cdff1c063aad730d214cd3e295522a8140be7ee593 reports/stage6/P2O_PACKED_PREFILL_RC_GATE_RUNBOOK_20260604.md
```

## Gate Summary

# Stage 6 P2-O RC Smoke Summary

| Field | Value |
|---|---|
| Verdict | **PASS** (functional + generated-token exact smoke) |
| Preset | `rc-mini` |
| Prompt indices | `[0, 1, 2, 3, 4]` |
| Partial preset | `True` |
| Model | `/home/merkyor/models/Qwen3.6-35B-A3B-lynn-native-w4a16-nvfp4-from-bf16-20260526` |
| Max new tokens | `8` |
| Functional non-degenerate | `True` |
| Prompt count gate | `True` |
| Generated-ID exact prompts | `5/5` |
| Text-prefix prompts | `5/5` |
| Release meaningful | `True` |
| Memory drop meaningful | `True` |
| Reload not called | `True` |
| Baseline degenerate | `0/5` |
| Opt-in degenerate | `0/5` |
| Loaded memory | `88.161 GiB` |
| After release memory | `28.178 GiB` |
| Memory drop | `59.983 GiB` |
| Released memory | `60.000 GiB` |
| Baseline prefill avg | `0.857 s` |
| Opt-in prefill avg | `108.562 s` |
| Prefill speed ratio | `0.008x` |
| Baseline decode TPS avg | `35.443` |
| Opt-in decode TPS avg | `43.327` |
| Decode TPS ratio | `1.222x` |

## Per Prompt

| # | Token Exact | Prefix | Baseline IDs | Opt-in IDs |
|---|---|---:|---|---|
| 0 | True | 3/3 | `[19, 17, 248046]` | `[19, 17, 248046]` |
| 1 | True | 8/8 | `[90, 198, 220, 328, 13766, 763, 328, 14764]` | `[90, 198, 220, 328, 13766, 763, 328, 14764]` |
| 2 | True | 8/8 | `[71093, 12305, 198, 727, 884, 11331, 2007, 1590]` | `[71093, 12305, 198, 727, 884, 11331, 2007, 1590]` |
| 3 | True | 8/8 | `[96129, 119348, 114174, 97260, 95952, 9616, 15, 29922]` | `[96129, 119348, 114174, 97260, 95952, 9616, 15, 29922]` |
| 4 | True | 8/8 | `[9, 256, 561, 328, 43, 56301, 4560, 1]` | `[9, 256, 561, 328, 43, 56301, 4560, 1]` |

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
      "index": 3,
      "token_exact": true,
      "token_prefix_match": 8,
      "token_prefix_n": 8,
      "text_prefix_200_match": true,
      "baseline_ids": [
        96129,
        119348,
        114174,
        97260,
        95952,
        9616,
        15,
        29922
      ],
      "optin_ids": [
        96129,
        119348,
        114174,
        97260,
        95952,
        9616,
        15,
        29922
      ],
      "baseline_text": "\u5f53\u6c34\u6e29\u964d\u81f3\u51b0\u70b9\uff080\u00b0C",
      "optin_text": "\u5f53\u6c34\u6e29\u964d\u81f3\u51b0\u70b9\uff080\u00b0C"
    },
    {
      "index": 4,
      "token_exact": true,
      "token_prefix_match": 8,
      "token_prefix_n": 8,
      "text_prefix_200_match": true,
      "baseline_ids": [
        9,
        256,
        561,
        328,
        43,
        56301,
        4560,
        1
      ],
      "optin_ids": [
        9,
        256,
        561,
        328,
        43,
        56301,
        4560,
        1
      ],
      "baseline_text": "*   The \"Lynn engine\"",
      "optin_text": "*   The \"Lynn engine\""
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
[W604 06:30:39.571533165 AllocatorConfig.cpp:28] Warning: PYTORCH_CUDA_ALLOC_CONF is deprecated, use PYTORCH_ALLOC_CONF instead (function operator())
```
