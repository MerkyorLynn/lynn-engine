# Stage 6 Phase 3-C — resident-runner real-prompt gate

Date: 2026-06-04

Verdict: **PASS** (resident-prompt no-reload gate passed).

P3-C runs real prompts through the resident runner after active-MoE BF16
expert shadows are released. Candidate prefill uses
`LYNN_PACKED_PREFILL_SLOW_MODE=p3a_grouped` plus block linear-attn.

**Boundary:** a PASS here is not a server/default/RC promotion. P3-D owns
promotion and quality batteries.

## Artifact

Artifact directory: `reports/stage6/p3c_basic_resident_prompt_gate_20260604_145940`

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
| Expected HEAD | `a435fbf490a7f629ba12a669b88b21f91ea671f4` |
| Remote HEAD | `a435fbf490a7f629ba12a669b88b21f91ea671f4` |
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
4604dddf23648aa6bb96e2ae24a73796d982cb93a032002d856e5f3d25368c3f scripts/run_spark_stage6_p3c_resident_prompt_gate.sh
03b0d54b120ab6d8663adf60285dafe9e3a6bbb6d5b4552c5297a481c28b9c41 scripts/spark_stage6_p3c_resident_prompt_gate.py
99ae211740b85a7ee7288ddde6d1bf8fa786eff480ee4653bc1813b5071b3ff0 scripts/summarize_stage6_p3c_resident_prompt_gate.py
f314473fbb69c766d91db4a69bd6da69c334b73630e07a223bf432fa27fc0836 scripts/write_stage6_p3c_report.py
b43206e748962c78d9771ecda0879a800f1600e4fef0be8b7dfc77d83a22005b scripts/spark_stage6_p2o_packed_prefill_rc_smoke.py
cd0e4f30b931972ee5e603b9e0c7dad24ace08e353d0feedf38dababcd26b923 engine/full_forward.py
7e1d76fee25878a9fe8668254a85b38fe819d7e964b3f9f214f1a0146693103c engine/resident_runner.py
3b4afc553d412f826e9f5129e2bf0c35f70fdfb1f33ff156f8b3ebd8dc87f113 reports/stage6/P3C_RESIDENT_PROMPT_GATE_RUNBOOK_20260604.md
```

## Gate Summary

# Stage 6 P3-C Resident-Prompt Gate Summary

| Field | Value |
|---|---|
| Verdict | **PASS** (resident-prompt no-reload gate passed) |
| Preset | `basic` |
| Model | `/home/merkyor/models/Qwen3.6-35B-A3B-lynn-native-w4a16-nvfp4-from-bf16-20260526` |
| Max new tokens | `8` |
| Banked server path | `False` |
| Banked RC quality | `False` |
| P3-B predecessor pass | `True` |
| Functional non-degenerate | `True` |
| Prompt count gate | `True` |
| Generated-ID exact prompts | `3/3` |
| Text-prefix prompts | `3/3` |
| Release meaningful | `True` |
| Memory drop meaningful | `True` |
| Reload not called | `True` |
| Baseline degenerate | `0/3` |
| Candidate degenerate | `0/3` |
| Loaded memory | `88.161 GiB` |
| After release memory | `28.178 GiB` |
| Memory drop | `59.983 GiB` |
| Released memory | `60.000 GiB` |
| Baseline prefill avg | `1.228 s` |
| Candidate prefill avg | `151.861 s` |
| Prefill speed ratio | `0.008x` |
| Baseline decode TPS avg | `30.553` |
| Candidate decode TPS avg | `40.812` |
| Decode TPS ratio | `1.336x` |

## Per Prompt

| # | Token Exact | Prefix | Baseline IDs | Candidate IDs |
|---|---|---:|---|---|
| 0 | True | 8/8 | `[16, 22, 478, 220, 17, 20, 16327, 220]` | `[16, 22, 478, 220, 17, 20, 16327, 220]` |
| 1 | True | 8/8 | `[90, 198, 220, 328, 8944, 763, 328, 2665]` | `[90, 198, 220, 328, 8944, 763, 328, 2665]` |
| 2 | True | 8/8 | `[71093, 12305, 198, 727, 884, 11331, 2007, 1590]` | `[71093, 12305, 198, 727, 884, 11331, 2007, 1590]` |

## Caveats

- P3-C uses real prompts in the resident runner with candidate mode p3a_grouped.
- Projection shadows intentionally remain resident in this gate.
- P3-C is not a server/default/RC promotion; P3-D owns promotion.
- Default path remains unchanged when opt-in flags are unset.

## Hard Gates

| Gate | Value |
|---|---|
| P3-B predecessor pass | `true` |
| Prompt count | `true` |
| Functional non-degenerate | `true` |
| Generated-token exact | `true` |
| Release meaningful | `true` |
| Memory drop meaningful | `true` |
| Reload not called | `true` |
| Banked server path flag is false | `True` |
| Banked RC quality flag is false | `True` |
| Released tensors | `80` |
| Released GiB | `60.0` |
| Memory drop GiB | `59.98265838623047` |

## Decision

Bank P3-C as a resident-runner real-prompt smoke only. Do not promote server/default/RC.

Do not treat this report as logits, hidden-state, full RC, or server default proof.

## Run Log Tail

```text
        478,
        220,
        17,
        20,
        16327,
        220
      ],
      "candidate_ids": [
        16,
        22,
        478,
        220,
        17,
        20,
        16327,
        220
      ],
      "baseline_text": "17 + 25 equals ",
      "candidate_text": "17 + 25 equals "
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
      "candidate_ids": [
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
      "candidate_text": "{\n  \"answer\": \"No"
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
      "candidate_ids": [
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
      "candidate_text": "```python\ndef add_one(x):"
    }
  ],
  "passes": {
    "p3b_pass": true,
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
    "P3-C uses real prompts in the resident runner with candidate mode p3a_grouped.",
    "Projection shadows intentionally remain resident in this gate.",
    "P3-C is not a server/default/RC promotion; P3-D owns promotion.",
    "Default path remains unchanged when opt-in flags are unset."
  ]
}
[W604 07:12:10.446346073 AllocatorConfig.cpp:28] Warning: PYTORCH_CUDA_ALLOC_CONF is deprecated, use PYTORCH_ALLOC_CONF instead (function operator())
```
