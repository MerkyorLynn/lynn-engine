# Stage 6 P4C Tile2 Server Smoke Summary

| Field | Value |
|---|---|
| Verdict | **PASS** (P4C tile2 server smoke passed) |
| Decision | `PASS_P4C_TILE2_SERVER_SMOKE` |
| Model | `/home/merkyor/models/Qwen3.6-35B-A3B-lynn-native-w4a16-nvfp4-from-bf16-20260526` |
| Preset | `basic` |
| Prompt limit | `2` |
| Max new tokens | `4` |
| Candidate backend | `fused_zero_shadow_active_reuse_contract` |
| Gate/up tile_inter | `2` |
| Banked P4C server smoke | `True` |
| Banked default promotion | `False` |
| Banked full RC quality | `False` |
| P4C runtime predecessor pass | `True` |
| Server surface | `True` |
| Functional non-degenerate | `True` |
| Completion text exact | `2/2` |
| Chat text exact | `0/0` |
| Baseline degenerate | `0/2` |
| Candidate degenerate | `0/2` |
| P4C native call delta | `240` |
| P4C native calls after | `240` |
| P4C layers with calls | `40` |
| Recorded tile_inter | `2` |
| Recorded inter scratch | `[1, 8, 512]` |
| Recorded out scratch | `[1, 2048]` |
| Release enabled | `True` |
| Release consumed | `True` |
| Decode shadows currently released | `True` |
| Release/reload count | `1` |
| Reload expected min | `1` |
| Last release GiB | `60.000` |
| Last reload seconds | `23.246013402938843` |

## Completion Comparisons

| # | Text Exact | Baseline | Candidate |
|---|---|---|---|
| 0 | True | `

<think>
Here` | `

<think>
Here` |
| 1 | True | `

<think>
Here` | `

<think>
Here` |

## Caveats

- P4C tile2 server smoke launches OpenAI-compatible baseline and candidate servers.
- A PASS banks only opt-in server evidence for fused_zero_shadow_active_reuse_contract.
- Default promotion remains false until full RC quality and sustained server speed gates pass.
- This gate checks native call counters and tile recording, not MMLU/GPQA/tool/long-context quality.
