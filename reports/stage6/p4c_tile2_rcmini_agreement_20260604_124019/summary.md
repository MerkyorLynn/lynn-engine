# Stage 6 P4C Tile2 Server Smoke Summary

| Field | Value |
|---|---|
| Verdict | **FAIL** (server_text_exact gate fail) |
| Decision | `FAIL_P4C_TILE2_SERVER_SMOKE` |
| Model | `/home/merkyor/models/Qwen3.6-35B-A3B-lynn-native-w4a16-nvfp4-from-bf16-20260526` |
| Preset | `rc-mini` |
| Prompt limit | `6` |
| Max new tokens | `8` |
| Candidate backend | `fused_zero_shadow_active_reuse_contract` |
| Gate/up tile_inter | `2` |
| Banked P4C server smoke | `False` |
| Banked default promotion | `False` |
| Banked full RC quality | `False` |
| P4C runtime predecessor pass | `True` |
| Server surface | `True` |
| Functional non-degenerate | `True` |
| Completion text exact | `3/6` |
| Chat text exact | `2/2` |
| Baseline degenerate | `0/8` |
| Candidate degenerate | `0/8` |
| P4C native call delta | `2040` |
| P4C native calls after | `2040` |
| P4C layers with calls | `40` |
| Recorded tile_inter | `2` |
| Recorded inter scratch | `[1, 8, 512]` |
| Recorded out scratch | `[1, 2048]` |
| Release enabled | `True` |
| Release consumed | `True` |
| Decode shadows currently released | `True` |
| Release/reload count | `7` |
| Reload expected min | `7` |
| Last release GiB | `60.000` |
| Last reload seconds | `21.863040685653687` |

## Completion Comparisons

| # | Text Exact | Baseline | Candidate |
|---|---|---|---|
| 0 | False | `

<think>

</think>

42` | `

<think>
Here's a thinking process` |
| 1 | True | `

<think>

</think>

{
 ` | `

<think>

</think>

{
 ` |
| 2 | True | `

'''python
def add_one(x` | `

'''python
def add_one(x` |
| 3 | True | `

<think>

</think>

当水温降至` | `

<think>

</think>

当水温降至` |
| 4 | False | `

<think>
Here's a thinking process` | `

<think>

</think>

*   The` |
| 5 | False | `

<think>
The user wants me to` | `

<think>

</think>

LYNN-Z` |

## Chat Comparisons

| # | Text Exact | Baseline | Candidate |
|---|---|---|---|
| 0 | True | `42` | `42` |
| 1 | True | `{
  "tool": "weather` | `{
  "tool": "weather` |

## Caveats

- P4C tile2 server smoke launches OpenAI-compatible baseline and candidate servers.
- A PASS banks only opt-in server evidence for fused_zero_shadow_active_reuse_contract.
- Default promotion remains false until full RC quality and sustained server speed gates pass.
- This gate checks native call counters and tile recording, not MMLU/GPQA/tool/long-context quality.
