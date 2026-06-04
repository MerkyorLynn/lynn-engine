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
