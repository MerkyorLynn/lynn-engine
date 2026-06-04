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
