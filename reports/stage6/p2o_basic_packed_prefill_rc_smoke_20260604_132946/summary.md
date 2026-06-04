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
