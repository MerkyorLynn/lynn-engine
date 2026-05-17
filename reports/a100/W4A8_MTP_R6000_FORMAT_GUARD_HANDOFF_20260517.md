# Lynn 27B A3B W4A8 + MTP Progress Handoff - 2026-05-17

## Bottom Line

The 155 TPS target is still not closed. R6000 safe serving remains around the
previous ~100 tok/s class, and MTP has not been integrated into decode yet.

The useful new signal is narrower and actionable:

- R6000 v2 active-MoE interval work now has a clean P97 serial result.
- A100 `structured_v16_top6_damped075` is the best exact-count Recovery
  candidate so far; with teacher-clean v4 serving templates it reaches GREEN
  on the 12-prompt structured gate.
- A100 format-start guard v3 turns the structured-anchor mini gate GREEN,
  proving first-token/few-token format correction is a high-ROI path.
- Qwen3.6-35B-A3B MTP sidecar is now a Lynn-shape-aligned warm-start asset.

## R6000 P97 v2

Report:

```text
reports/a100/r6000_p97_v2_multilayer_summary_20260517_092812.json
```

Result:

| Field | Value |
|---|---:|
| layers | 4, 12, 20, 28, 36 |
| all contract pass | true |
| best variant | p93_gateup_native_down_tile1 |
| speedup min | 1.1001x |
| speedup mean | 1.1177x |
| speedup max | 1.1263x |
| gate median mean | 0.05777 ms |
| down median mean | 0.02253 ms |

Interpretation: the local active-MoE native-down path gives a repeatable
~1.12x interval micro gain, but this alone is not a 155 TPS bridge. Gate/up
remains the dominant interval, so the next runtime ROI is gate/up scheduling,
fusion, or graph capture, not promoting full-active W4A8.

## A100 Recovery

Reports:

```text
reports/a100/a100_w4a8_generation_gate_structured_v16_top6_damped075_12prompt_48tok.json
reports/a100/a100_w4a8_generation_gate_structured_v17_top6_damped050_12prompt_48tok.json
reports/a100/a100_w4a8_prefix_margin_recovery_structured_v19_top6_expert_20260517_100551.json
reports/a100/a100_w4a8_generation_gate_structured_v19_prefix_margin_top6_expert_12prompt_48tok.json
reports/a100/a100_w4a8_generation_gate_structured_v20_v16_plus_prefix025_12prompt_48tok.json
reports/a100/a100_w4a8_generation_gate_structured_v21_v16_plus_prefix050_12prompt_48tok.json
reports/a100/a100_w4a8_prefix_margin_recovery_structured_v22_guarded12_expert_20260517_103453.json
reports/a100/a100_w4a8_generation_gate_structured_v22_guarded12_prefix_margin_expert_12prompt_48tok.json
```

| Candidate | Damping | Exact | Min Prefix | Mean Prefix | Decision |
|---|---:|---:|---:|---:|---|
| structured_v16_top6_damped075 | 0.75 | 9/12 | 12 | 35.92 | RED |
| structured_v17_top6_damped050 | 0.50 | 6/12 | 7 | 29.58 | RED |
| structured_v19_prefix_margin_top6_expert | n/a | 6/12 | 7 | 31.50 | local GREEN, generation regression |
| structured_v20_v16_plus_prefix025 | v16 + 0.25 prefix | 7/12 | 7 | 30.67 | regression |
| structured_v21_v16_plus_prefix050 | v16 + 0.50 prefix | 5/12 | 1 | 23.75 | severe regression |
| structured_v22_guarded12_prefix_margin_expert | guarded12 conservative expert | 5/12 | 1 | 23.75 | local AMBER, generation regression |
| structured_v10_top6 | 1.00 | 8/12 | 13 | 33.33 | previous best handoff |

Interpretation: 0.75 damping is useful and is now the highest-exact Recovery
candidate, but 0.50 under-corrects and regresses. v16 fixes several structured
JSON/code failures, while remaining misses are mostly explanation/bullet
style drift rather than total collapse.

Prefix-Margin Recovery v19 is an important negative/positive split: expert alpha
over the first 8 guarded structured tokens reduces local active-MoE drift from
4.03% to 2.76%, but the folded artifact regresses the unguarded 12-prompt
generation gate to 6/12. Treat it as evidence for the first-token repair
mechanism, not as a replacement for v16.

Follow-up blends close this branch for now: multiplying v19 back into v16 at
0.25 and 0.50 damping regresses to 7/12 and 5/12 respectively. A more
conservative guarded12 prefix-margin alpha (`v22`, alpha range 0.9-1.1) improves
local drift from 3.92% to 3.15%, but still regresses generation to 5/12. The
lesson is that prefix-margin information must enter a broader teacher-cleanup
or QAT objective; post-hoc multiplicative alpha is too brittle.

## Format Guard Evidence

Reports:

```text
reports/a100/a100_w4a8_format_anchor_gate_structured_v16_no_guard_6prompt_32tok.json
reports/a100/a100_w4a8_format_anchor_gate_structured_v16_guard_6prompt_32tok.json
reports/a100/a100_w4a8_format_anchor_gate_structured_v16_guard_v2_6prompt_32tok.json
reports/a100/a100_w4a8_format_anchor_gate_structured_v16_guard_v3_6prompt_32tok.json
```

| Gate | Forced Prefix | Exact | Min Prefix | Mean Prefix | Ref Format | Cand Format | Raw Prefix | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| no guard | no | 2/6 | 0 | 11.17 | 1/6 | 1/6 | 6/6 | RED |
| guard v1 | short | 5/6 | 6 | 25.33 | 4/6 | 4/6 | 0/6 | RED |
| guard v2 | key-specific | 4/6 | 3 | 24.33 | 4/6 | 5/6 | 0/6 | RED |
| guard v3 | generic JSON-entry | 6/6 | 22 | 27.00 | 6/6 | 6/6 | 0/6 | GREEN |
| guard v3 on v19 | generic JSON-entry | 6/6 | 22 | 27.00 | 6/6 | 6/6 | 0/6 | GREEN |
| serving guard12 on v16 | structured stops | 8/12 served text | 12 | 34.75 | 12/12 | 12/12 | 4/12 | RED, format-clean |

Interpretation: prompt cleanup alone is insufficient because the raw teacher
and candidate can both enter the wrong format domain. A small format-start
guard fixes the mini gate when the prefix is generic enough:

```text
JSON:    "{\n  \""
YAML:    "```yaml\n"
Python:  "```python\n"
Bullets: "- "
```

The raw prefix match stays 0/6 under guarded runs, which means the guard is
actively correcting first-token/few-token drift rather than merely confirming
what the model already wanted.

The original 12-prompt serving guard gate on v16 is more nuanced:

```text
reports/a100/a100_w4a8_serving_guard12_gate_structured_v16_12prompt_48tok.json
```

Balanced JSON/code/bullet stops improve the service-facing result to 8/12
served-text exact with 12/12 reference and candidate format clean. This rescues
the structured JSON/YAML/tool-call failures, but leaves natural-language
wording, `normalize_city` implementation style, and bullet wording drift.

The OpenAI-compatible server now has the same primitive wired in:

```text
response_format: {"type": "json_object"}

or private:
lynn_format_guard: {
  "forced_prefix": "{\n  \"",
  "stop_after": "balanced_json",
  "stop_before": ["<think>", "</think>"]
}
```

Supported `stop_after` modes are `balanced_json`, `code_fence`, and
`bullet_count`. The hook is intentionally opt-in and should be used only for
structured modes while v16 remains the general Recovery baseline.

R6000 live OpenAI HTTP smoke confirms the JSON path is not just an offline gate:

```text
reports/a100/server_guard_http_smoke_20260517_1051.json
reports/a100/server_guard_chat_smoke_metrics_20260517_1055.json
reports/a100/server_guard_repeated_bench_20260517_110648.json
reports/a100/p25_server_decode_tps_v2_20260517_114239.json
```

| Endpoint | Guard | Result | Decode TPS | Notes |
|---|---|---|---:|---|
| `/v1/completions` | `response_format={"type":"json_object"}` | parseable JSON | 96.14 | first token forced from raw newline to `{` |
| `/v1/chat/completions` | `response_format={"type":"json_object"}` | parseable JSON, `finish_reason=stop` | 96.59 | `stopped_reason=stop_token`, no markdown |
| `/v1/completions` | private bullet guard | not sufficient | 98.48 | only the first prefix is forced; content quality can still drift |
| `/v1/chat/completions` repeated bench | `response_format={"type":"json_object"}` | 8/8 parseable JSON, 8/8 stop | 98.99 mean | min/max 95.62/101.07 |
| `/v1/completions` long decode, 512 tok | no guard | wall 88.23, decode 100.11 | 100.11 mean | confirms 155 gap is below service wrapper |

This validates the low-risk serving escape hatch for JSON/tool-call style
outputs, but also draws the boundary: format guard fixes entry-domain and
service-facing trimming, not semantic parity for prose or code.

Teacher-cleanup check with `use_chat_template=True` did not improve the v16
serving gate:

```text
reports/a100/a100_w4a8_serving_guard12_gate_structured_v16_chat_template_12prompt_48tok.json
reports/a100/w4a8_structured_recovery_prompts_v2_teacher_clean_guard_specs.json
reports/a100/a100_w4a8_teacher_clean_v2_gate_structured_v16_12prompt_48tok.json
reports/a100/w4a8_structured_recovery_prompts_v3_teacher_clean_guard_specs.json
reports/a100/a100_w4a8_teacher_clean_v3_gate_structured_v16_12prompt_48tok.json
```

| Gate | Served Exact | Min Prefix | Mean Prefix | Ref Format | Cand Format | Raw Prefix | Decision |
|---|---:|---:|---:|---:|---:|---:|---|
| serving guard12 v16, raw prompts | 8/12 | 12 | 34.75 | 12/12 | 12/12 | 4/12 | RED, format-clean |
| serving guard12 v16, chat template | 5/12 | 8 | 28.42 | 12/12 | 12/12 | 7/12 | RED, regression |
| teacher-clean v2 prompts, raw template | 10/12 | 9 | 32.33 | 12/12 | 12/12 | 0/12 | RED, improved |
| teacher-clean v3 prompts, raw template | 11/12 | 16 | 34.33 | 12/12 | 12/12 | 0/12 | AMBER by plan threshold |
| teacher-clean v4 prompts, raw template | 12/12 | 16 | 34.58 | 12/12 | 12/12 | 0/12 | GREEN, structured-template gate |
| teacher-clean v5 heldout prompts | 9/12 | 3 | 21.50 | 12/12 | 12/12 | 0/12 | RED, format-clean heldout drift |
| teacher-clean v6 heldout-template prompts | 12/12 | 4 | 23.17 | 12/12 | 12/12 | 0/12 | GREEN, explicit heldout templates |

The chat template makes the teacher format clean, but shifts the teacher token
path and worsens parity on this prompt set. Keep raw structured prompts as the
current gate baseline; teacher cleanup needs prompt rewrites, not just wrapping
the same prompts in chat format.

The first prompt-rewrite pass is useful: `teacher_clean_v2` improves served
text exactness to 10/12. The remaining failures are `moe_router_expert_clean`
at prefix 9 and `normalize_city_code_clean` at prefix 39. Both are style or
implementation-choice drift, not JSON/YAML/tool-call format collapse. This is
evidence for a serving-template strategy on structured outputs, but it is not
yet enough to promote full-active W4A8.

The v3 rewrite clears the code-style miss and reaches the planned teacher-clean
AMBER threshold (`>=11/12` served exact and min prefix `>=16`). Only
`moe_router_expert_v3` remains divergent, at prefix 24. This makes v16 the
current quality baseline to keep improving; NVFP4 v2 remains a runtime research
artifact, not the best quality artifact.

The v4 rewrite closes that remaining `moe_router_expert` style gap by making
the router/expert serving template explicit and forbidding the candidate's
previous "final answer" wording. The result is the first 12/12 structured gate:
token-exact, served-text exact, and format-clean for both reference and
candidate. This should be used as evidence for opt-in structured serving
templates on v16, not as a claim that open-ended full-active W4A8 is ready.

The v5 heldout rewrite intentionally changes the structured tasks instead of
reusing the v4 templates. It keeps both teacher and candidate format-clean
(`12/12`) but exposes real heldout drift: served-text exact falls to `9/12`,
token exact to `8/12`, with misses in `normalize_unit` code style, an MTP
Chinese short-answer synonym, and linear-attention bullet expansion. This is a
healthy boundary: v16 remains the best quality baseline, but the next A100 work
should target heldout semantic/code-style stability rather than count v4 as a
general promotion gate.

The v6 heldout-template pass tightens those ambiguous tasks into explicit
service templates and turns the heldout check GREEN: `12/12` token-exact,
`12/12` served-text exact, and `12/12` format-clean. The quality rule is now
clear: use v16 plus explicit structured templates for JSON/code/tool routes;
do not treat raw open-ended W4A8 as promoted.

## Full-Token Graph Slot Trigger

The existing whole-decode graph-slot path is opt-in:

```text
LYNN_FULL_TOKEN_GRAPH_SLOT=1
LYNN_ROUTER_TOPK_SORTED=1
```

`LYNN_ALLOW_UNSORTED_FULL_TOKEN_GRAPH_SLOT=1` bypasses the sorted-router guard
only for diagnostics. This path captures a current-position graph slot for all
40 layers plus `lm_head`, then replays it. The current `generate()` wiring still
captures inside the hot path for every decode step, so it is a correctness
primitive first; reusable slot lifecycle work is needed before it can be a
serving TPS jump.

There is also a direct env-contract conflict with the current R6000 fast MoE
path: `LYNN_MOE_FAST_FIXED=1` requires `LYNN_ROUTER_TOPK_SORTED=0`, while the
strict full-token graph-slot path requires sorted router top-k. This confirms
that MoE graph safety is the real blocker, not just missing benchmarking.

R6000 v2 quick check with `LYNN_MOE_FAST_FIXED=0` confirms the same boundary:

```text
/root/autodl-tmp/reports/p16_155/p13_full_token_graph_slot_v2_fastfixed0_20260517_095829.json
```

| Metric | Value |
|---|---:|
| parity | true |
| new tokens | 8 |
| avg capture | 81.31 ms/token |
| avg replay | 9.60 ms/token |
| replay-only TPS | 104.15 tok/s |
| graph-slot decode TPS | 10.99 tok/s |
| eager decode TPS in this config | 25.75 tok/s |

This validates the strict graph body but rejects capture-per-token as a runtime
path.

## MTP Warm Start

Reports:

```text
reports/mtp/a100_qwen36_a3b_mtp_sidecar_shape_audit_v2_20260517_0935.json
reports/mtp/a100_qwen36_a3b_mtp_warm_start_mapping_aligned_20260517_0938.json
```

Warm-start sidecar on A100:

```text
/mnt/data2/lynn-a100/models/mtp_sidecars/qwen36-35b-a3b-mtp-lynn-warm-start-aligned/mtp.safetensors
```

Forward-smoke report:

```text
reports/mtp/a100_mtp_forward_smoke_20260517_110159.json
reports/mtp/a100_mtp_fc_train_smoke_20260517_111251.json
reports/mtp/a100_mtp_fc_calibration_teacher_clean_v2_20260517_113413.json
reports/mtp/a100_mtp_fc_calibration_saved_teacher_clean_v2_20260517_113808.json
reports/mtp/a100_mtp_fc_calibration_heldout_v1_20260517_113808.json
```

Mapping result:

| Field | Value |
|---|---:|
| sidecar tensors | 19 |
| direct copy | 16 |
| slice first dim | 3 |
| missing source | 0 |
| shape mismatch | 0 |
| dtype mismatch | 0 |
| output size | ~1.6 GiB |

The 3 adapted tensors are the MoE expert tensors, where official sidecar uses
256 experts and the Lynn folded base has 254 stored experts. The aligned
sidecar slices the first dimension and writes a new safetensors file.

Forward smoke now runs the aligned sidecar through the MTP `fc -> decoder layer
-> norm -> lm_head` path and produces finite logits:

| Field | Value |
|---|---:|
| mtp logits finite | true |
| mtp hidden shape | [1, 1, 2048] |
| mtp logits shape | [1, 248320] |
| base next argmax | token 4754 (`{"`) |
| mtp draft argmax | token 98175 (`十二`) |
| argmax match | false |

This upgrades MTP from a shape-only asset to a real forward-wired draft head,
but it is not useful for speculative decode yet.

2026-05-17 R6000 native update: activation-aware fake-native training plus a
rank-flip filter is now the productive MTP repair path. The broad v32
continuation regresses, but v33/v34 train only misses where the native label is
already in the sidecar top-k. Real R6000 P107 native shadow moves:

| Sidecar | Native Shadow Accept | Note |
|---|---:|---|
| v29/v30 plateau | 62/121 = 51.24% | native-label proxy saturated |
| v33 rank-flip | 63/121 = 52.07% | first native positive move |
| v34 rank-flip | **65/121 = 53.72%** | current best, 2 accepts short of 55% |
| v35 rank-flip | not promoted | proxy regresses 63/121 -> 62/121 |
| v37 wider `fc_mtp_layer` | not promoted | `miss_not_in_topk` proxy regresses 63/121 -> 62/121 |
| v38 conservative rank-flip | not promoted | proxy stays 63/121 but loss worsens |

This is still below GREEN-CREDIT, but it is the clearest MTP-native direction:
fix top-k rank flips first, then add calibration for semantic/code labels that
are not yet in top-5.

P107 now supports `LYNN_MTP_SHADOW_TOPK` for containment ceilings. On the
current v34 sidecar under native FP4 lm_head:

| Draft set | Covered | Rate |
|---|---:|---:|
| top1 | 65/121 | 53.72% |
| top2 | 73/121 | 60.33% |
| top4 | 80/121 | 66.12% |
| top8 | 87/121 | 71.90% |

This says the next R6000 runtime ROI is multi-candidate MTP verification or a
small reranker, not another blind low-LR single-candidate continuation.

There is one more serving-shaped caveat: when P107 is run on the v4
structured-template prompts with `--force-prefix-from-spec` and forced-prefix
events skipped, v34 falls to `54/271 = 19.93%` accept and top8 containment is
only `109/271 = 40.22%`. So the quality guard path is not yet MTP-friendly.
The next R6000/A100 MTP calibration must explicitly include guard-forced hidden
states before MTP can be enabled on structured/template routes.

The first guard-forced calibration attempt is v39. The trainer now supports
`--force-prefix-from-spec --skip-forced-prefix-cases`, so it can collect cases
after the service template has injected its prefix. A conservative `fc_norms`
rank-flip pass on v4 templates moves train accept `0/44 -> 3/44`, but v6
heldout stays flat at `22/172 = 12.79%`. This makes the boundary concrete:
quality templates are solved for W4A8 v16, but MTP needs a separate
guard-aware specialist or a multi-candidate runtime path. v40 widens the same
guard-forced test to `fc_mtp_layer`; it also leaves heldout flat at `22/172`
and worsens eval loss, so the tiny rank-flip continuation branch is closed for
guarded structured routes.

P109 tests the lowest-cost reranker idea using only MTP top1 margin. It does
not justify runtime work: on raw v34, the best threshold never switches and
stays `65/121`; on the guarded v4 trace, the best switch only reaches
`55/271`. If we invest in multi-candidate decode, it needs a trained reranker
or a refreshed guard-aware sidecar, not a hand-tuned margin rule.

P111 adds the throughput budget that matters for 155 TPS. Raw v34 top1 gives a
zero-overhead production projection of `153.72` tok/s from a 100 tok/s
baseline, just below target. Raw top2/top4/top8 could clear 155 only with draft
overhead under `0.34/0.72/1.09 ms`, but the measured draft head is `~7.6 ms`.
Guarded structured top8 is only `140.22` tok/s even at zero overhead. So MTP
must either become hidden/near-free, move to multi-token credit, or wait for a
higher native serving baseline; one-token serial sidecar MTP is not enough.

The fc-only train smoke confirms gradient wiring:

| Field | Value |
|---|---:|
| mode | fc-only, single prompt, base argmax label |
| steps | 4 |
| loss before | 8.9365 |
| loss after | 0.0 |
| draft argmax after | token 4754 (`{"`) |
| weights saved | false |

The 12-prompt fc-only calibration gate is stronger:

| Field | Value |
|---|---:|
| prompts | teacher-clean v2, 12 |
| accept before | 0/12 |
| accept after | 12/12 |
| mean loss before | 11.4534 |
| mean loss after | 0.0076 |
| saved sidecar | `/mnt/data2/lynn-a100/models/mtp_sidecars/qwen36-35b-a3b-mtp-lynn-fc-teacherclean-v2-20260517_113808/mtp.safetensors` |
| heldout accept | 5/8 |

This confirms that the MTP draft path can be trained toward base greedy tokens
with only the `mtp.fc.weight` bridge unfrozen. The saved sidecar shows
non-random heldout behavior but is not yet GREEN: failures are JSON-start
drafting ` ``` ` instead of `{` and a Chinese router short-answer drifting to
bullet form. Next MTP calibration needs format-start weighted examples or a
small partial-head unfreeze before iterative accept-rate eval.

## Next Work

1. Keep full-active W4A8 out of default promotion.
2. Use `structured_v16_top6_damped075` as the next Recovery research baseline.
3. Move from format-guard proof to Prefix-Margin Recovery: optimize the first
   1-8 decode tokens on structured JSON/code/tool prompts.
4. Convert format-start guard into a serving-side option for structured modes,
   with balanced JSON stop for object-only outputs.
5. Test whether the full-token graph-slot primitive can be turned into a
   reusable current-position slot lifecycle without capture-per-token overhead.
6. Start Lynn MTP wiring from the aligned 2048-hidden sidecar only after W4A8
   structured/tool-call generation gates reach AMBER/GREEN.
