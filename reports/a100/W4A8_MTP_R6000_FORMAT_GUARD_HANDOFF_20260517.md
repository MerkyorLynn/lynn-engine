# Lynn 27B A3B W4A8 + MTP Progress Handoff - 2026-05-17

## Bottom Line

The 155 TPS target is still not closed. R6000 safe serving remains around the
previous ~100 tok/s class, and MTP has not been integrated into decode yet.

The useful new signal is narrower and actionable:

- R6000 v2 active-MoE interval work now has a clean P97 serial result.
- A100 `structured_v16_top6_damped075` is the best exact-count Recovery
  candidate so far, but still RED.
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
sidecar slices the first dimension and writes a new safetensors file. This is
a warm-start asset only; it is not decode integration and gives no TPS gain
until MTP training and accept-rate evaluation exist.

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
