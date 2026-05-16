# P106: A100 W4A8 Intermediate-Scale Recovery

Date: 2026-05-16

## Decision

W4A8 Recovery should continue. The first A100 adaptation probe found a cheap,
foldable correction that clears the 40-layer real-prompt drift gate.

The winning idea is expert-wise intermediate-channel scaling:

```text
inter_q_corrected = inter_q * alpha[layer, expert, channel]
```

This is not a permanent runtime tax. It can be folded into down weights:

```text
down(inter_q * alpha) == linear(inter_q, down_weight * alpha)
```

So the next artifact can carry the correction inside `down_proj`.

## Inputs

Baseline gate:

```text
reports/a100/a100_w4a8_real_prompt_gate_40layer.json
```

Baseline result:

```text
decision: AMBER
all_gateup_relaxed: true
all_full_relaxed: false
max_gateup_rel_l2: 2.88%
max_full_rel_l2: 3.67%
min_gateup_cosine: 0.999588
min_full_cosine: 0.999371
```

This established that gate/up W4A8 is already clean and the remaining problem
is the intermediate/down activation margin.

## Probe 1: Shared Layer Alpha

Script:

```text
scripts/a100_w4a8_intermediate_scale_recovery_probe.py
```

Report:

```text
reports/a100/a100_w4a8_intermediate_scale_recovery_worst6.json
```

Target layers:

```text
20, 26, 24, 12, 23, 32
```

Result:

```text
decision: AMBER
before_max_rel_l2: 3.67%
after_max_rel_l2: 3.07%
delta: 0.60 percentage points
ratio: 0.836
```

Interpretation:

A single `alpha[512]` vector per layer is already useful. It repairs the worst
layer L20 and several other layers, but L12/L23 remain just over the 3% gate.

## Probe 2: Expert-Wise Alpha, L12/L23

Report:

```text
reports/a100/a100_w4a8_intermediate_scale_recovery_expert_l12_l23.json
```

Result:

```text
decision: GREEN
before_max_rel_l2: 3.36%
after_max_rel_l2: 1.49%
ratio: 0.445
```

Layer detail:

```text
L12: 3.36% -> 1.25%, cosine 0.999922
L23: 3.31% -> 1.49%, cosine 0.999888
```

Interpretation:

Expert-wise correction is much stronger than shared layer correction and still
folds directly into expert down weights.

## Probe 3: Expert-Wise Alpha, All 23 Failing Layers

Report:

```text
reports/a100/a100_w4a8_intermediate_scale_recovery_expert_fail23.json
```

Target layers:

```text
20, 26, 24, 12, 23, 32, 5, 3, 19, 29, 15, 17,
7, 2, 10, 6, 30, 8, 25, 13, 37, 18, 11
```

Result:

```text
decision: GREEN
before_max_rel_l2: 3.67%
after_max_rel_l2: 1.79%
delta: 1.88 percentage points
ratio: 0.488
all_after_under_3pct: true
```

Overlay artifact:

```text
/mnt/data/lynn-a100/artifacts/w4a8_alpha_overlay_fail23
files: 23 x layer_*_expert_alpha.pt
size: 11 MiB
checksum manifest: reports/a100/w4a8_alpha_overlay_fail23_SHA256SUMS
```

Worst repaired layers:

```text
L24: 3.43% -> 1.79%, cosine 0.999840
L02: 3.18% -> 1.64%, cosine 0.999865
L25: 3.06% -> 1.56%, cosine 0.999878
L23: 3.31% -> 1.49%, cosine 0.999888
L37: 3.05% -> 1.47%, cosine 0.999894
L26: 3.47% -> 1.43%, cosine 0.999899
```

## Training Meaning

The useful Recovery target is now precise:

```text
primary: expert-wise down/intermediate scale correction
secondary: structured prompts (JSON/code/tool-call)
not primary: router
not primary: gate/up
```

This is materially better than a blind QAT run:

- It repairs the measured drift with only expert/down-side foldable parameters.
- It preserves router semantics.
- It does not add runtime operations after folding.
- It gives the A100 run an immediate quantitative objective.

Important metric framing:

```text
1.79% is local active-MoE output relative-L2 drift after correction.
It is not yet V8/V9 score loss, final-logit loss, or end-to-end generation loss.
```

Still, reducing the worst local active-MoE drift from 3.67% to 1.79% puts the
W4A8 route in the same engineering comfort band as many FP8 activation paths:
small enough to justify artifact folding and generation-level revalidation.

## Next Step

Turn this probe into an artifact-producing Recovery stage:

1. Save `alpha[layer][expert][channel]` tensors for the repaired layers.
2. Fold `alpha` into BF16 `mlp.experts.down_proj` columns.
3. Re-run the 40-layer W4A8 real-prompt gate on the folded artifact.
4. Re-run P105 generation gate.
5. If P105 improves, extend calibration prompts and begin MTP/NEXTN head prototype.

## Folded Overlay Artifact Validation

Follow-up scripts:

```text
scripts/a100_fold_w4a8_alpha_overlay.py
scripts/a100_w4a8_folded_vs_original_gate.py
```

Folded artifact:

```text
/mnt/data/lynn-a100/models/lynn-27b-variable-recovery-step5000-bf16-w4a8-alpha-overlay-v0
copy-on-write size: 11 GiB
source BF16 untouched: 60 GiB
inventory: 1026 / 1026 shards, missing 0
```

Validation report:

```text
reports/a100/a100_w4a8_folded_vs_original_fail23.json
```

Correct validation口径:

```text
reference = original BF16 artifact active-MoE output
candidate = folded artifact + full W4A8 fake-quant active-MoE output
```

Result:

```text
decision: GREEN
max_folded_w4a8_rel_l2: 1.7836%
min_folded_w4a8_cosine: 0.999841
```

This proves the improvement is not just a temporary in-script multiply. The
folded artifact itself, under the W4A8 activation contract, stays under the 3%
local active-MoE drift gate versus the original BF16 reference.

Important caveat:

```text
max_folded_bf16_rel_l2: 3.554%
```

The folded artifact should therefore be treated as a **W4A8/FP8-activation
artifact**, not as a BF16 fallback artifact. The original BF16 final remains
the BF16 reference/fallback.

The folded artifact manifest must carry a hard runtime contract:

```json
{
  "inference_path_required": "w4a8",
  "fallback_path_allowed": false,
  "bf16_fallback_drift_estimate": 0.0355,
  "w4a8_drift_vs_bf16_reference": 0.017836
}
```

Loader behavior:

```text
If inference_path_required == w4a8 and the active backend is not a
W4A8/FP8-active backend, refuse to launch. Do not silently run the folded
artifact through the BF16 fallback path.
```

Implementation helper:

```text
engine/w4a8_contract.py
```

This helper is intentionally not wired into the default BF16 loader yet. It is
for W4A8-capable runtime paths and artifact publishing gates, where fail-loud is
safer than a hidden 3.55% BF16-path drift.

## Renewal Signal

This probe is a strong positive renewal signal.

A100 is now producing actionable W4A8 adaptation evidence, not just loading
the model. Unless the folded-artifact generation gate regresses badly, keeping
the A100 through the next checkpoint is justified.

## P107 Early Generation Gate

Follow-up reports:

```text
reports/a100/a100_p105_original_bf16bmm_2prompt_24tok.json
reports/a100/a100_p105_folded_overlay_bf16bmm_2prompt_24tok.json
```

Why this gate exists:

```text
The original P105 script was written for the packed NVFP4 runtime and requires
packed MoE aliases. BF16 and folded-BF16 artifacts do not carry those aliases,
so a BF16/Torch MoE fake-quant path was added for generation-level triage.
```

Implementation note:

```text
LYNN_W4A8_FAKE_QUANT_ACTIVE now also works for the BF16 optimized / bmm /
Triton MoE research paths. Default behavior remains unchanged when the env var
is off.
```

Result:

```text
2 prompts, 24 new tokens, BF16 bmm decode path

original BF16 self W4A8 gate:
  exact: 1 / 2
  min_same_prefix_tokens: 9
  mean_same_prefix_tokens: 13.0
  decision: RED

folded W4A8-contract artifact self gate:
  exact: 1 / 2
  min_same_prefix_tokens: 8
  mean_same_prefix_tokens: 12.5
  decision: RED

cross-model compare:
  reference = original BF16 baseline
  candidate = folded artifact + full W4A8 fake-quant
  exact: 1 / 2
  min_same_prefix_tokens: 8
  mean_same_prefix_tokens: 12.5
```

Interpretation:

```text
This is a generation-level RED for direct runtime promotion, not a route-level
RED for W4A8.
```

The local active-MoE correction is real (`3.67% -> 1.79%`), but greedy decode
still diverges early on at least one explanatory prompt. Therefore:

- Do not publish a W4A8/NVFP4 production artifact from the current overlay.
- Do start W4A8 Recovery training / adaptation from this overlay signal.
- Keep MTP/NEXTN as a parallel engineering stream, but do not combine it with
  W4A8 until the base W4A8 generation gate turns AMBER/GREEN.

Working time estimate from this point:

```text
W4A8 Recovery first checkpoint: hours, not weeks.
W4A8 generation-gated NVFP4 v0: same day if the next checkpoint moves the
  divergence later or exact; otherwise 1-2 days of Recovery iteration.
MTP/NEXTN smoke: 1-3 days because the current artifact has no draft-head
  tensors and the installed Qwen3.5-MoE class does not expose a ready MTP head.
MTP/NEXTN quality candidate: 3-7 days after the head implementation is stable.
```

## P108 6-Prompt Generation Gate

Follow-up script:

```text
scripts/a100_w4a8_generation_gate.py
```

Report:

```text
reports/a100/a100_w4a8_generation_gate_6prompt_48tok.json
reports/a100/a100_w4a8_generation_gate_6prompt_48tok_triage.json
```

Result:

```text
max_new: 48
original self W4A8: exact 2/6, min prefix 17, mean prefix 36.17
folded self W4A8:   exact 4/6, min prefix 13, mean prefix 37.83
cross compare:
  reference = original BF16 baseline
  candidate = folded artifact + full W4A8 fake-quant
  exact: 2/6
  min prefix: 17/48
  mean prefix: 38.67/48
```

Interpretation:

```text
Still RED for direct promotion, but much stronger than the earlier 2-prompt
24-token gate. The folded artifact now mostly diverges late or with semantically
near-equivalent phrasing. The worst prompt remains the structured JSON/OpenAPI
case, which should be overweighted in the next Recovery batch.
```

First-diff triage:

```text
prompt 0 first diff margin: ref 0.125 / candidate 0.0625
prompt 1 first diff margin: ref 0.125 / candidate 0.125
prompt 3 first diff margin: ref 0.0   / candidate 1.0625
prompt 5 first diff margin: ref 0.125 / candidate 0.0
```

This means the generation RED is mostly **margin-fragile greedy tie breaking**,
not output collapse. The JSON/OpenAPI first diff is especially informative:
the BF16 reference has a zero-margin top-1 tie at the divergence point.

Next Recovery objective:

```text
push min same-prefix above 36/48 on structured prompts;
turn folded self gate from 4/6 exact to 6/6 exact;
then rerun strict tool-call and no-think guards.
```
