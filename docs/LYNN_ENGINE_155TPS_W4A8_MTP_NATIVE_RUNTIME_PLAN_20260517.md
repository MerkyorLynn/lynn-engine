# Lynn Engine 155 TPS Plan: W4A8 + MTP + Native Runtime

Date: 2026-05-17

## Goal

Hard target:

```text
R6000: Lynn-27B-A3B W4A8+MTP NVFP4 serving >155 tok/s, quality-gated.
Spark: Lynn engine serving must exceed the local llama.cpp baseline on the same
       prompt/token budget, not merely match it.
```

Current confirmed state:

```text
R6000 safe serving: ~99-101 tok/s decode class
R6000 JSON guarded serving: 8/8 parseable, mean decode 98.99 tok/s
R6000 long decode serving: 512-token wall 88.23 tok/s, decode 100.11 tok/s
R6000 v2 active-MoE micro gain: ~1.12x interval, not enough alone
A100 best W4A8 recovery baseline: structured_v16_top6_damped075
A100 teacher-clean v2 serving gate: 10/12 served exact, still RED
A100 teacher-clean v3 serving gate: 11/12 served exact, min prefix 16, AMBER by plan threshold
MTP: aligned sidecar forward works; first-token fc calibration is GREEN, but
     iterative accept is still 22/94 heldout after v3, so no TPS credit yet
```

155 requires a compound win. There is no single safe env flag left.

## Required Speed Stack

The practical route is:

```text
100 tok/s safe baseline
  x 1.10-1.18 native active-MoE / decode-loop consolidation
  x 1.25-1.45 MTP accepted-token multiplier
  -> 138-171 tok/s effective
```

This means MTP cannot be a cosmetic sidecar. It must reach useful accept rate,
and the runtime must make draft verification cheap enough that accepted tokens
actually reduce wall time.

## Workstream A: Quality Floor

Purpose: keep W4A8 from entering the wrong output domain, especially structured
JSON/tool/code prompts.

Current facts:

- Raw v16 generation gate is still RED.
- Serving format guard turns structured format clean.
- `use_chat_template=True` regresses this gate.
- Prompt rewrite teacher-clean v2 improves served exact to 10/12 but remains
  RED due to style/code drift.

Next gates:

| Gate | Promotion Threshold |
|---|---|
| teacher-clean v3 serving gate | >=11/12 served exact, min prefix >=16 |
| structured/tool-call JSON path | 100% parseable, 100% stop, no markdown |
| code path | exact or AST-equivalent gate, not raw wording only |

Decision rule: do not default-promote full-active W4A8 until structured/tool
generation is at least AMBER.

2026-05-17 update: teacher-clean v3 on `structured_v16_top6_damped075` reaches
11/12 served exact, min prefix 16, and 12/12 reference/candidate serving-format
clean. The generic gate script still labels it RED because it expects all exact
or very late divergence, but for the plan threshold this is the first AMBER
quality signal. NVFP4 v2 is still a runtime package, not the best quality
candidate.

## Workstream B: MTP Training

Stage B0 is complete:

- official Qwen3.6-35B-A3B MTP sidecar extracted;
- Lynn expert-count aligned sidecar written;
- MTP forward produces finite logits;
- fc-only one-prompt training can force the draft top-1 to base top-1.

Stage B1 now starts:

```text
frozen base
aligned MTP sidecar
trainable: start with mtp.fc.weight only
target: base greedy next token over calibration prompts
metric: one-token accept = draft argmax == base argmax
```

Stage B1 gates:

| Gate | GREEN |
|---|---:|
| fc-only 12 prompt calibration | accept >= 70% |
| fc-only heldout calibration | accept >= 55% |
| head-only MTP layer unfreeze | accept >= 70% heldout |
| iterative 2-token draft simulation | effective accepted tokens/request >= 1.25x |

2026-05-17 update: fc-only saved calibration reaches 12/12 on the teacher-clean
training set and 5/8 on heldout. This clears the "worth continuing" bar but not
the final GREEN bar. The next calibration pass should overweight format-start
tokens (`{`, code fence, router prefix) or unfreeze the small MTP norm/attention
surface.

2026-05-17 iterative update: first-token success did not automatically become
usable speculative decode. The heldout iterative accept ladder is:

| Stage | Trainable | Heldout Accept | Note |
|---|---|---:|---|
| weighted-math v3 sidecar | `mtp.fc.weight` | 8/94, 8.51% | first token works, later tokens drift |
| iterative v1 | `fc` | 15/94, 15.96% | clear gain |
| iterative v2 | `fc_norms` | 21/94, 22.34% | best ROI so far |
| iterative v3 | `fc_mtp_layer` | 22/94, 23.40% | loss improves, accept barely moves |
| iterative v4 | `fc_mtp_layer` | 25/94, 26.60% | 16-token curriculum keeps moving |
| iterative v5 | `fc_norms` | 26/94, 27.66% | loss drops, accept gain is small |
| iterative v6 | `fc_mtp_layer` | 27/94, 28.72% | step1-weighted, but step1 still 0/8 |
| iterative v7 | `fc_mtp_layer`, step1 only | 28/94, 29.79% | first heldout step1 accept: 1/8 |
| iterative v8 | `fc_mtp_layer`, step1 only | 31/94, 32.98% | heldout step1 improves to 3/8 |
| iterative v9 | `fc_mtp_layer`, steps 1-2 | 32/94, 34.04% | preserves step1/2, adds one later token |

A100 v5 exposes the next bottleneck: heldout step 0 is 8/8, but heldout step 1
is still 0/8. The trainer now has explicit `--step1-weight` and
`--later-token-weight` controls so v6 can stop over-rewarding the already-green
first token. The accept target remains >=55% heldout before this can be counted
as a runtime multiplier.

A100 v6 confirms the bottleneck is not just case weighting: step 1 remains 0/8
and the draft repeatedly prefers `"type"` where the base wants keys like
`city`, `status`, or `text`. The trainer now supports `--train-steps`; v7
should train only step-1 cases to test whether targeted curriculum can break
that mode bias.

A100 v7 gives the first positive step-1 movement: training step1 reaches 3/26
and heldout step1 reaches 1/8. A100 v8 continues this step1-only curriculum
with lower LR and more steps.

A100 v8 confirms the targeted curriculum is the right direction: heldout step1
moves to 3/8 and total heldout accept reaches 31/94. A100 v9 is running on
steps 1 and 2 together to keep the format-key fix while improving the next
token.

A100 v9 holds step1 at 3/8 and step2 at 4/8, with total heldout accept 32/94.
A100 v10 is now targeting steps 3/4/5, where the current heldout rates are
3/8, 1/8, and 2/7.

If fc-only cannot clear 55-70%, unfreeze in this order:

1. `mtp.fc.weight`
2. `mtp.norm.weight`
3. MTP layer norms + q/k norms
4. MTP self-attention projections
5. MTP MoE router/experts last

Do not train base weights in this stage.

## Workstream C: Speculative Decode Integration

MTP only counts if verification changes wall time.

First implementation should be single-stream and greedy:

```text
base prefill -> base token t0
for each step:
  mtp drafts token d1 from current hidden/token
  base verifies d1 while computing next base hidden
  if d1 == base token: accept and advance without emitting mismatch
  else: emit base token and reset draft context
```

Initial serving knob:

```text
LYNN_MTP_SIDECAR=/path/to/mtp.safetensors
LYNN_MTP_DRAFT_STEPS=1
LYNN_MTP_ACCEPT_GATE=exact_argmax
```

Then extend to 2-token draft only after 1-token exact verification is stable.

## Workstream D: R6000 Native Runtime

The remaining base-runtime gap is active MoE and decode-loop ownership.

Short-term rejected shortcuts:

- capture-per-token whole-decode CUDA graph: ~10 tok/s, rejected;
- merged-topk gate/up: slower;
- scale-hoist gate/up: slower;
- down-only native tile: local win but not enough and not a promotion by itself;
- plain `_scaled_mm` selected-expert composition: speed/quality fail.
- top-k reduction or skipping shared expert: small speed gain, immediate quality
  divergence.

2026-05-17 profile update: the safe Config D MoE budget is about
`8.12 ms/token` across 40 layers. Router/top-k is `1.91 ms` (23.5%), active
routed experts are `2.58 ms` (31.8%), shared expert is `2.29 ms` (28.2%), and
residual composition/dispatch is `1.34 ms` (16.5%). The one-run budget ladder
only improved median TPS from `100.77` to `107.90` at the most aggressive
setting, and every approximation variant diverged early. The next speed work
must preserve top-k/shared semantics and replace the real runtime boundary.

2026-05-17 P97 v2 layer-28 rerun: `p93_gateup_native_down_tile1` passes the
quantized-activation contract and is the best local active-MoE interval:
`0.08774 ms` baseline -> `0.08022 ms`, or `1.094x`. This validates the local
micro-gain on the v2 package, but it is still far below the full-server delta
needed for 155 TPS. A serial 5-layer P97 job is running to avoid repeating the
earlier parallel OOM.

2026-05-17 P97 v2 5-layer summary: layers 4/12/20/28/36 all pass contract, and
all choose `p93_gateup_native_down_tile1`. The speedup range is `1.094-1.120x`,
mean `1.105x`. The best down interval is stable at about `0.02253 ms`, while
the best gate/up interval still averages `0.05797 ms`; the next R6000 speed
lever is therefore gate/up scheduling/fusion, not more down-only work.

2026-05-17 split16+tile1 generate gate: the P97 local winner fails in real
decode. Config D baseline median is `101.13 tok/s`; split16 gate/up plus native
down tile1 falls to `25.15 tok/s` median and `0/3` exact IDs because it must run
with graph reuse disabled and activation quantization changes the generated
path. Do not pursue this as a serving flag combination.

2026-05-17 full-token diagnostic profile: the older `_decode_layer` benchmark
path estimates `36.59 ms/token`, so it is not the service TPS number. Its
relative shape is still useful: 30 linear-attention layers average `0.905 ms`
each, 10 full-attention layers average `0.791 ms` each. R6000 is now profiling
layer-34 linear-attention segments to choose the next runtime boundary.

2026-05-17 linear-attention segment profiles: layers 0/24/34 are consistent.
The full linear-attention core is `0.330-0.341 ms`; fused native-FP4 in-proj is
the largest single segment at `0.075-0.079 ms`, followed by recurrent
`~0.036 ms` and conv `~0.033 ms`. This makes in-proj fusion worth improving, but
it is too small to be the whole 155 TPS lever.

2026-05-17 Config D service anchor: the R6000 OpenAI server confirms decode TPS
`99.42 / 100.01 / 100.35` at max tokens `128 / 256 / 512`. This is the current
usable serving baseline; 155 still requires a real multiplier.

Required native path:

```text
router/top-k unchanged
Lynn-native per16 active-MoE kernel consumes:
  hidden[2048]
  expert_ids[8]
  routing_weights[8]
  gate_up packed E2M1 + fp32 per16 scales
  down packed E2M1 + fp32 per16 scales
returns active_out[2048]
```

C++/CUDA refactor gates:

| Gate | Purpose |
|---|---|
| native active-MoE fused contract | one boundary for gate/up + down |
| C++ decode loop prototype | token loop outside Python, Python only control plane |
| resident state ABI | KV + linear recurrent state owned by runtime |
| server parity gate | same greedy IDs as Python runner |
| server TPS gate | >115 without MTP, then >155 with MTP |

The C++ refactor is justified if either:

- MTP reaches useful accept but Python runner cannot realize wall-time speedup;
- native active-MoE fused kernel cannot be called efficiently from Python/Triton
  orchestration.

## Workstream E: Spark > llama.cpp

Spark target is not the same kernel path as R6000:

- Spark-class hardware favors FP8/W4A8 and MTP; it should not chase R6000-only
  FP4 tensor-core assumptions.
- Spark must benchmark against llama.cpp on the same prompt set, context, max
  tokens, batch=1, and temperature=0.
- The win condition is service-facing tok/s with correctness guards, not an
  isolated kernel microbench.

Spark plan:

1. keep manifest-driven v2 loader compatibility;
2. preserve Config D as stable fallback;
3. add the same MTP sidecar accept-rate gate;
4. port only the runtime pieces that survive R6000 gates;
5. compare against llama.cpp after warmup and with identical token budget.

Spark should win by specializing for Lynn A3B + W4A8 + MTP, not by becoming a
generic GGUF runner.

## Immediate Execution Order

1. Run fc-only MTP calibration gate on teacher-clean v2 prompts.
2. If accept improves, add saveable trained sidecar and heldout eval.
3. Write one-token speculative simulation gate using saved sidecar.
4. Start native active-MoE fused contract implementation or C++ decode-loop
   scaffold depending on whether MTP is runtime-bound or quality-bound.
5. Keep R6000 server JSON guarded path as the safe serving baseline while
   the 155 path matures.

## Definition Of Done

155 is closed only when all are true:

```text
R6000 OpenAI server >155 effective tok/s on repeated 256/512 token requests
structured JSON/tool-call guard stays GREEN
W4A8 generation gate is AMBER/GREEN
MTP accept-rate report explains the speedup
no hidden full-active promotion without gate evidence
```
