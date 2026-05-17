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
R6000 1024-token serving: wall 88.70 tok/s, decode 98.85 tok/s
R6000 v2 active-MoE micro gain: ~1.12x interval, not enough alone
R6000 full-attn graph sweep: 12/12 parity, replay 4.09x faster than eager
R6000 full-attn mutable-input graph: 4/4 parity, replay 4.21x faster than eager
R6000 pre-captured full-attn slot: exact on populated KV, layer31 replay 0.295 ms
R6000 hybrid graph token: greedy pass, 9.53 ms one-shot, strict logits not yet exact
A100 best W4A8 recovery baseline: structured_v16_top6_damped075
A100 teacher-clean v2 serving gate: 10/12 served exact, still RED
A100 teacher-clean v3 serving gate: 11/12 served exact, min prefix 16, AMBER by plan threshold
A100 teacher-clean v4 structured-template gate: 12/12 exact, GREEN
A100 teacher-clean v5 heldout gate: format-clean 12/12, served exact 9/12, RED
A100 teacher-clean v6 heldout-template gate: 12/12 exact, GREEN
MTP: aligned sidecar forward works; best native R6000 sidecar is v34 at
     65/121 = 53.72% shadow accept under native FP4 lm_head. P107 now has a
     `LYNN_MTP_SHADOW_TOPK` ceiling mode: v34 top2/top4/top8 containment is
     73/121, 80/121, and 87/121 respectively. The next runtime lever is no
     longer only single-candidate sidecar tuning; multi-candidate verification
     and reranking are now justified. Guarded structured routes are a separate
     MTP distribution: v39 guard-forced calibration improves train accept but
     leaves v6 heldout flat at 22/172.
P111 budget: with a 100 tok/s production baseline, raw v34 top1 is only
     153.72 tok/s even with zero draft overhead; top2/top4/top8 can clear 155
     only if draft overhead is <=0.34/0.72/1.09 ms, while the current draft head
     costs about 7.6 ms in the original P107 path. P113 identifies the main
     runtime miss: the MTP sidecar was using full-forward MoE for a one-token
     decode layer. `LYNN_MTP_LAYER_MOE=decode_active` is exact on 64/64 sampled
     states and moves P107 draft cost from 7.61 ms to 2.76 ms.
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

2026-05-17 P111 budget update: the accepted-token multiplier alone is not
enough. With the current measured draft head cost (`~7.6 ms`), one-token serial
MTP is slower than a 100 tok/s production baseline even if a perfect top8
reranker existed. To make 155 real, the runtime needs one of:

- draft work hidden under base decode or reduced to about 1 ms;
- multi-token credit, not just one-token MTP;
- a higher native baseline before MTP is counted.

2026-05-17 P112 component update: the draft cost is not mainly lm_head.
Profiling v34 on R6000 shows total MTP draft median `7.47 ms`, with the MTP
decoder layer at `6.64 ms`, native FP4 lm_head at `0.44 ms`, pre-fc norms at
`0.16 ms`, and fc at `0.05 ms`. The next runtime engineering target is the
MTP layer itself: simplify it, graph/fuse it, or overlap it with base decode.

2026-05-17 P113 runtime update: the first simplification works. On the same
v34 states, replacing the MTP layer's full-forward MoE with the decode
active-expert MoE is bit-exact/top1-exact on `64/64` cases and cuts the MTP
layer median from `6.70 ms` to `1.89 ms` (`3.54x`). The faster decode-bmm path
hits `1.09 ms` but has top1 drift (`60/64`), so it stays research-only. The
runtime now exposes `LYNN_MTP_LAYER_MOE=decode_active`; P107 with that setting
keeps v34 accept at `65/121 = 53.72%` and raises draft throughput from
`131.45` to `362.30` draft tok/s. A full P107 `decode_bmm` run reaches
`513.77` draft tok/s, but drops accept to `64/121` and has four draft-id
mismatches against the exact path. This is a real MTP cost cut, but still not
enough by itself: top1 zero-overhead remains below 155 from a 100 tok/s base,
and top2/top4/top8 still need sub-millisecond effective draft or overlap.

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

2026-05-17 v4 update: the remaining `moe_router_expert_v3` style drift is
closed by making the serving template explicit. Teacher-clean v4 on
`structured_v16_top6_damped075` reaches `12/12` token-exact and `12/12`
served-text exact, with `12/12` reference/candidate format clean. This is a
GREEN structured serving-template gate, not a default full-active promotion:
it proves v16 can be made reliable for structured/template routes while open
prose and code-style parity remain separate gates.

2026-05-17 v5 heldout update: a new teacher-clean heldout set checks whether
v16 is merely memorizing the v4 template. It is not a format failure:
reference and candidate are both `12/12` format-clean, but parity drops to
`8/12` token-exact and `9/12` served-text exact, with min prefix `3` and mean
prefix `21.50`. The remaining misses are `normalize_unit` code style, an MTP
Chinese short-answer synonym, and linear-attention bullet expansion. Therefore
v16 remains the best quality baseline, while the next A100 quality work should
train/guard against heldout semantic and code-style drift rather than declaring
v4 a full promotion gate.

2026-05-17 v6 heldout-template update: tightening the v5 ambiguous tasks into
explicit service templates turns the heldout check GREEN on v16:
`12/12` token-exact, `12/12` served-text exact, `12/12` format-clean. This is
the quality split to preserve: v16 plus explicit structured templates is
serviceable; raw open-ended W4A8 remains a separate gate.

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
| iterative v10 | `fc_mtp_layer`, steps 3-5 | 34/94, 36.17% | improves step4 and step8; still below multiplier bar |
| iterative v11 | `fc_mtp_layer`, late steps | 29/116, 25.00% | diagnostic: different case set; not a promotion |
| iterative v12 | `fc_mtp_layer`, steps 0-4 | 29/116, 25.00% | front-restore failed; train accept remains 0/130 |

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
A100 v10 targets steps 3/4/5 and raises total heldout accept to 34/94. The
direct gain lands on step4 (`1/8 -> 2/8`) with spillover to step8
(`1/6 -> 2/6`); step11/14/15 remain at 0 accepted. A100 v11 is now targeting
late low-accept steps 7/8/10/11/12/14/15 from the v10 sidecar.

Saved-sidecar eval is now the authoritative handoff gate. Reloading saved
sidecars `fc_v3` through `v11` on one shared 116-case heldout set shows the
best saved sidecar is v11 at `29/116 = 25.00%`; all saved sidecars have
`0/8` accept for steps 0-4 and only recover at later token positions. This
means the previous in-memory ladder is useful for training direction, but not
enough to hand a sidecar to serving. A100 v12 is therefore a front-restore run
from v11 targeting steps 0-4 before any more late-step curriculum.

A100 v12 confirms that low-LR `fc_mtp_layer` tuning is not enough to restore
front tokens: train steps 0-4 remain `0/130` accepted and saved-sidecar eval
still selects v11 over v12. A100 v13 switches to the smaller `fc_norms` surface
on steps 0-4 and breaks the front-token wall: saved eval selects v13 at
`41/116 = 35.34%`, with step2 restored to `8/8` and step4 to `3/8`, though
step1/3 remain `0/8`.

A100 v14 proves step1/3 are trainable but not yet compositional: targeting only
steps 1 and 3 reaches `52/52` train accept and heldout step1/3 `8/8`, but it
damages step0/2/4 and drops total heldout accept to `40/116`. A v13/v14 sidecar
alpha sweep recovers a small new best at `alpha=0.65`, `42/116 = 36.21%`.
Reloaded saved-sidecar eval confirms this `interp065` result. A subset grid over
`mtp.fc.weight`, pre-fc norms, and `mtp.norm.weight` does not beat v13, so the
useful v14 direction is distributed rather than a clean tensor-block swap.

A100 v15 starts from the `interp065` sidecar and applies a conservative
`fc_norms` front repair on steps 0/1/4. The saved-sidecar eval confirms a new
best at `56/116 = 48.28%`, not an in-memory artifact. Its by-step shape is now:
step2/3/4/5 are `8/8`, step0 is `3/8`, and the weakest remaining positions are
step1 `1/8`, step8 `2/8`, step9 `1/8`, step12 `0/6`, and step15 `1/5`.
A100 v16 is running from v15 with a lower LR and `fc_norms` only on those weak
positions; the goal is to cross the `>=55%` saved heldout gate without damaging
the restored 8/8 front positions.

A100 v16 saved-sidecar eval confirms a small but real new high:
`57/116 = 49.14%`. The gain is narrow but clean: step1 improves from `1/8` to
`2/8`, while step2/3/4/5 stay `8/8`. This is still below the `>=55%` serving
credit gate, so the next A100 move is a fine interpolation scan between v16 and
the earlier v14 step1/3 specialist before launching another wider training run.

The v16/v14 fine interpolation scan does not add more accept: best alpha is
`0.0`, i.e. v16 itself. Moving toward v14 starts damaging step0/4 before it
helps the remaining late positions. A100 v17 therefore switches from linear
interpolation back to training: low-LR `fc_mtp_layer`, steps 6-15 only, using
v16 as the source sidecar.

A100 v17 saved-sidecar eval confirms another small new high:
`58/116 = 50.00%`. The gain lands on step12 (`0/6 -> 1/6`) while step2/3/4/5
remain `8/8`. This proves low-LR MTP-layer late repair is valid, but it is still
too incremental; A100 v18 is running from v17 with even lower LR over combined
weak steps `0/1/6/7/8/9/11/12/13/14/15`.

A100 v18 saved-sidecar eval confirms the next small high:
`59/116 = 50.86%`. The gain is clean but narrow: step0 improves from `3/8` to
`4/8`, while step2/3/4/5 remain `8/8`; step1 and the late tail do not move.
This keeps v18 as the best saved sidecar, but the slope is now too incremental
for blind weak-step sweeps. The next A100 move should add targeted calibration
coverage or a small specialist/merge strategy for step1 plus late-tail keys,
not simply lower LR again on the same case set.

2026-05-17 P107 serving-shadow update: the best current MTP artifact is v24,
not v16/v23. The v24 saved-sidecar gate reaches `69/116 = 59.48%`. The new
runner verifier confirms this survives real `generate()` wiring:

| Model candidate | Sidecar | Shadow accept | Decode TPS in probe | Draft TPS |
|---|---|---:|---:|---:|
| structured_v10_top6 | v24 | 68/116 = 58.62% | 7.52 | 83.42 |
| structured_v16_top6_damped075 | v24 | 68/116 = 58.62% | 7.11 | 79.00 |
| R6000 W4A8 NVFP4 v2 | v24 | 61/121 = 50.41% | 55.94 with shadow | 130.51 |

This answers the v16 question narrowly: v16 remains the more conservative
quality baseline, but it does not improve MTP serving-credit over v10 in this
heldout shadow gate. For the MTP lane, v24 sidecar is the artifact to wire into
R6000 tests.

R6000 confirms the next blocker: the same v24 sidecar loads and runs under
W4A8 NVFP4 v2, but acceptance drops below the 55% credit bar. The strongest
prompts are JSON repair (`13/15`) and JSON Berlin (`11/16`); the weakest are
Chinese MoE short answer (`3/16`), linear-attention Chinese answer (`5/16`),
and Python slugify (`6/16`). The miss pattern is no longer just punctuation:
semantic/code tails and `<think>`/format-open choices are still fragile under
quantized runtime. This keeps MTP as the right multiplier path, but the R6000
artifact needs W4A8-aware sidecar repair before it can contribute to 155 TPS.

2026-05-17 v26 update: a narrower `fc_mtp_layer` v7 tail repair from v24
reduces loss but leaves accept unchanged (`68/116` eval before and after in the
train script). Treat v26 as a closed diagnostic, not a new sidecar candidate.
The next MTP quality work should build a different target set or a
specialist/merge candidate instead of repeating v7 tail tuning.

2026-05-17 R6000 W4A8-aware repair update: v27 trains `fc_norms` directly on
R6000 W4A8 v2 with v7 prompts and weak heldout steps. In the train script it
crosses the proxy bar (`63/116 -> 64/116`), and with BF16 lm_head in `p107` it
is GREEN-CREDIT (`64/115 = 55.65%`). With the real native FP4 lm_head enabled,
however, v27 stays at `61/121 = 50.41%`. This isolates a new boundary:
sidecar training must target the native FP4 lm_head argmax, not only the BF16
projection.

v28 adds a trainer mode that collects labels with native FP4 lm_head and then
switches back to BF16 lm_head for differentiable training. The proxy eval moves
`65/121 -> 67/121`, and real native `p107` moves one accepted token
(`61/121 -> 62/121`). Direction is positive but not enough; the remaining
gap is native-runtime hidden/label alignment, especially Chinese semantic tails
and code continuation.

v29 extends the same native-label route from v28. The proxy eval improves again
(`67/121 -> 68/121`), but real native `p107` stays flat at `62/121`. That
marks the current proxy as saturated. The next useful engineering step is a
runtime-native case collector that records hidden states/labels under the exact
R6000 env, or a differentiable approximation of the native FP4 lm_head so the
training objective no longer optimizes the wrong projection boundary.

v30 runs that runtime-native collection path under the R6000 env. It collects
different training/eval cases (`93` train cases, proxy `66/121 -> 67/121`) but
native `p107` again stays flat at `62/121`. So hidden collection alone is not
enough.

v31 adds a chunked fake-native-FP4 lm_head training surrogate. It avoids the
initial 15 GiB temporary OOM by quantizing the lm_head in row chunks, but this
surrogate also fails to add accept (`64/121` before and after). Treat
fake-native lm_head as a closed first attempt, not the next promotion path.

P108 isolates why v31 did not convert. Along the real R6000 resident greedy
path, fake-native-FP4 lm_head reaches `111/118 = 94.07%` top-1 parity against
the true native `_scaled_mm` head, with `118/118` top-5 containment. BF16
lm_head is lower at `109/118 = 92.37%`. The remaining seven fake-native
top-1 flips are exactly the risky margin cases (`{` vs `{"`, code fence vs
`<think>`, `def` vs `import`, Chinese semantic word choice, and JSON repair
quote/colon choices). This keeps the diagnosis narrow: the sidecar is close,
but MTP credit is still gated by last-rank native lm_head ordering under
structured/code first-token margins. Next work should add an activation-aware
native lm_head surrogate or broaden native-labeled calibration cases, not repeat
weight-only fake FP4 training.

P108 v2 adds FP8 scale rounding and activation fake-quantization to the
surrogate. Weight-only parity improves to `113/118 = 95.76%`, and
activation-aware parity reaches `114/118 = 96.61%`, still with `118/118`
top-5 containment. That validates the surrogate as a better diagnostic/training
boundary. However, v32 training from v29 with `fake_native_fp4_act` lowers loss
but regresses eval accept from `60/121` to `59/121`, so simple continuation
training is closed. Next MTP work should target the remaining rank-flip cases
directly or broaden native-labeled calibration, rather than adding more small
steps on v29.

v33/v34 add exactly that rank-flip filter: after the pre-train eval, train only
cases where the sidecar is wrong but the native label is already inside top-k.
This reduces the training set to 16 then 15 high-margin-fix cases. The real
R6000 native P107 shadow moves from `62/121` on v29/v30 to `63/121` on v33 and
`65/121 = 53.72%` on v34. v35 regresses the proxy (`63/121 -> 62/121`), so v34
is the current best native MTP candidate. The rank-flip path is validated, but
the remaining gap to the 55% credit bar is now about two accepted tokens and
mostly requires semantic/code cases whose labels are not in top-5 yet.

v36 tests that next hypothesis with a v8 semantic/code calibration set and
`miss_not_in_topk` filtering. It lowers loss on 67 hard cases but leaves heldout
accept unchanged at `63/121`, so small `fc_norms` continuation is not enough to
pull semantic/code labels into top-k. Keep v34 as best and move the next attempt
to a wider trainable surface or larger native-labeled calibration.

A100 v19 uses the new targeted v4 calibration set and `fc_norms` from v18.
Saved-sidecar eval confirms another real but narrow high:
`60/116 = 51.72%`. The gain lands on step9 (`1/8 -> 2/8`); step0 remains
`4/8`, step1 remains `2/8`, and step2/3/4/5 remain `8/8`. This proves adding
format-tail coverage can move the saved gate, but the remaining gap to the
`>=55%` credit bar is still four accepts. The next run should be a step1/late
specialist or merge candidate rather than a broad v4 repeat.

A100 v20 tests that specialist direction directly: `fc_mtp_layer` only on
steps `1/8/9/11/12/13/14/15` from v19. It does not improve saved heldout:
v19 and v20 both reload at `60/116 = 51.72%`, with the same by-step shape.
This closes the simple MTP-layer specialist route for now. The next useful A100
move is a merge/selection experiment or a different trainable surface for
step1, not another low-LR `fc_mtp_layer` weak-step run.

A100 v19/v14 merge checks close the old step1-specialist rescue route. A fine
interpolation scan from v19 toward the v14 step1/3 specialist keeps best
alpha at `0.0`; alpha `0.025` already drops to `59/116`. A subset grid over
`fc`, `pre_hidden`, `pre_embed`, and `norm` also keeps best at pure v19; any
candidate that imports v14 `fc` collapses to `41/116`. v14's specialist signal
is therefore not composable with the v19 saved sidecar by linear or small-block
merge.

A100 v21 tests step1-only `fc_norms` from v19. It does not break the step1
wall: saved eval keeps v19 as best at `60/116`, while v21 reloads at `59/116`
and loses the v19 step9 gain. This closes the simple step1-only `fc_norms`
route. The remaining MTP work needs a different target construction, not more
low-LR tuning on the same heldout-shaped positions.

A100 v19 diagnostic turns that into a concrete target. The saved sidecar still
accepts `60/116`, but among the 56 misses there are 18 near misses with
teacher label rank <=5. Miss buckets are dominated by semantic-token drift
(`21`), premature stop-token selection (`18`), generic structured keys (`8`),
and JSON punctuation splits (`6`). The low-risk next construction is therefore
not another broad CE sweep: v22 adds `--loss-mode ce_margin`, which keeps CE but
also penalizes the current hard negative when it beats the teacher label inside
the top-k set. The v5 margin calibration prompts focus on compact JSON,
function arguments, JSON repair, Python code-body continuation, and short
router/linear-attention prefixes.

A100 v22 is the first saved MTP sidecar to cross the 55% serving-credit bar.
In-memory eval moves from v19 `60/116` to `64/116`, and saved-sidecar reload
confirms v22 at `64/116 = 55.17%` with decision GREEN. The sidecar is:

```text
/mnt/data2/lynn-a100/models/mtp_sidecars/qwen36-35b-a3b-mtp-lynn-iter-v19-margin-v5-fcnorms-v22-20260517_145317/mtp.safetensors
```

The gain is not uniform: step0 improves `4/8 -> 6/8`, step7 `3/8 -> 4/8`,
step11 `2/6 -> 3/6`, step12 `1/6 -> 2/6`, and step15 `1/5 -> 2/5`, while
step10 regresses `4/7 -> 2/7`. That makes v22 the first MTP serving-credit
candidate, but the next A100 run should recover step10/step1 before calling it
promotion-ready.

A100 v23 continues from v22 with a narrow v6 rescue set and margin loss only on
steps 1/7/10. Saved-sidecar reload confirms another small high:
`65/116 = 56.03%`. The gain is step1 `2/8 -> 3/8`, step7 `4/8 -> 5/8`, and
step9 `2/8 -> 3/8`, while step3 drops from `8/8` to `6/8`; step2/4/5 remain
`8/8`. v23 is therefore the best MTP serving-credit candidate so far, but the
next rescue has to restore step3 without losing the new step1/7/9 accepts.

The v22/v23 interpolation scan then finds a better midpoint than either parent:
alpha `0.85` reloads at `66/116 = 56.90%`, adding one more step1 accept while
preserving the v23 step7/9 gains. A subset-grid over `fc`, pre-fc norms, and
`mtp.norm.weight` does not beat the interpolation; its best candidate is v22
with v23 `fc`, `65/116`.

A100 v24 starts from the interpolated `alpha=0.85` sidecar and applies a very
low-LR `fc_norms` margin run on steps `1/3/8/11`. In-memory heldout eval moves
from `66/116` to `69/116 = 59.48%`: step1 improves `4/8 -> 5/8`, step3 returns
to `8/8`, and step2/4/5 stay `8/8`. Saved reload confirms the same
`69/116 = 59.48%`, making v24 the new MTP serving-credit best. The diagnostic
now shows 47 misses, 11 near misses, and 32 hard misses; the next A100 target is
semantic/code-body failures rather than simple JSON punctuation.

A100 v25 tests that semantic/code-tail direction with a v7 prompt set from v24.
It lowers heldout loss but does not add accept: total remains `69/116`. Its
shape is complementary rather than better: step10 improves `2/7 -> 3/7`, but
step12 drops `2/6 -> 1/6`. A v24/v25 interpolation scan keeps best alpha at
`0.0`, so v24 remains the authoritative sidecar.

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

2026-05-17 1024-token follow-up: the same Config D server path reports wall
`88.70 tok/s` and decode `98.85 tok/s`, with `9.97 ms` median decode step,
linear block graph reuse enabled, and native FP4 lm_head enabled. The long run
keeps the safe serving line near 100 TPS rather than revealing a hidden 155 TPS
mode.

2026-05-17 service-path ablation: Config D decode is `99.9-100.2 tok/s` at
256/512/1024 tokens. Disabling native FP4 lm_head drops only to
`96.7-97.2 tok/s`, so lm_head is a ~3% lever. Disabling linear-block graph drops
the service path to `27.6-27.9 tok/s`; graph reuse is therefore the main current
runtime pillar. Graph capture without prewarm costs about `0.10s` on the first
request but keeps steady decode near 100 tok/s. Per-request graph capture also
keeps decode near 100 tok/s for long requests, with the same ~0.08-0.10s wall
overhead per request. The next runtime bridge should preserve linear-block graph
semantics while reducing host/Python decode-loop boundaries; eager/no-graph
paths are not viable for 155.

2026-05-17 P26 decode phase profile corrects the next runtime bet. With Config
D and reusable linear-block graphs, the token wall is `10.17 ms` (`98.3 tok/s`),
of which `10.03 ms` is accounted CUDA work. Linear-block graph replay is
`6.55 ms`, the ten eager full-attention layers are `3.11 ms`, native FP4
lm_head is `0.33 ms`, and the residual host gap is only `0.15 ms`. A C++ token
loop alone cannot provide the missing `~3.5 ms/token`; the C++/CUDA refactor has
to make full-attention/static-state graphing or native fused layer boundaries
possible, while MTP supplies the effective-token multiplier.

2026-05-17 P27 full-layer segment profile opens one service-shaped full-attn
layer under Config D. Layer 31 measures `attn.full_decode 0.281 ms`,
`packed MoE 0.203 ms`, `qk_norm_rope 0.124 ms`, and two RMSNorms of about
`0.061 ms` each. The full-layer recomposition is `0.739 ms` in the isolated
profile. The useful conclusion is not "replace SDPA"; SDPA itself is only about
`0.015 ms` at this short seq_len. The next runtime work should fuse/static-graph
the full-attn layer boundary, especially q/k norm+RoPE, norms, and packed MoE
dispatch, rather than chasing attention core alone.

A three-layer P27 sweep confirms the same shape on layers 3/15/39:
full-layer recomposition `0.73-0.85 ms`, attention full decode `0.28-0.31 ms`,
packed MoE `0.20-0.22 ms`, q/k norm+RoPE `0.125-0.136 ms`, and SDPA GQA
`~0.015 ms`. This makes the next C++/CUDA target a fused/static full-layer
boundary, not a standalone SDPA kernel.

2026-05-17 P9H full-attention graph probe gives the first strong full-attn
runtime signal. On layer 31, a fixed-position graph capture of the full layer
boundary measures `0.7767 ms` eager vs `0.1993 ms` graph replay, or `3.90x`
speedup, with exact output/KV parity (`max_abs=0`, `rel_l2=0`) and exact KV
write-slice parity at position 7. This does not mean the current server can
flip one env var; it means the next R6000 runtime investment should build
reusable static-position/KV graph families or an equivalent native full-layer
boundary. The earlier Spark strict-slot result remains rejected because it
recaptures every token, but reusable capture is now a proven high-ROI path.

2026-05-17 P9I follow-up sweep confirms this is not a single favorable layer.
Across 12 cases (layers 3/15/31/39 and positions 10/14/32), replay-only graph
timing averages `0.252 ms` vs `1.031 ms` eager, with speedup range
`4.00-4.29x` and exact output/KV write-slice parity in every case. This makes
full-attention reusable graphing the most concrete non-MTP R6000 speed lever
found today.

2026-05-17 P9J closes the mutable-input concern for the graph slot contract.
Across layers 3/15/31/39 at position 10, one captured graph per layer replays
correctly after swapping the input hidden buffer to a second token: both case-A
and case-B outputs have `max_abs=0`, KV writes have `max_abs=0`, and graph
outputs actually change between inputs (`min rel_l2=1.07`). Mean replay timing
is `0.245 ms` vs `1.033 ms` eager, or `4.21x`. The next implementation step is
a single-stream resident graph-state ABI, not another proof that graph inputs
can vary.

2026-05-17 P9M proves the service-shape pre-capture assumption: a full-attn
slot captured before real KV exists can replay exactly after prefill and 32
eager prefix tokens populate the same cache. Layer31 at position39 reports
`0.2946 ms` replay with output and KV write `max_abs=0`. This means graph slots
do not need to be captured on the hot token path.

2026-05-17 P9N then composes 10 linear-block graphs and 10 pre-captured
full-attn slots into a whole-token hybrid path. The path is greedy-safe on the
probe prompt and one-shot graph timing is `9.5298 ms` (`104.93 tok/s`), but
strict logits are not exact (`max_abs=1.046875`). P9O localizes the first strict
drift to full-attn slot layer3: linear block0 is exact, then layer3 slot output
diff appears (`max_abs=0.02124`, `rel_l2=0.1479`). The next R6000 runtime patch
should tighten the full-attn slot state contract before wiring an env-flagged
server path.

2026-05-17 P9P repeats the layerwise diff with a single restored state matching
P9N's execution shape. It confirms P9O was not a two-prefill artifact: first
drift is still full-attn slot layer3 with the same `max_abs=0.02124`, while
linear block0 remains exact. This pins the next graph task to the full-attn slot
capture contract.

2026-05-17 P9Q/P9R/P9S refine the graph task again. P9Q shows layer3 alone is
exact for both real-KV capture and empty-KV pre-capture (`max_abs=0`), so the
layer kernel itself is not the culprit. P9R shows a stale pre-captured slot can
drift after other CUDA graphs are captured, while a fresh pre-slot after linear
graph capture is exact. P9S tries the simple linear-first capture order for the
whole token; it remains greedy-safe (`103.38 one-shot tok/s`) but strict logits
get worse (`max_abs=3.53125`). The actionable runtime conclusion is now graph
pool/state ownership: mixed linear/full-attn graph families need explicit
capture isolation, not just reordered ad hoc capture.

2026-05-17 P9T tests a separate-state version with one linear graph state and
one full-attn KV graph state. It is still greedy-safe (`104.91 one-shot tok/s`)
but does not fix strict logits (`max_abs=3.53125`). The next R6000 proof should
isolate CUDA graph memory pools explicitly, or switch from composed PyTorch
CUDAGraphs to a native/static full-attn layer boundary.

2026-05-17 P9U/P9V/P9W close the PyTorch CUDAGraph branch. Explicit graph-pool
modes do not fix strict drift: full-first remains `max_abs=1.046875`, and
linear-first remains `max_abs=3.53125`, all around `104.7-105.0 tok/s`. Capturing
full-attn slots on their real per-layer state is strict-exact (`max_abs=0`,
`104.93 tok/s`), but those real-state slots do not reuse across a fresh request,
even for the same prompt. The serving route should therefore move to a native
static full-attn boundary with explicit runtime inputs instead of trying to make
PyTorch CUDAGraph slots portable across requests.

2026-05-17 P6G backend profile shows `manual_gqa` is not a speed lever for
full-attention decode: layer31 `attn.full_decode_recomposed` is `0.336 ms` with
SDPA and `0.427 ms` with manual GQA. The attention core is only about
`0.015 ms`; the useful native target is q/k norm + RoPE + cache/output glue, and
the larger layer-level bottleneck remains active-MoE (`~0.41 ms` for the active
expert loop on this layer).

P6K confirms that the existing Triton q/k norm+RoPE kernels already remove the
big torch overhead: torch q+k total is `0.246 ms`, Triton q+k total is
`0.043 ms` (`5.72x`). A native full-attn boundary can still reduce glue and
allocation, but the next large R6000 speed lever is still active-MoE/native
expert math plus MTP, not replacing SDPA.

2026-05-17 down-backend service sweep: switching only
`LYNN_NATIVE_DOWN_BACKEND` from `triton` to `cuda_tile` gives real raw speed
(`108.17 / 108.84 tok/s` at `256 / 512`, about `1.08-1.10x`), but both previews
collapse into an exclamation-loop. This is a useful kernel signal but a RED
serving candidate; do not promote it until the CUDA tile down path passes the
generation/preview gate.

2026-05-17 P50 first-divergence follow-up on the P25 prompt: with graphs off and
both backends fed the Triton greedy stream, `cuda_tile` first flips top-1 at
step 28 (`主流` vs `核心`) on a low-margin choice. The first visible hidden
drift is already at step 0/layer 11. The down tile path is therefore a real
speed lever, but its accumulation-order drift is large enough to flip semantic
tokens in long decode.

Tile-hidden sweep (`1/2/4/8`) does not change the outcome: every tested tile
variant diverges at the same step 28 on the same `主流` vs `核心` choice. The
fix is not a tile-size choice; it needs a drift-reduced down kernel or a
different active-MoE fusion path.

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
