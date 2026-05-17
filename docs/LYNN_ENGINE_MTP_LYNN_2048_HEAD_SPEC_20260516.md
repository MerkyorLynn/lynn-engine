# Lynn Engine MTP Head Spec: 2048-Hidden Qwen3-Next Style

Date: 2026-05-16

## Decision

Lynn 27B needs a **Lynn-owned 2048-hidden MTP predictor**. The community
Qwen3.6-27B MTP sidecar is useful as a shape/architecture reference, but it is
not a direct initializer because it is built for hidden size 5120.

2026-05-17 update: the official Qwen3.6-35B-A3B MTP sidecar is different from
the earlier 5120-hidden community sidecar. It has 2048-hidden `mtp.*` tensors
and is now available as a Lynn warm-start asset after expert-dimension
alignment.

```text
reports/mtp/a100_qwen36_a3b_mtp_sidecar_shape_audit_v2_20260517_0935.json
decision: GREEN

reports/mtp/a100_qwen36_a3b_mtp_warm_start_mapping_aligned_20260517_0938.json
decision: GREEN
direct_copy: 16
slice_first_dim: 3
```

Aligned sidecar:

```text
/mnt/data2/lynn-a100/models/mtp_sidecars/qwen36-35b-a3b-mtp-lynn-warm-start-aligned/mtp.safetensors
```

Forward-smoke report:

```text
reports/mtp/a100_mtp_forward_smoke_20260517_110159.json
decision: GREEN
mtp_logits_finite: true
mtp_logits_shape: [1, 248320]
argmax_match: false
base_next_argmax: {"token_id": 4754, "text": "{\""}
mtp_draft_argmax: {"token_id": 98175, "text": "十二"}
```

FC-only train-smoke report:

```text
reports/mtp/a100_mtp_fc_train_smoke_20260517_111251.json
decision: GREEN
mode: fc_only_single_prompt_ce_to_base_argmax
loss_before: 8.9365
loss_after: 0.0
argmax_match_after: true
weights_saved: false
```

FC-only calibration report:

```text
reports/mtp/a100_mtp_fc_calibration_teacher_clean_v2_20260517_113413.json
reports/mtp/a100_mtp_fc_calibration_saved_teacher_clean_v2_20260517_113808.json
reports/mtp/a100_mtp_fc_calibration_heldout_v1_20260517_113808.json
decision: GREEN
case_count: 12
accept_before: 0/12
accept_after: 12/12
mean_loss_before: 11.4534
mean_loss_after: 0.0076
trained_sidecar:
/mnt/data2/lynn-a100/models/mtp_sidecars/qwen36-35b-a3b-mtp-lynn-fc-teacherclean-v2-20260517_113808/mtp.safetensors
heldout_accept: 5/8
```

Weighted fc-only update:

```text
reports/mtp/a100_mtp_fc_calibration_weighted_v2_20260517_115058.json
train_accept: 0/22 -> 21/22

reports/mtp/a100_mtp_fc_calibration_weighted_v2_heldout_20260517_115058.json
heldout_accept: 7/8

reports/mtp/a100_mtp_fc_calibration_weighted_math_v3_20260517_115509.json
train_accept: 0/26 -> 26/26

reports/mtp/a100_mtp_fc_calibration_weighted_math_v3_heldout_20260517_115509.json
heldout_accept: 8/8
trained_sidecar:
/mnt/data2/lynn-a100/models/mtp_sidecars/qwen36-35b-a3b-mtp-lynn-fc-weighted-math-v3-20260517_115509/mtp.safetensors
```

Iterative accept boundary:

```text
reports/mtp/a100_mtp_iterative_accept_weighted_math_v3_heldout16_20260517_115852.json
events: 94
accepted: 8
accept_rate: 8.51%
```

Current saved-sidecar best:

```text
reports/mtp/a100_mtp_iterative_train_interp_front014_v15_20260517_131902.json
reports/mtp/a100_mtp_saved_sidecar_eval_interp_v15_20260517_132219.json
decision: AMBER
best_label: v15
best_accept: 56/116 = 48.28%
sidecar:
/mnt/data2/lynn-a100/models/mtp_sidecars/qwen36-35b-a3b-mtp-lynn-iter-interp-front014-v15-20260517_131902/mtp.safetensors

reports/mtp/a100_mtp_iterative_train_v15_weaksteps_v16_20260517_132509.json
reports/mtp/a100_mtp_saved_sidecar_eval_v15_v16_20260517_132954.json
decision: AMBER
best_label: v16
best_accept: 57/116 = 49.14%
sidecar:
/mnt/data2/lynn-a100/models/mtp_sidecars/qwen36-35b-a3b-mtp-lynn-iter-v15-weaksteps-v16-20260517_132509/mtp.safetensors

reports/mtp/a100_mtp_interpolate_v16_v14_fine_20260517_133151.json
decision: AMBER
best_alpha: 0.0
best_accept: 57/116 = 49.14%

reports/mtp/a100_mtp_iterative_train_v16_late_mtplow_v17_20260517_133334.json
reports/mtp/a100_mtp_saved_sidecar_eval_v16_v17_20260517_134040.json
decision: AMBER
best_label: v17
best_accept: 58/116 = 50.00%

reports/mtp/a100_mtp_iterative_train_v17_weakall_mtplow_v18_20260517_134200.json
reports/mtp/a100_mtp_saved_sidecar_eval_v17_v18_20260517_135418.json
decision: AMBER
best_label: v18
best_accept: 59/116 = 50.86%
sidecar:
/mnt/data2/lynn-a100/models/mtp_sidecars/qwen36-35b-a3b-mtp-lynn-iter-v17-weakall-mtplow-v18-20260517_134200/mtp.safetensors

reports/mtp/mtp_fc_calibration_prompts_v4_targeted_format_tail.json
reports/mtp/a100_mtp_iterative_train_v18_targeted_v4_fcnorms_v19_20260517_135834.json
reports/mtp/a100_mtp_saved_sidecar_eval_v18_v19_20260517_140826.json
decision: AMBER
best_label: v19
best_accept: 60/116 = 51.72%
sidecar:
/mnt/data2/lynn-a100/models/mtp_sidecars/qwen36-35b-a3b-mtp-lynn-iter-v18-targeted-v4-fcnorms-v19-20260517_135834/mtp.safetensors

reports/mtp/a100_mtp_iterative_train_v19_step1_late_mtplow_v20_20260517_141105.json
reports/mtp/a100_mtp_saved_sidecar_eval_v19_v20_20260517_142412.json
decision: AMBER
best_label: v19
v20_accept: 60/116 = 51.72%

reports/mtp/a100_mtp_interpolate_v19_v14_fine_20260517_142753.json
decision: AMBER
best_alpha: 0.0
best_accept: 60/116 = 51.72%

reports/mtp/a100_mtp_subset_grid_v19_v14_20260517_142927.json
decision: AMBER
best_groups_from_b: []
best_accept: 60/116 = 51.72%

reports/mtp/a100_mtp_iterative_train_v19_step1only_fcnorms_v21_20260517_143251.json
reports/mtp/a100_mtp_saved_sidecar_eval_v19_v21_20260517_143629.json
decision: AMBER
best_label: v19
v21_accept: 59/116 = 50.86%

reports/mtp/a100_mtp_v19_saved_sidecar_diagnostic_20260517_144739.json
decision: AMBER
best_label: v19
best_accept: 60/116 = 51.72%
near_misses_label_rank_le_5: 18
hard_misses_label_rank_gt_20: 32
largest miss buckets: semantic_token 21, stop_token 18, generic_structured_key 8,
json_punctuation 6

reports/mtp/mtp_fc_calibration_prompts_v5_margin_nearmiss.json
reports/mtp/a100_mtp_iterative_train_v19_margin_v5_fcnorms_v22_20260517_145317.json
reports/mtp/a100_mtp_saved_sidecar_eval_v19_v22_20260517_145645.json
decision: GREEN-CREDIT
best_label: v22
best_accept: 64/116 = 55.17%
sidecar:
/mnt/data2/lynn-a100/models/mtp_sidecars/qwen36-35b-a3b-mtp-lynn-iter-v19-margin-v5-fcnorms-v22-20260517_145317/mtp.safetensors

reports/mtp/a100_mtp_v19_v22_saved_sidecar_diagnostic_20260517_150044.json
v22_miss_count: 52
v22_near_misses_label_rank_le_5: 16
v22_largest_miss_buckets: semantic_token 22, generic_structured_key 7,
json_punctuation 7, stop_token 7, special_token 5

reports/mtp/mtp_fc_calibration_prompts_v6_step1_step10_rescue.json
reports/mtp/a100_mtp_iterative_train_v22_margin_v6_rescue_v23_20260517_150922.json
reports/mtp/a100_mtp_saved_sidecar_eval_v22_v23_20260517_151100.json
decision: GREEN-CREDIT
best_label: v23
best_accept: 65/116 = 56.03%
sidecar:
/mnt/data2/lynn-a100/models/mtp_sidecars/qwen36-35b-a3b-mtp-lynn-iter-v22-margin-v6-rescue-v23-20260517_150922/mtp.safetensors
```

The weighted-math v3 sidecar is a GREEN first-token draft candidate, but it is
not yet a speculative decode candidate. It accepts the first token for each
heldout prompt and then diverges on later positions. Stage 1 has now moved from
prompt-boundary training into saved iterative sidecar repair. v15 is the first
saved sidecar to approach the 55% serving-credit bar, with steps 2/3/4/5 at
8/8 and the remaining weakness concentrated at steps 1/8/9/12/15. v16 is a
low-LR `fc_norms` repair run from v15 targeting only those weak positions; it
adds one saved heldout accept by moving step1 from `1/8` to `2/8` without
damaging the restored step2/3/4/5 band. A fine interpolation back toward the
v14 step1/3 specialist does not help; even small alpha values start hurting
front positions before improving the late tail. The next repair direction is
therefore low-LR late-step MTP-layer training from v16, not more v14 blending.
v17 confirms that this path can move the saved gate again, but only by one
late-tail accept so far: step12 moves from `0/6` to `1/6`, total `58/116`.
v18 continues from v17 with lower LR and a combined front+late weak-step
curriculum; saved eval confirms one more accept, moving step0 from `3/8` to
`4/8` for total `59/116`. Step1 and the late tail stay flat, so the next
useful move is not another blind low-LR weak-step sweep. Add targeted
calibration coverage or train a small specialist/merge candidate for step1 and
late-tail key positions before spending more A100 cycles. v19 adds targeted v4
format-tail calibration and trains `fc_norms` from v18; it confirms one more
saved accept by moving step9 from `1/8` to `2/8`, reaching `60/116`. The
remaining 55% gap is now four accepts, concentrated in step1 and late-tail
positions. Continue with a step1/late specialist or merge route, not another
broad v4 repeat. v20 tests a direct `fc_mtp_layer` specialist on
steps `1/8/9/11/12/13/14/15` from v19 and reloads at the same `60/116`; it does
not damage the restored band, but it also does not add accept. That closes this
simple specialist shape. The next A100 move should be merge/selection or a
different trainable surface for step1. v19/v14 merge tests then close the old
v14 rescue route: interpolation is best at alpha `0.0`, and subset grid is best
with no v14 groups. Importing v14 `fc` drops to `41/116`, so the v14
step1-specialist direction is not composable with v19 by simple linear or
small-block merge. v21 then tests step1-only `fc_norms` from v19; it fails to
move step1 and drops the saved total to `59/116` by losing the v19 step9 gain.
The next path needs a different target construction, not another low-LR run on
the same heldout-shaped step positions. The v19 diagnostic now identifies the
new target construction: 18 of 56 misses are near misses where the teacher token
is still rank <=5, including stop-token over-selection (`<|im_end|>` vs
newline/end-of-text), JSON punctuation splits (`":` vs `":"`), and generic
structured-key substitutions (`action`/`city` over the desired key/value). A100
v22 therefore switches from plain CE to CE plus hard-negative margin on a new
v5 near-miss calibration set:

```text
scripts/a100_mtp_saved_sidecar_diagnostic.py
reports/mtp/mtp_fc_calibration_prompts_v5_margin_nearmiss.json
loss_mode: ce_margin
target: make label logit beat the current hard negative by margin
```

v22 confirms the new objective is useful. In-memory eval moves from `60/116`
to `64/116`, and the saved-sidecar reload gate keeps v22 as best at
`64/116 = 55.17%`. The gain comes from step0 `4/8 -> 6/8`, step7 `3/8 -> 4/8`,
step11 `2/6 -> 3/6`, step12 `1/6 -> 2/6`, and step15 `1/5 -> 2/5`, while the
already-restored step2/3/4/5 band remains `8/8`. Step10 regresses from `4/7`
to `2/7`, so v22 is a serving-credit candidate, not yet a final promotion
candidate. The next MTP move should preserve v22 and recover step10/step1
without losing the new step0/late-tail accepts.

The v22 diagnostic shows the remaining easy wins are still near misses:
16 misses keep the teacher label in rank <=5. The most obvious low-margin cases
are JSON punctuation (`":` vs `":"`, `","` vs `",`) and step1 `<think>` versus
format-open tokens. The hard failures are now concentrated in Python code-body
continuation and semantic Chinese MoE tails, so the next rescue should be
targeted rather than a broad replay of v5.

v23 is a narrow continuation from v22 using v6 prompts and margin loss on
steps 1/7/10. Saved reload confirms `65/116 = 56.03%`, with gains at step1,
step7, and step9. It also drops step3 from `8/8` to `6/8`, so v23 is the best
serving-credit candidate but not the final sidecar. The next route should
restore step3 while preserving the v23 front/format gains.

The v22/v23 interpolation scan finds the next clean serving-credit step:
alpha `0.85` reaches `66/116 = 56.90%` and saved reload confirms it. A
subset-grid between the same parents does not beat it; the best group swap is
only `65/116`.

v24 starts from the interpolated sidecar and applies a conservative `fc_norms`
margin run on steps `1/3/8/11`. In-memory heldout eval reaches
`69/116 = 59.48%`, with step1 `5/8`, step3 restored to `8/8`, and steps
2/4/5 still `8/8`. Saved reload confirms the same `69/116 = 59.48%`, making
v24 the new authoritative serving-credit sidecar. The remaining miss profile is
47 misses total: 11 near misses and 32 hard misses, dominated by semantic/code
tokens rather than the earlier low-margin punctuation cases.

v25 adds a v7 semantic/code-tail calibration set from v24. It lowers eval loss
but does not add heldout accept: `69/116` before and after. The by-step trade is
useful for diagnosis but not promotion: step10 improves to `3/7`, while step12
drops to `1/6`. Interpolating v24 and v25 keeps best alpha at `0.0`, so v24
remains the sidecar to wire into serving-credit experiments.

v26 tests a narrower `fc_mtp_layer` repair from v24 on the v7 semantic/code-tail
set, using steps `10/12/13/14/15`, `ce_margin`, LR `2e-6`, and only two train
steps. It lowers loss but does not move accept: eval remains `68/116 = 58.62%`
inside the train script. This closes the simple "unfreeze the MTP layer on tail
cases" route for now; the next A100 MTP move should use new target construction
or a separate specialist/merge, not another low-LR v7 replay.

P107 moves the v24 sidecar from script-only eval into the resident runner
shadow path. `LYNN_MTP_SIDECAR=/path/to/mtp.safetensors` loads the sidecar and
`LYNN_MTP_SHADOW_VERIFY=1` records draft-vs-base argmax matches inside
`LynnIncrementalRunner.generate()` without changing emitted tokens. The first
A100 P107 gates are:

```text
reports/mtp/a100_p107_mtp_shadow_v24_structured_v10_top6_20260517_154801.json
structured_v10_top6 + v24: 68/116 = 58.62%
draft_tps: 83.42
max_one_token_speculative_multiplier: 1.586x

reports/mtp/a100_p107_mtp_shadow_v24_structured_v16_damped075_20260517_154859.json
structured_v16_top6_damped075 + v24: 68/116 = 58.62%
draft_tps: 79.00
max_one_token_speculative_multiplier: 1.586x

reports/mtp/r6000_p107_mtp_shadow_v24_w4a8_v2_20260517_155539.json
R6000 W4A8 NVFP4 v2 + v24: 61/121 = 50.41%
draft_tps: 130.51
max_one_token_speculative_multiplier: 1.504x

reports/mtp/r6000_mtp_iterative_train_w4a8_v2_v24_v7_fcnorms_v27_20260517_155838.json
R6000 W4A8-aware v27 proxy eval: 63/116 -> 64/116

reports/mtp/r6000_p107_mtp_shadow_v27_w4a8_v2_bf16_lmhead_20260517_160100.json
R6000 v27 with BF16 lm_head: 64/115 = 55.65%

reports/mtp/r6000_p107_mtp_shadow_v27_w4a8_v2_20260517_160001.json
R6000 v27 with native FP4 lm_head: 61/121 = 50.41%

reports/mtp/r6000_mtp_iterative_train_w4a8_v2_v27_native_label_v28_20260517_160254.json
native-label v28 proxy eval: 65/121 -> 67/121

reports/mtp/r6000_p107_mtp_shadow_v28_native_label_w4a8_v2_20260517_160422.json
R6000 v28 with native FP4 lm_head: 62/121 = 51.24%

reports/mtp/r6000_mtp_iterative_train_w4a8_v2_v28_native_label_v29_20260517_160531.json
native-label v29 proxy eval: 67/121 -> 68/121

reports/mtp/r6000_p107_mtp_shadow_v29_native_label_w4a8_v2_20260517_160659.json
R6000 v29 with native FP4 lm_head: 62/121 = 51.24%

reports/mtp/r6000_mtp_iterative_train_w4a8_v2_v29_runtime_native_v30_20260517_160917.json
runtime-env collection v30 proxy eval: 66/121 -> 67/121

reports/mtp/r6000_p107_mtp_shadow_v30_runtime_native_w4a8_v2_20260517_161038.json
R6000 v30 with native FP4 lm_head: 62/121 = 51.24%

reports/mtp/r6000_mtp_iterative_train_w4a8_v2_v29_fake_native_v31_20260517_161411.json
fake-native-FP4 lm_head v31 proxy eval: 64/121 -> 64/121

reports/mtp/r6000_p108_lm_head_native_surrogate_parity_w4a8_v2_20260517_162000.json
P108 fake-native lm_head parity: 111/118 top-1 = 94.07%, 118/118 top-5
P108 BF16 lm_head parity: 109/118 top-1 = 92.37%, 118/118 top-5

reports/mtp/r6000_p108_lm_head_native_surrogate_parity_v2_w4a8_20260517_162436.json
P108 v2 activation-aware fake-native parity: 114/118 top-1 = 96.61%, 118/118 top-5
P108 v2 weight-only fake-native parity: 113/118 top-1 = 95.76%, 118/118 top-5

reports/mtp/r6000_mtp_iterative_train_w4a8_v2_v29_fake_act_v32_20260517_162607.json
activation-aware fake-native v32 proxy eval: 60/121 -> 59/121

reports/mtp/r6000_mtp_iterative_train_w4a8_v2_v29_fake_act_rankflip_v33_20260517_163654.json
rank-flip filtered v33 proxy eval: 60/121 -> 61/121

reports/mtp/r6000_p107_mtp_shadow_v33_rankflip_w4a8_v2_20260517_163823.json
R6000 v33 with native FP4 lm_head: 63/121 = 52.07%

reports/mtp/r6000_mtp_iterative_train_w4a8_v2_v33_fake_act_rankflip_v34_20260517_163912.json
rank-flip filtered v34 proxy eval: 61/121 -> 63/121

reports/mtp/r6000_p107_mtp_shadow_v34_rankflip_w4a8_v2_20260517_164039.json
R6000 v34 with native FP4 lm_head: 65/121 = 53.72%

reports/mtp/r6000_mtp_iterative_train_w4a8_v2_v34_fake_act_rankflip_v35_20260517_164129.json
rank-flip filtered v35 proxy eval: 63/121 -> 62/121

reports/mtp/mtp_fc_calibration_prompts_v8_native_semantic_code.json
semantic/code/native-format calibration set for miss-not-in-topk cases

reports/mtp/r6000_mtp_iterative_train_w4a8_v2_v34_v8_missnotopk_v36_20260517_164518.json
v8 miss-not-in-topk v36 proxy eval: 63/121 -> 63/121
```

This is a GREEN-CREDIT serving-shadow result, not a final TPS claim. It proves
the v24 sidecar survives the real runner hidden-state boundary and gives R6000
a concrete sidecar to test under W4A8 NVFP4 v2. The first R6000 test is AMBER:
the sidecar runs and is fast enough to measure, but quantized-runtime accept is
below the 55% credit bar. It also shows v16 is not better than v10 for MTP
credit on this heldout gate; v16's value remains quality conservatism, not
higher draft acceptance.

The v27/v28 follow-up narrows the failure boundary. A W4A8-aware `fc_norms`
repair can cross the credit bar when `p107` uses the BF16 lm_head, but the
credit disappears under `LYNN_NATIVE_FP4_LM_HEAD=1`. Collecting labels from the
native FP4 lm_head recovers one native accept in v28, so the direction is valid.
v29 improves the proxy again but leaves native `p107` flat, so the current
native-label/BF16-backprop proxy is saturated. The next sidecar training loop
must either collect full runtime-native hidden states/labels or train against a
native-lm-head-aware target while retaining a BF16 differentiable projection for
backprop.

v30 and v31 test both ideas in minimal form. Runtime-env collection changes the
case distribution but does not move native accept. The first fake-native-FP4
lm_head surrogate is memory-safe after row chunking, but it lowers loss without
adding accept. P108 explains the miss: the fake-native head has perfect top-5
containment against the real native `_scaled_mm` head, but still only reaches
`94.07%` top-1 parity, and the few flips land on structured/code first-token
margin cases (`{`/`{"`, code fence/`<think>`, `def`/`import`, JSON repair
quote/colon choices). The next useful route is therefore not another low-LR
replay; it needs either broader native-labeled calibration coverage or an
activation-aware native lm_head surrogate that pushes those rank-1/rank-2
flips across the final boundary.

P108 v2 validates the activation-aware surrogate itself: adding FP8 scale
rounding plus fake activation quantization raises top-1 parity against native
`_scaled_mm` to `96.61%`, with native top-5 still fully contained. v32 then
tests whether this improved boundary is sufficient as a small continuation
training target from v29. It is not: eval accept moves `60/121 -> 59/121`
despite lower loss. Treat `fake_native_fp4_act` as a validated probe and a
future loss component, but not as a standalone continuation recipe.

v33/v34 add a more selective use of that probe: train only the cases where the
native label is already in the sidecar top-k but not top-1. That rank-flip
filter produces the first native R6000 MTP improvement after the v29 plateau:
`62/121 -> 63/121 -> 65/121`. v35 regresses, so v34 is the current best
warm-start sidecar for native W4A8 serving-shadow credit. The next jump to
GREEN-CREDIT needs either more top-k rank-flip cases or new calibration that
brings semantic/code labels into top-5 before rank training.

v36 tries the second route with a targeted v8 semantic/code prompt set and
`miss_not_in_topk` filtering. It is a useful negative: 67 hard cases train with
lower loss, but heldout proxy accept stays `63/121`. The remaining gap is not
fixed by a tiny `fc_norms` pass alone; use a wider surface (`fc_mtp_layer` or a
merged specialist) or a larger native-labeled set before re-running P107.

Report:

```text
reports/a100/a100_mtp_sidecar_shape_audit_readme.json
decision: RED
reason: no tensor shape contains base hidden_size
base hidden: 2048
sidecar hidden: 5120
```

## Base Lynn Shapes

Sampled from the BF16 final artifact, full-attention layer 3:

```text
hidden_size: 2048
q_proj:     [8192, 2048]
k_proj:     [512, 2048]
v_proj:     [512, 2048]
o_proj:     [2048, 4096]
q_norm:     [256]
k_norm:     [256]
expert inter size: 512
```

The MTP predictor should preserve these projection conventions, not Qwen3.6's
5120-hidden tensor shapes.

## Target Tensor Contract

First Lynn-owned MTP head:

```text
mtp.fc.weight                            [2048, 4096]
mtp.norm.weight                          [2048]
mtp.pre_fc_norm_embedding.weight         [2048]
mtp.pre_fc_norm_hidden.weight            [2048]
mtp.layers.0.input_layernorm.weight      [2048]
mtp.layers.0.post_attention_layernorm.weight [2048]
mtp.layers.0.self_attn.q_norm.weight     [256]
mtp.layers.0.self_attn.k_norm.weight     [256]
mtp.layers.0.self_attn.q_proj.weight     [8192, 2048]
mtp.layers.0.self_attn.k_proj.weight     [512, 2048]
mtp.layers.0.self_attn.v_proj.weight     [512, 2048]
mtp.layers.0.self_attn.o_proj.weight     [2048, 4096]
```

FFN options:

```text
compact v0:
  gate_proj [512, 2048]
  up_proj   [512, 2048]
  down_proj [2048, 512]

wider v1 if accept rate is weak:
  gate_proj [2048-4096, 2048]
  up_proj   [2048-4096, 2048]
  down_proj [2048, 2048-4096]
```

Use the compact v0 first. Lynn's base MoE experts already use 512 intermediate
channels, so a 512-intermediate dense predictor is the cheapest faithful smoke.

## Parameter Budget

Approximate compact v0:

```text
fc:        8.4M params
attention: ~27.3M params
FFN:       ~2.1M params
norms:     negligible
total:     ~38M params
BF16 size: ~76 MB
```

This is much smaller than the Qwen3.6 5120-hidden sidecar (~811 MB), which is
expected because Lynn's hidden size is much smaller and the first predictor can
use compact dense FFN.

## Training Plan

Stage 0: wiring smoke

```text
initialize MTP head from matching Lynn base full-attn layer where shape matches
initialize dense FFN from shared-expert projections where possible
freeze base model
num_speculative_tokens = 2
```

2026-05-17 status: Stage 0 shape mapping, forward smoke, and fc-only
calibration training are complete for the official 2048-hidden Qwen3.6-35B-A3B
sidecar. Weighted fc-only calibration now reaches 8/8 on the heldout first-token
gate, but iterative 16-token probing reaches only 8/94 accepts. This is a
useful first-token bridge, not a complete draft model. Stage 1 now needs
token-position training and likely partial MTP layer unfreeze before serving
integration.

Stage 1: short head-only training

```text
inputs: calibration prompts + structured JSON/code/tool/no-think prompts
loss: draft token cross entropy for token+1 and token+2
body: frozen
target: verified greedy output unchanged, accept rate enough for >1.1x single-stream
```

Stage 2: combined W4A8 + MTP

```text
only after W4A8 generation gate is AMBER/GREEN
evaluate accept rate under W4A8 active-MoE runtime, not BF16-only
```

## What Not To Do

- Do not download or transplant 5120-hidden Qwen3.6 MTP weights into Lynn 27B.
- Do not treat simple Linear NEXTN as the quality route; it is wiring-only.
- Do not start batched MTP until single-stream verification is correct.
- Do not use MTP to hide a RED W4A8 generation gate.

## Open Questions

1. Whether compact 512-intermediate FFN is enough for 70%+ accept rate.
2. Whether a wider dense FFN is needed before spending A100 time.
3. Whether MTP should be trained on BF16 base first, then W4A8, or directly on
   the W4A8 Recovery candidate.

Current default answer:

```text
W4A8 Recovery first. Train compact Lynn-owned MTP head after W4A8 generation
gate reaches AMBER/GREEN.
```
