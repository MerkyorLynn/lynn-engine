# A100 W4A8 + MTP Preflight

Date: 2026-05-16

## Executive Decision

The A100 campaign should continue on the **W4A8 + MTP/NEXTN** route.

The first A100 gates separate three facts that were easy to blur:

1. **The BF16 artifact is complete and loadable.**
2. **Real-prompt W4A8 active-MoE drift is trainable, not route-killing.**
3. **MTP/NEXTN is not present as checkpoint weights and needs a Lynn-owned head implementation.**

## Artifact Integrity

Model:

```text
/mnt/data/lynn-a100/models/lynn-27b-variable-recovery-step5000-bf16-final
```

Inventory:

```text
shards: 1026 / 1026
missing: 0
present shard bytes: 63,855,985,856
ready_for_tensor_load: true
```

Runtime loadability:

```text
resident BF16 load peak: 59.10 GiB on one A100-80G
load + 4-layer real-prompt gate elapsed: 28.0s
load + 40-layer real-prompt gate elapsed: 125.6s
```

## W4A8 Gate Results

### Synthetic Stress Gate

Script:

```text
scripts/a100_w4a8_mtp_preflight.py
```

Report:

```text
reports/a100/a100_w4a8_mtp_preflight_4layer.json
```

Result:

```text
decision: RED on synthetic stress
max_gateup_rel_l2: 3.57%
max_full_rel_l2: 3.99%
min_gateup_cosine: 0.999364
min_full_cosine: 0.999207
```

Interpretation:

The synthetic `wide` distribution is a harsh stress case. It is useful for
finding the failure mode, but it should not be used alone to reject W4A8.
The synthetic outlier case was less damaging than the wide Gaussian case.

### Real Prompt 4-Layer Gate

Script:

```text
scripts/a100_w4a8_real_prompt_gate.py
```

Report:

```text
reports/a100/a100_w4a8_real_prompt_gate_4layer.json
```

Result:

```text
decision: GREEN
layers: 4, 16, 28, 36
prompts: 6
all_gateup_relaxed: true
all_full_relaxed: true
max_gateup_rel_l2: 2.54%
max_full_rel_l2: 2.99%
min_gateup_cosine: 0.999678
min_full_cosine: 0.999554
```

Interpretation:

On real Lynn prompts, W4A8 active-MoE drift stays inside the relaxed gate for
the representative layer set. This is the first green light for A100 Recovery.

### Real Prompt 40-Layer Gate

Report:

```text
reports/a100/a100_w4a8_real_prompt_gate_40layer.json
```

Result:

```text
decision: AMBER
layers: 0..39
prompts: 6
all_gateup_relaxed: true
all_full_relaxed: false
max_gateup_rel_l2: 2.88%
max_full_rel_l2: 3.67%
min_gateup_cosine: 0.999588
min_full_cosine: 0.999371
```

Worst layers:

```text
L20: full_rel_l2 3.67%, gateup_rel_l2 2.86%
L26: full_rel_l2 3.47%, gateup_rel_l2 2.81%
L24: full_rel_l2 3.43%, gateup_rel_l2 2.85%
L12: full_rel_l2 3.36%, gateup_rel_l2 2.88%
L23: full_rel_l2 3.31%, gateup_rel_l2 2.78%
L32: full_rel_l2 3.31%, gateup_rel_l2 2.82%
```

The worst cases are mostly structured prompts such as JSON and code. That
matches the generation-level P105 observation that W4A8 causes late/local
greedy divergences rather than immediate collapse.

## Training Interpretation

W4A8 should **not** be promoted directly on the current artifact.

W4A8 should **continue** as the primary A100 training route because:

- gate/up W4A8 is clean across all sampled real-prompt layers;
- full-active drift is near the relaxed gate, not catastrophic;
- cosine remains high even at the worst layer;
- the failure mode is localized to down/intermediate activation margin.

The Recovery target is therefore:

```text
primary: down/intermediate activation rounding margin
secondary: structured-output prompts (JSON/code/tool-call)
not primary: router
not primary: gate/up projection
```

## MTP/NEXTN Preflight

The BF16 artifact config contains:

```text
mtp_num_hidden_layers = 1
mtp_use_dedicated_embeddings = false
```

But inventory found:

```text
artifact_mtp_like_key_count = 0
```

The installed `transformers.models.qwen3_5_moe` package does not expose a
named MTP/NEXTN/draft-head class for this model. The modeling file contains
some MTP strings, but there is no usable checkpoint head to load.

Decision:

```text
MTP/NEXTN needs a Lynn-owned head implementation or a compatible upstream port.
Do not assume this is only a missing tensor problem.
```

Recommended first MTP shape:

```text
base model: frozen or mostly frozen after W4A8 Recovery stabilizes
draft head: qwen3_next_mtp-style one transformer predictor layer
embedding/lm_head: share with the base model
first target: num_speculative_tokens=2, single-stream smoke
fallback only: simple Linear NEXTN head
```

Reason:

```text
Community Qwen3.6 MTP implementations and vLLM support point to a transformer
predictor layer as the practical contract. A single Linear head is useful only
as a cheap wiring smoke; it should not be the quality target.
```

## Next Actions

1. Start W4A8 Recovery short run on A100.
2. Bias calibration/eval prompts toward JSON, code, tool-call, and no-think guard.
3. Track the 40-layer gate as a fast training metric.
4. Build MTP/NEXTN head prototype in parallel, but do not combine serious MTP training with W4A8 until W4A8 Recovery improves or preserves the 40-layer gate.

## Renewal Signal

Renewal is currently justified if at least one short Recovery checkpoint moves:

```text
max_full_rel_l2: 3.67% -> <= 3.0%
or
P105 64-token divergence moves later / exact count improves
```

If neither moves after short Recovery attempts, pause and redesign the adapter
target before burning more A100 time.
