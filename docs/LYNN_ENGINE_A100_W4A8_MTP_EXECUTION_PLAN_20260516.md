# A100 Execution Plan: W4A8 Recovery + MTP/NEXTN

Date: 2026-05-16

## Current A100 Environment

Machine:

```text
2 x NVIDIA A100-SXM4-80GB
torch 2.6.0+cu124
```

Available Python packages:

```text
torch:        yes
transformers: yes
peft:         yes
accelerate:   yes
datasets:     yes
safetensors:  yes
deepspeed:    no
trl:          no
```

Implication: start with lightweight PEFT / custom probes. Do not block on
installing a large training framework until the first W4A8 gates prove the
artifact is trainable.

## Workstreams

### Stream 1: W4A8 Recovery

Purpose:

```text
Adapt the model to E4M3-per16 active expert activation rounding.
```

Why first:

- P104 active-MoE tensor gate is AMBER, not red.
- P105 generation gate is AMBER, with late same-prefix divergences.
- W4A8 unlocks R6000 FP8 x FP4 active-MoE and Spark FP8 mirror.

Initial target:

```text
P105 24-token gate: gateup/full 6/6 exact
P105 64-token gate: no divergence before token 48, then push to 6/6 exact
```

Then run:

```text
6-prompt smoke
strict tool-call
no-think loop guard
V8/V9 retention
```

### Stream 2: MTP/NEXTN

Purpose:

```text
Add a serving multiplier after the base W4A8 behavior is stable.
```

The config has:

```text
mtp_num_hidden_layers = 1
mtp_use_dedicated_embeddings = false
```

but the current BF16 artifact inventory found no MTP/NEXTN/draft tensors. This
means the architecture has a slot but the artifact still needs head weights and
training.

A100 environment preflight adds one more constraint: the installed
`transformers.models.qwen3_5_moe` implementation ignores `mtp.*` weights as
unexpected and does not expose a draft/NEXTN head class. So MTP is **not** just
missing checkpoint tensors. It needs either:

1. a Lynn-owned MTP head implementation in engine/training code; or
2. a compatible NVIDIA/Qwen MTP implementation ported into this model class.

This makes MTP a real engineering stream. It remains valuable, but W4A8
Recovery is the faster first training target.

Priority:

```text
prepare in parallel
train seriously after W4A8 Recovery is stable
```

Reason: if W4A8 drift and MTP draft error are introduced at the same time,
debugging becomes needlessly tangled.

## First 24 Hours After BF16 Transfer Completes

### 1. Integrity and inventory

```text
scripts/a100_model_inventory.py
verify 1026/1026 shards
confirm no MTP/NEXTN tensors
record disk budget
```

### 2. W4A8 fake-quant forward dry-run

Use the BF16 final as reference and inject the same E4M3-per16 activation
rounding used by P104/P105.

Start small:

```text
single active-MoE layer
four-layer P104 set: 4,16,28,36
6-prompt generation gate
```

### 3. Lightweight Recovery prototype

Start with LoRA or small adapter targets that are likely to repair activation
margin without rewriting the whole model:

```text
shared expert gate/up/down
active expert gate/up/down for sensitive layers
router-adjacent projection only if routing drift appears
linear-attn projections only if P105 drift propagates outside MoE
```

Use calibration data:

```text
pruning/calibration/calibration_set_v1.1.jsonl  # 1436 prompts
P105 prompts
tool-call and no-think guard prompts
```

### 4. MTP/NEXTN head preflight

Do not start with a full expensive MTP run. First answer:

```text
which implementation will own the MTP head
what hidden states feed the draft head
what missing keys need initialization
whether the base model can stay frozen
how serving verifies/rejects draft tokens
how to evaluate accept rate and quality drift
```

## Training Order

Recommended order:

```text
1. W4A8 Recovery dry-run
2. W4A8 short adapter run
3. merge/evaluate W4A8 candidate
4. MTP/NEXTN head prototype on top of W4A8-stable base
5. combined W4A8 + MTP evaluation
```

This is "parallel preparation, staged training".

## Early Stop Rules

Continue W4A8 if:

```text
P105 exactness improves or divergence moves later
V8/V9/tool/no-think do not regress materially
loss decreases on calibration without output collapse
```

Pause W4A8 if:

```text
P105 divergence moves earlier
tool-call or no-think guard breaks
V8 drops below the BF16 baseline by a large margin
```

Continue MTP if:

```text
draft accept rate rises on deterministic prompts
final verified output remains identical or quality-neutral
latency model predicts >1.2x serving gain
```

Pause MTP if:

```text
it masks base-model W4A8 regressions
accept rate is too low on Lynn workloads
training destabilizes final logits
```

## Resource Decision For Renewal

By Monday 2026-05-18 noon, decide renewal based on:

```text
BF16 transfer complete and loadable
W4A8 dry-run completed
at least one short Recovery checkpoint evaluated
MTP head preflight complete
```

Renew A100 if W4A8 P105 moves in the right direction or MTP head preflight is
green. If both are blocked by framework/model-loading issues, pause and debug
offline before extending.

## Bottom Line

W4A8 is the quality-first acceleration path. MTP/NEXTN is the multiplier. They
belong in the same A100 campaign, but the base W4A8 behavior should stabilize
before serious MTP training.
