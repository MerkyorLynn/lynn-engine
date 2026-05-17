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

This changes the initialization and wiring path, not the serving state: MTP can
now run a real draft forward pass, but the warm-start head is not yet an
acceptable draft predictor. It still needs head-only training and accept-rate
evaluation before it can contribute TPS.

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

2026-05-17 status: Stage 0 shape mapping and forward smoke are complete for
the official 2048-hidden Qwen3.6-35B-A3B sidecar. The forward path is finite
but not token-aligned, so Stage 1 training is now the blocker.

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
