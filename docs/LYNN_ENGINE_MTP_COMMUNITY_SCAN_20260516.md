# Lynn Engine MTP Community Scan

Date: 2026-05-16

## Decision

The A100 MTP/NEXTN stream should target a **Qwen3.6 / `qwen3_next_mtp` style
one-layer predictor**, not a simple Linear NEXTN head.

The W4A8 Recovery stream remains first in the critical path. MTP is prepared in
parallel, then trained/evaluated on top of the first generation-stable W4A8
candidate.

## Why The Contract Changed

The earlier single-Linear NEXTN sketch is useful for wiring, but it is
under-specified for the target accept-rate band. Community and framework
evidence points to native MTP heads as model-owned predictor layers:

```text
base model hidden state
  -> qwen3_next_mtp-style transformer predictor layer
  -> shared lm_head
  -> draft token(s)
  -> target-model verification
```

The first Lynn-owned smoke should use:

```text
num_speculative_tokens = 2
single stream
frozen or mostly frozen base model
shared embedding / lm_head
```

Do not combine this first smoke with continuous batching. Lynn-engine is still
single-stream; batched MTP is a separate serving-engine project.

## Community Findings

### Qwen3.6 MTP Head Sidecar Exists

`guru87/Qwen3.6-27B-MTP` publishes a standalone BF16 MTP head extracted from
`Qwen/Qwen3.6-27B`.

Key properties to verify after download:

```text
file: mtp.safetensors
expected tensors: 15
expected size: ~811 MB
provenance: Qwen/Qwen3.6-27B MTP shards
intended use: re-attach MTP weights to quantized Qwen3.6 checkpoints missing mtp.* tensors
```

This does **not** automatically make it compatible with Lynn 27B. The README
tensor manifest is already enough to reject direct weight transplant:

```text
Qwen3.6 sidecar hidden size: 5120
Lynn 27B text hidden size:  2048
sidecar tensors containing 2048: 0 / 15
decision: RED for direct initializer
```

Report:

```text
reports/a100/a100_mtp_sidecar_shape_audit_readme.json
```

Therefore the sidecar is an **architecture oracle**, not a direct initializer.
Lynn needs its own 2048-hidden MTP predictor weights.

Lynn-owned shape spec:

```text
docs/LYNN_ENGINE_MTP_LYNN_2048_HEAD_SPEC_20260516.md
```

### vLLM Supports `qwen3_next_mtp`

vLLM exposes `qwen3_next_mtp` in speculative decoding configuration, and the
Qwen3.6 model card uses:

```bash
--speculative-config '{"method":"qwen3_next_mtp","num_speculative_tokens":2}'
```

This makes Qwen3.6-style MTP the ecosystem-compatible direction.

### Spark / GB10 Data Lowers Single-Stream Expectations

Community Spark/GB10 measurements report that MTP is most valuable in
concurrent workloads, not necessarily in single-stream decode.

Planning implication:

```text
single-stream MTP expectation: modest, roughly +5-10% until measured
multi-concurrent expectation: larger, but Lynn-engine needs continuous batching
```

So MTP remains valuable, but it is not a magic single-stream 200 TPS switch.

## A100 Work Plan

### Step 1: Sidecar Shape Audit

Download and inspect the community sidecar:

```text
/mnt/data/lynn-a100/models/mtp_sidecars/guru87-qwen36-27b-mtp
```

Checks:

```text
keys start with mtp.*
hidden size matches Lynn 27B
intermediate size / attention heads match or can be mapped
shared lm_head contract is plausible
no variable-expert pruning conflict in predictor layer
```

If a future sidecar matches, use it as:

```text
architecture template + possible initializer
```

For the current Qwen3.6-27B sidecar, keep the architecture and initialize a
Lynn-owned head from scratch.

### Step 1b: Official Qwen3.6-35B-A3B MTP Warm-Start

The official Qwen3.6-35B-A3B checkpoint is more directly useful than the
community 27B sidecar:

```text
hidden_size: 2048
mtp_num_hidden_layers: 1
mtp_use_dedicated_embeddings: false
mtp key count: 19
mtp shard files:
  model-00025-of-00026.safetensors
  model-00026-of-00026.safetensors
```

Its `mtp.*` tensor names match the Qwen3.6 `qwen3_next_mtp` contract and the
base hidden size matches Lynn. This makes it the preferred warm-start source for
the A100 MTP stream:

```text
download only the two MTP shards if possible;
extract mtp.* tensors;
load all shape-compatible tensors into the Lynn-owned 2048-hidden MTP module;
fine-tune on Lynn W4A8 calibration prompts with the body frozen or mostly frozen.
```

This does not change the ordering: W4A8 Recovery remains the first promotion
gate, while MTP is prepared in parallel as the serving multiplier.

### Step 2: Lynn MTP Smoke

Minimum smoke:

```text
load base hidden state
run MTP predictor for 2 draft tokens
verify with target model
record accept rate and exact verified output
```

Promotion gate:

```text
verified output identical to target greedy
accept rate high enough to predict >1.1x single-stream gain
no no-think/tool-call regression
```

### Step 3: Train Only After W4A8 Base Stabilizes

Do not train MTP on an unstable W4A8 base. If W4A8 generation is still RED, MTP
accept-rate measurements become noisy and hard to attribute.

Recommended order:

```text
W4A8 Recovery -> generation AMBER/GREEN
MTP sidecar shape audit -> smoke
MTP head train/fine-tune -> accept-rate gate
combined W4A8 + MTP eval
```

## Sources

- Hugging Face sidecar: <https://huggingface.co/guru87/Qwen3.6-27B-MTP>
- Qwen3.6 model card speculative config: <https://huggingface.co/Qwen/Qwen3.6-27B>
- vLLM MTP documentation: <https://docs.vllm.ai/en/latest/features/speculative_decoding/mtp/>
- vLLM speculative config source/docs include `qwen3_next_mtp`: <https://docs.vllm.ai/en/stable/api/vllm/config/speculative.html>
- Spark/GB10 MTP measurements: <https://docai.hu/en/blog/qwen36-mtp-gb10>

## Bottom Line

Mainline:

```text
W4A8 Recovery first, Lynn-owned 2048-hidden qwen3_next_mtp-style predictor second.
```

Fallback:

```text
simple Linear NEXTN only as wiring smoke, not as the target quality route.
```
