# Lynn-native vs vendor-friendly NVFP4: tradeoff note

Date: 2026-05-16

## Executive summary

We can build a vendor-friendly NVFP4 artifact, but it should be treated as a
second artifact family, not as a replacement for Lynn-native 27B.

```text
Lynn-native path:
  maximize control, variable-expert efficiency, exact model identity

Vendor-friendly path:
  maximize ecosystem compatibility, official kernel reuse, external serving
```

The strategic answer is dual-track:

```text
production / engine sprint:
  keep Lynn-native per-16 artifact

compatibility / public ecosystem:
  create vendor-friendly NVFP4 v2 from BF16 final
```

## Lynn-native advantages

### 1. Preserves physical variable-expert pruning

Lynn 27B physically removes a different number of experts per layer:

```text
remaining expert counts:
  [222, 228, 231, 232, 233, 235, 236, 242, 248, 252, 254, 256]
```

That is not a normal HF fixed-256-expert checkpoint. Keeping this physical
layout preserves the real parameter/memory reduction instead of padding dead
experts back into the model.

### 2. Keeps the current best quality scale contract

The current Lynn-native artifact uses packed E2M1 weights with per-16 scales.
P54 showed that directly compressing this into e8m0/group32-style vendor scales
does not pass quality:

```text
best optimistic inter cosine: 0.9869-0.9918
safety gate:                  >0.995
```

So staying native avoids a known quality cliff.

### 3. Enables custom kernels for the actual model shape

Lynn engine can write kernels for the real route:

```text
top_k=8 selected experts
variable expert counts per layer
packed E2M1 + per-16 scales
linear-attn / hybrid cache assumptions
Lynn 27B batch=1 / resident serving profile
```

General frameworks must optimize for many models. Lynn can optimize for one
model family.

### 4. Keeps product differentiation

The whole engine story is strongest when Lynn owns:

- pruning format;
- quantization format;
- runtime memory lifecycle;
- tool/no-think serving behavior;
- long-context linear-attn behavior;
- custom active expert kernels.

Giving all of that back to a general framework reduces differentiation.

## Lynn-native disadvantages

- More engineering burden.
- Need to own CUDA/CUTLASS/Triton correctness.
- Fewer off-the-shelf deployment options.
- Harder for external users who expect vLLM/SGLang/llama.cpp.
- Every new model variant needs loader/runtime validation.

## Vendor-friendly advantages

### 1. Ecosystem compatibility

If we produce a ModelOpt / compressed-tensors-compatible NVFP4 artifact, it can
potentially run in SGLang/vLLM/vendor stacks without users adopting Lynn engine.

That matters for public adoption.

### 2. Official kernel reuse

NVIDIA's public NVFP4 MoE checkpoints show that Blackwell native FP4 serving is
a real vendor-supported direction. A compatible artifact can benefit from:

- upstream kernel improvements;
- broader testing;
- standard deployment recipes;
- less custom maintenance.

### 3. Better benchmark comparability

External comparisons are easier if one artifact runs in common frameworks.

### 4. Lower operational risk for non-Lynn deployments

Users who already operate SGLang/vLLM may prefer a standard artifact even if it
is not the most memory-efficient Lynn-native form.

## Vendor-friendly disadvantages

### 1. Likely gives up physical variable-expert savings

Unless the vendor stack supports per-layer variable expert counts, we need to
pad/mask back to fixed 256 experts. That can lose part of the 27B design
advantage:

```text
physical variable expert:
  no dead expert tensors loaded

padded/masked vendor artifact:
  easier framework compatibility
  but may carry padded/dead expert storage or dispatch overhead
```

### 2. May require a new quantization recipe

P54 rejects post-hoc conversion. The vendor-friendly artifact should be
quantized from BF16 final with calibration / activation-aware scale selection /
possibly QAT-style recovery.

That is a real artifact project, not a file rename.

### 3. May lose exact identity with the current Lynn-native 27B

Different scale contract, possible padding, and different quantization recipe
can shift output. It must pass V8/V9/tool/no-think/longctx gates before being
called equivalent.

### 4. Less room for bespoke long-term optimization

Official frameworks optimize a broad surface. Lynn engine can specialize:

- one model;
- one hardware family;
- one serving profile;
- one memory lifecycle.

## Recommended strategy

Do both, but do not merge their goals.

### Mainline

```text
Lynn-native per-16 27B
  -> custom grouped active expert kernel
  -> 155 TPS
  -> product/runtime differentiation
```

### Sidecar

```text
BF16 final
  -> vendor-friendly NVFP4 v2
  -> SGLang/vLLM compatibility
  -> public artifact / external benchmark path
```

## Decision rule

Use Lynn-native as the canonical production brain model if it wins on:

- memory;
- latency;
- long-context;
- tool/no-think behavior;
- controllability.

Use vendor-friendly as the public compatibility model if it wins on:

- user deployability;
- standard benchmark coverage;
- framework support;
- lower maintenance for external users.

If vendor-friendly unexpectedly matches native on memory, quality, and speed,
then it can become the default. Until then, converting should be additive, not
a replacement.
