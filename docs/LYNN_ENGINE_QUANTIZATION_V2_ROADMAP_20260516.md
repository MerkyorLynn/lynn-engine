# Lynn Quantization v2 roadmap: imatrix, layer strategy, and NVFP4 artifacts

Date: 2026-05-16
Branch: `codex/p16-r6000-155-tps`

## Short answer

Activation-aware quantization is worth doing, but it belongs to a separate
quality track from the current 155TPS engine work.

```text
current engine track:
  grouped per-16 native FP4 active expert FFN
  goal: runtime speed, memory, serving stability

quantization v2 track:
  imatrix / calibration / layer strategy / scale search
  goal: lower PPL/KLD, better V8/V9/tool retention, stronger public artifacts
```

The two tracks compound later, but they should not block each other.

## Where it helps immediately

### 1. GGUF / Q4_K_M public artifacts

This is the clearest near-term win. GGUF quantization already has a mature
calibration ecosystem, and imatrix is effectively a small quantization-aware
step at quantization time:

- no extra training loop;
- no runtime cost;
- can improve layer/codebook choices;
- can reduce quality loss on academic, coding, tool-call, and long-context
  prompts.

For Lynn, the calibration/imatrix set should not be generic only. It should mix:

- Lynn HAS / style prompts;
- V8 tool-call prompts with strict `tool_calls` validation;
- V9 academic / coding / long-context prompts;
- Chinese daily-writing / creative prompts;
- router-sensitive MoE prompts from the 27B activation profile.

Recommended gates:

```text
PPL / KLD:
  generic corpus + Lynn calibration corpus

Task retention:
  6-prompt smoke
  V8 strict tool-call
  V9 academic/coding
  longctx
  no-think loop guard
```

### 2. Future vendor-compatible NVFP4 artifacts

P54 shows that a simple e8m0/group32 exponent search cannot convert the current
Lynn-native per-16 artifact into a vendor-style scale contract:

```text
best activation-aware upper-bound inter cosine:
  layer 4:  0.98754
  layer 16: 0.99136
  layer 28: 0.98692
  layer 36: 0.99184

gate: >0.995 cosine and <0.08 rel_l2
result: all fail
```

However, that does not rule out a second artifact quantized from BF16 with
calibration, imatrix-like weighting, or QAT-style scale selection. It only
rules out direct post-hoc conversion of the existing Lynn-native NVFP4 scales.

This becomes a useful follow-up track:

```text
BF16 final
  -> activation/calibration profile
  -> layer-aware NVFP4 scale/layout search
  -> vendor-friendly artifact candidate
  -> full V8/V9/tool/longctx gates
```

If this passes, Lynn gets a compatibility artifact for SGLang/vLLM/vendor
kernels. If it fails, the Lynn-native per-16 kernel remains the right path.

## Where it does not help immediately

It does not solve the current 100→155TPS blocker by itself.

P16-P54 already locate the runtime bottleneck:

- active routed experts dominate the remaining token time;
- top-k reduction / skip-shared shortcuts break quality first;
- PyTorch `_scaled_mm` selected-expert composition is slower and drifts;
- direct e8m0/group32 bridge fails quality even under optimistic scale search.

So imatrix/layer strategy can make artifacts better, but it does not replace
the kernel work needed for:

```text
Lynn packed E2M1 + FP32 per-16 scales
  -> grouped active expert gate/up
  -> grouped active expert down
  -> exact/near-exact active MoE output
```

## Recommended execution plan

### Track A: keep engine sprint unblocked

Continue grouped per-16 native FP4 active expert FFN work on R6000:

1. start from P52 active expert contract;
2. avoid e8m0/group32 bridges for the existing artifact;
3. validate with full-generate parity or task retention gates;
4. promote only after OpenAI server path passes tool/no-think guards.

### Track B: start GGUF quantization v2

For public llama.cpp users:

1. build a Lynn-specific imatrix/calibration set;
2. quantize Q4_K_M baseline and imatrix variants;
3. add layer-strategy variants if tooling supports it;
4. measure PPL/KLD and V8/V9/task retention;
5. publish only if the new artifact beats current Q4_K_M on quality without
   worse usability.

### Track C: investigate vendor-compatible NVFP4 v2

After engine kernel work has a stable lane:

1. quantize from BF16 final, not from the current Lynn-native NVFP4 artifact;
2. use calibration/imatrix-like weighting for scale/layout selection;
3. test e8m0/group32 or vendor-friendly layouts against P52/P54 gates;
4. only keep it if quality retention is strong enough to justify a second
   artifact family.

## Decision

The recommendation is yes, with scope control:

- **current stage**: do not derail the 155TPS kernel sprint;
- **parallel stage**: begin GGUF Q4_K_M imatrix/layer-strategy work because it
  directly helps public users;
- **next stage**: use activation-aware quantization ideas to revisit
  vendor-compatible NVFP4 artifacts from BF16, not as a direct conversion of
  the current Lynn-native artifact.

In short: imatrix/layer strategy is a quality multiplier, not the current
runtime bottleneck fix. It is still worth doing because Lynn needs both:
fast native serving and strong portable artifacts.
