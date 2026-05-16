# Lynn Engine dual NVFP4 artifact policy

Date: 2026-05-16

## Authorization

We may maintain two NVFP4 artifacts when needed:

1. **Lynn-native NVFP4**
2. **Vendor-friendly NVFP4 v2**

They are not interchangeable and must never overwrite each other.

## Artifact A: Lynn-native NVFP4

Purpose:

- canonical Lynn engine runtime artifact;
- physical variable-expert 27B;
- custom per-16 scale contract;
- target for grouped native-FP4 active expert kernels.

Current path:

```text
/root/autodl-tmp/models/lynn-27b-variable-recovery-step5000-nvfp4-final
```

Properties:

```text
format:        Lynn-native
expert shape:  physical variable experts per layer
scale:         per-16 FP16/FP32-style scales
runtime:       Lynn engine
public status: internal / engine-first
```

Do not:

- convert it in-place to vendor layout;
- overwrite it with ModelOpt/compressed-tensors output;
- call it SGLang/vLLM-compatible unless a specific adapter proves that.

## Artifact B: vendor-friendly NVFP4 v2

Purpose:

- official ecosystem compatibility;
- SGLang/vLLM/vendor-kernel experiments;
- external benchmark/public deployment path.

Suggested path:

```text
/root/autodl-tmp/models/lynn-27b-variable-recovery-step5000-nvfp4-vendor-v2
```

Source:

```text
/root/autodl-tmp/models/lynn-27b-variable-recovery-step5000-bf16-final
```

Properties:

```text
format:        ModelOpt / compressed-tensors-compatible NVFP4 if possible
expert shape:  either padded/masked fixed-256 or vendor-supported variable expert
scale:         vendor expected contract
runtime:       SGLang / vLLM / vendor kernels, if gates pass
public status: compatibility candidate
```

Do not:

- derive it by post-hoc converting the current Lynn-native artifact;
- publish it without V8/V9/tool/no-think/longctx gates;
- silently rename it as the canonical Lynn-native artifact.

## Required validation for vendor v2

Before calling vendor v2 successful:

```text
load gate:
  config/index/tokenizer load
  framework server launch or direct model load

numeric gate:
  layer-level or prompt-level comparison vs BF16 / Lynn-native

quality gate:
  6-prompt coherent smoke
  strict tool-call
  no-think loop guard
  V8 / V9 / coding spike
  long-context sanity

runtime gate:
  TPS / memory / TTFT compared with Lynn-native
```

## Storage rule

Both artifacts may coexist. Use explicit names:

```text
...-nvfp4-final          # Lynn-native canonical engine artifact
...-nvfp4-vendor-v2      # vendor-compatible candidate
```

Any script writing an NVFP4 artifact must print:

```text
ARTIFACT_KIND=lynn_native|vendor_friendly
SOURCE_BF16=...
OUTPUT=...
OVERWRITE=false unless explicit
```

## Strategy

Dual-track is deliberate:

- Lynn-native preserves the unique advantages of the pruned 27B and engine
  specialization.
- Vendor-friendly maximizes compatibility and lets us benefit from official
  kernels if the quality gates pass.

The winning production default can be decided by measured memory, latency,
quality, and deployability, not by ideology.
