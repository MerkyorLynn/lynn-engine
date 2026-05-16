# Lynn Engine P57: official/vendor NVFP4 route inventory

Date: 2026-05-16
Branch: `codex/p16-r6000-155-tps`

## Short answer

Yes, if Lynn wants to use the official / vendor ecosystem path, we need a
separate artifact quantized into the vendor-expected NVFP4 layout.

That is different from the current Lynn-native artifact:

```text
current Lynn engine artifact:
  physical variable-expert 27B
  packed E2M1 weights
  FP16/FP32-ish per-16 scales
  custom Lynn manifest
  loaded by Lynn engine

vendor-friendly artifact:
  likely HF-vanilla-compatible layout
  ModelOpt / compressed-tensors NVFP4 contract
  e8m0/group32-style scale expectations
  usable by SGLang / vLLM / vendor kernels if quality gates pass
```

The current artifact should not be post-hoc converted into the vendor layout.
P54 already showed that simple e8m0/group32 scale search fails the safety gate.

## Ground truth inventory

Probe:

```text
benchmarks/p57_vendor_route_inventory.py
reports/p16_155/p57_vendor_route_inventory_bf16.json
```

R6000 current eval environment:

| Item | Status |
|---|---|
| `modelopt` | ❌ not installed |
| `llmcompressor` | ❌ not installed |
| `compressed_tensors` | ✅ 0.15.0.1 |
| `compressed_tensors.modelopt_nvfp4_converter` | ✅ available |
| PyTorch | ✅ 2.10.0+cu128 |
| Triton | ✅ 3.6.0 |
| BF16 final | ✅ `/root/autodl-tmp/models/lynn-27b-variable-recovery-step5000-bf16-final` |
| Lynn-native NVFP4 final | ✅ `/root/autodl-tmp/models/lynn-27b-variable-recovery-step5000-nvfp4-final` |

Important nuance:

```text
compressed_tensors.modelopt_nvfp4_converter
  converts already-quantized ModelOpt NVFP4 tensors into CT naming;
  it does not perform BF16 -> NVFP4 quantization.
```

## Variable-expert blocker

The 27B final BF16 checkpoint is physically variable-expert:

```text
hf_vanilla_compatible: false
remaining expert counts:
  [222, 228, 231, 232, 233, 235, 236, 242, 248, 252, 254, 256]
```

Official HF/SGLang/vLLM/ModelOpt stacks usually expect the architecture config
to allocate a fixed `num_experts=256` per layer. Lynn 27B intentionally removed
different numbers of experts per layer. That is why Lynn engine exists.

So the official route has an extra compatibility step:

1. pad/mask the variable-expert model back into a fixed 256-expert HF layout; or
2. teach the vendor stack to handle per-layer variable expert counts; or
3. keep vendor route for a separate future artifact while Lynn engine owns the
   physical variable-expert artifact.

## Decision

The official route is not rejected. It is simply a **new artifact track**, not a
replacement for the current engine sprint.

Recommended split:

### Track A: current runtime mainline

```text
Lynn-native 27B NVFP4 per-16 artifact
  -> Lynn engine
  -> grouped per-16 native FP4 active expert kernel
  -> 155 TPS target
```

This remains the fastest path for the model we already have.

### Track B: vendor-compatible NVFP4 v2

```text
BF16 final
  -> pad/mask or vendor-compatible variable-expert adapter
  -> ModelOpt / llmcompressor NVFP4 quantization
  -> compressed-tensors conversion if needed
  -> SGLang/vLLM/vendor kernel tests
  -> V8/V9/tool/no-think/longctx retention gates
```

This is the route to official ecosystem compatibility.

### Track C: post-hoc conversion of current artifact

```text
Lynn-native per-16 artifact -> e8m0/group32 vendor layout
```

Rejected by P54 for the current artifact. Even optimistic activation-aware
scale search failed:

```text
best upper-bound inter cosine: 0.9869-0.9918
required safety gate:          >0.995
```

## Practical next step

Do not install ModelOpt into the active engine env blindly. Create or reuse a
separate quantization env, then run a small vendor-route smoke on a subset or a
padded mini checkpoint:

1. verify ModelOpt/llmcompressor can load the 27B BF16 config;
2. if it fails on variable experts, build the padding/mask adapter first;
3. quantize one or a few layers to the official layout;
4. run P52/P54-style numeric gates before full 60G quantization.

If the vendor route passes, it is a valuable second artifact. If it fails, the
Lynn-native engine path remains fully justified.
