# Lynn Engine P99: activation quantization strategy after P98

## Why P98 Stops Here

P98 proved that the split16 SM120a FP4 MMA gate/up kernel can be wired into the
runtime, but it also proved that **runtime BF16 activation -> E2M1 activation
quantization is not a safe drop-in replacement** for the current production MoE
semantics.

The failed gate is useful because it separates two contracts:

| Contract | Meaning | Status |
|---|---|---|
| Quantized-activation contract | Native FP4 MMA matches an explicit E2M1 activation reference | PASS in P93/P97 |
| Production BF16-activation contract | Native path preserves current greedy decode behavior | FAIL in P98 |

This means the current Lynn-native NVFP4 artifact can keep driving kernel
research, but production promotion needs a different route.

## Practical Implication

Do not keep spending engineering time trying to hide activation quantization
inside the decode loop. It creates two problems at once:

1. graph capture breaks because regular PyTorch quantization ops allocate and
   dispatch during capture;
2. even graph-off, greedy outputs diverge because the active MoE no longer sees
   the same BF16 activation values.

The correct split is:

- R6000 runtime work should preserve current BF16-activation semantics unless a
  later gate explicitly proves otherwise.
- A100 training / re-quant work should absorb activation quantization if we want
  a true W4A4 active-MoE path.

## Next Engineering Routes

### Route A: BF16 activation + FP4 weight semantics

Keep the production activation as BF16 and optimize the weight side / scheduling
without changing the numerical contract. This is safer for immediate serving,
but it may not unlock full Blackwell FP4 tensor-core throughput.

Useful probes:

- native C++/CUDA loop reduction around current Triton kernels;
- fewer Python dispatch boundaries;
- fused scheduling around gate/up + down while preserving BF16 activation.

### Route B: Activation-quant-aware model line

Treat E2M1 activation as part of the model contract, not a hidden runtime trick.
This belongs with the A100 work:

- MTP / NEXTN training or fine-tuning;
- vendor-friendly NVFP4 v2 re-quant;
- calibration / imatrix-style datasets for activation-sensitive layers;
- strict BF16 vs NVFP4 quality gates after export.

This is the right place to make W4A4 a production promise.

### Route C: Dual artifact support

Keep both artifacts:

- Lynn-native NVFP4: current engine artifact, best for Lynn-specific runtime.
- Vendor/official-style NVFP4: future artifact for official kernels or cross
  framework loading.

The engine already has layout detection work, so supporting both is feasible.

## A100 Coordination

A100 now owns the training / re-quant artifact line:

1. pull the 27B BF16 final artifact from R6000;
2. inspect whether MTP metadata/head survives in the final model;
3. if not, prepare a small NEXTN/MTP training recipe from BF16;
4. export new NVFP4 artifacts only after BF16 gates pass;
5. never overwrite the current Lynn-native NVFP4 artifact.

## Decision

P99 does **not** promote P98. It redirects production work toward either
BF16-preserving runtime optimization or activation-quant-aware training /
re-quantization on A100.

