# Lynn Engine P102: BF16/FP16 x E2M1 mixed MMA probe

## Result

P102 answers the highest-ROI hardware question after P99/P101:

> Can R6000 `sm_120a` run a native mixed BF16/FP16 activation x E2M1 weight MMA,
> so Lynn can keep BF16 activation semantics while using FP4 tensor cores?

The answer on the current CuTe/CUDA stack is **no**.

```text
reports/p102/p102_sm120a_mixed_bf16_fp4_mma_probe.json

controls_ok:                         true
mixed_bf16_or_f16_fp4_supported:      false
supported_mixed_variants:             []
```

The controls compile:

| Variant | Status |
|---|---|
| raw E2M1 x E2M1 -> F32 | PASS |
| raw E2M1 x E2M1 -> F16 | PASS |
| blockscaled E2M1 x E2M1 -> F32 + UE8M0 | PASS |

All mixed candidates fail at CuTe atom instantiation:

| Variant family | Status |
|---|---|
| BF16 x E2M1 -> F32 | FAIL |
| E2M1 x BF16 -> F32 | FAIL |
| F16 x E2M1 -> F32 | FAIL |
| E2M1 x F16 -> F32 | FAIL |
| blockscaled BF16/F16 x E2M1 -> F32 + UE8M0 | FAIL |

Representative compiler failure:

```text
static assertion failed with "No MMA matches SM120_16x8x32_TN for given data types."
```

## Why This Matters

P93/P97 proved the native split16 active-MoE kernel math is sound when both
activation and weight operands are E2M1. P98 proved that hiding BF16 -> E2M1
activation quantization inside runtime changes greedy generation and is not a
drop-in production replacement.

P102 closes the remaining shortcut:

- the hardware/CuTe route does expose E2M1 x E2M1;
- it does not expose BF16/FP16 x E2M1 for this SM120a MMA atom;
- therefore the 155+ TPS native FP4 route needs quantized activation as an
  explicit model contract.

## Decision

Do **not** plan on a BF16-activation + FP4-weight mixed-MMA shortcut for R6000.

The R6000 runtime split is now:

1. keep current production serving BF16-activation semantics for the stable
   100-130 TPS class;
2. move 150+ TPS work to an A100-produced W4A4 / activation-aware artifact;
3. keep the old Lynn-native NVFP4 artifact as a stable baseline and export the
   W4A4/NVFP4-v2 line as a separate package.

## A100 Implication

A100 is no longer optional for the 155+ route. It owns the next artifact line:

```text
BF16 final
  -> MTP/NEXTN head training or validation
  -> activation-aware W4A4 / QAT-lite adaptation
  -> re-quant/export as a new NVFP4-v2 package
  -> strict V8/V9/tool/longctx gates
```

Training does not replace quantization. It teaches the model to tolerate the
quantized activation contract; quantization/export still produces the deployable
runtime artifact.

## Spark Note

Spark `sm_121` is a different route. SP-11 showed that it lacks FP4/FP6 tensor
core MMA but has FP8 tensor core support. Spark does not need this exact R6000
W4A4 FP4 artifact, but the same activation-aware training data and LoRA/recovery
recipe can feed a Spark-specific FP8-friendly export later.
