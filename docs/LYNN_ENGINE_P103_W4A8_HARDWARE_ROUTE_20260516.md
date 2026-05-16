# Lynn Engine P103: W4A8 hardware route confirmed

## Result

P103 checks the practical W4A8 question after P102:

> If BF16/FP16 x E2M1 is not exposed, can R6000 still run FP8 activation x
> E2M1 weight MMA for a W4A8 active-MoE route?

The answer is **yes** on the current `sm_120a` CuTe/CUDA stack.

```text
reports/p103/p103_sm120a_fp8_activation_fp4_weight_mma_probe.json

controls_ok:                         true
fp8_activation_fp4_weight_supported:  true
```

All tested W4A8 candidate atoms compile:

| Variant | Status |
|---|---|
| E4M3 x E2M1 -> F32 | PASS |
| E5M2 x E2M1 -> F32 | PASS |
| E2M1 x E4M3 -> F32 | PASS |
| E2M1 x E5M2 -> F32 | PASS |
| blockscaled E4M3/E5M2 x E2M1 -> F32 + UE8M0 | PASS |
| blockscaled E2M1 x E4M3/E5M2 -> F32 + UE8M0 | PASS |

## Decision

W4A8 + MTP/NEXTN becomes the **near-term mainline** for the 155-200 TPS band.

The route is:

```text
BF16 final
  -> FP8-activation-aware adaptation / calibration
  -> MTP/NEXTN head or adapter training
  -> W4A8/NVFP4-v2 export
  -> R6000 W4A8 active-MoE runtime gate
```

W4A4 remains the higher-ceiling research line, but W4A8 has the better first
shot at quality, cross-device compatibility, and a 155+ milestone.

## Why W4A8 Is Attractive

- It keeps weight-side NVFP4/E2M1 compression.
- It gives activation much more dynamic range than E2M1 W4A4.
- It has a real SM120a tensor-core atom path.
- It is more likely to preserve MTP acceptance rate than pure W4A4.
- It maps better to Spark `sm_121`, which lacks FP4 tensor cores but has FP8
  tensor-core support.

## Runtime Implication

R6000 should treat W4A8 as a first-class artifact family, not merely a fallback.

The engine should support:

- explicit artifact layout in the quant manifest;
- layer/expert precision masks;
- E4M3 vs E5M2 activation policy;
- separate promotion gates from current Lynn-native W4A4 and vendor-friendly
  NVFP4-v2 artifacts.

The current production BF16-activation path remains the stable baseline while
the W4A8 artifact line is trained and exported on A100.
