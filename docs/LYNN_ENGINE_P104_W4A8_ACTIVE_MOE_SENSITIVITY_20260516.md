# Lynn Engine P104: W4A8 Active-MoE Sensitivity

Date: 2026-05-16

## Purpose

P102 closed the BF16-activation x FP4-weight mixed-MMA shortcut. P103 proved
that R6000/SM120a supports FP8 activation x E2M1 FP4 weight MMA. P104 measures
whether the current Lynn-native NVFP4 artifact can tolerate FP8-rounded
activations at active expert FFN boundaries before training.

This is a fake-quant quality gate, not a final performance benchmark.

## Setup

Model:

```text
/root/autodl-tmp/models/lynn-27b-variable-recovery-step5000-nvfp4-final
```

Probe:

```text
benchmarks/p104_w4a8_active_moe_sensitivity.py
```

Layers:

```text
4, 16, 28, 36
```

Compared variants:

```text
E4M3 tensor / row / per16
E5M2 tensor / row / per16
```

Router logits and top-k selection remain BF16. Only active expert activation
boundaries are fake-quantized.

## Result

P104 is **AMBER**, not red.

The best variant is **E4M3 per16**.

```text
gate/up-only:
  all four layers pass relaxed gate
  max rel_l2 = 0.02647
  min cosine = 0.99965

full-active gate/up + down activation:
  near gate, but not fully green
  max rel_l2 = 0.03302
  min cosine = 0.99946
```

Layer 16 is the only material blocker for the full-active relaxed threshold
(`rel_l2 <= 0.03`). It lands at `0.03302`, which is close enough to treat this
as a training/adaptation target rather than a failed hardware route.

E5M2 is clearly worse and should not be the first training target.

## Decision

Proceed with **W4A8 E4M3 per16 + MTP/NEXTN** as the primary A100 adaptation
line.

Runtime staging:

1. Gate/up W4A8 can be developed first because it already passes the relaxed
   active-MoE isolation gate.
2. Full-active W4A8 should stay behind a QAT-lite/Recovery gate until the down
   activation drift is pulled below threshold.
3. Do not promote W4A8 full-active runtime on the current BF16-trained artifact
   without adaptation.

## A100 Training Target

The immediate A100 objective is not open-ended rescue. It is targeted:

```text
Reduce E4M3-per16 full-active active-MoE max rel_l2:
  0.03302 -> <= 0.03000

Keep:
  min cosine >= 0.9995
  V8 strict near BF16 baseline
  V9 adjusted near BF16 baseline
  tool-call strict pass
  no-think loop guard pass
```

MTP/NEXTN can run in the same adaptation campaign after the first W4A8
sensitivity gate confirms the artifact is trainable.

## Files

```text
reports/p104/p104_w4a8_active_moe_4layer_v2.json
reports/p104/p104_w4a8_active_moe_smoke_layer28.json
benchmarks/p104_w4a8_active_moe_sensitivity.py
```
