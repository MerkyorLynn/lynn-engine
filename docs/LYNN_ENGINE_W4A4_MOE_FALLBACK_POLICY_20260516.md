# Lynn Engine W4A4 MoE fallback policy

## Decision

The long-term mainline remains **MoE-first**.

Dense 27B is not the near-term fallback for W4A4 risk. It is simpler to run, but
it gives up the active-parameter advantage that makes Lynn 27B-A3B worth building
a dedicated runtime for.

## Why MoE Stays Primary

The current model is a variable-pruned MoE with a small active footprint. The
runtime bottleneck is active expert execution, not a lack of model capacity.
If W4A4 succeeds, MoE keeps all three strategic advantages:

- lower active compute than dense 27B;
- native FP4 active-expert tensor-core route on R6000 / 5090-class Blackwell;
- Lynn-specific routing/expert layout that generic frameworks do not exploit.

Dense 27B remains useful as a teacher, ablation baseline, or last-resort product
fallback, but not as the first answer to W4A4 difficulty.

## Fallback Ladder

If the W4A4 fake-quant gate is weak, use this order. Do not jump straight to a
dense model.

| Level | Fallback | Goal | Promotion gate |
|---|---|---|---|
| 0 | Better calibration / imatrix-style activation set | Fix bad scale/code choices without training | smoke/tool/longctx no worse than BF16 spot by an obvious margin |
| 1 | Layer precision mask | Keep router, lm_head, early dispatch, and transition layers BF16/FP8 | V8/V9/tool pass while most active-expert layers still run W4A4 |
| 2 | Expert-local rescue | Train LoRA/QAT-lite on sensitive experts/layers only | targeted categories recover without global regression |
| 3 | Mixed W4A4/W4A8 | Use FP8/BF16 activation only where E2M1 activation is too destructive | still enough W4A4 coverage for native FP4 MoE speedup |
| 4 | Longer Recovery/QAT-lite | Let the model adapt to the quantized activation contract | V8 strict >= 95%, V9 adjusted >= 60%, tool/longctx/no-think pass |
| 5 | Vendor-friendly NVFP4-v2 branch | Try official scale/layout if Lynn-native mask is the blocker | separate artifact, separate manifest, separate gates |
| 6 | Dense 27B fallback | Product fallback or teacher, not the engine mainline | only after MoE W4A4 and mixed-precision paths both fail |

## Practical A100 Rule

The first A100 gate should not answer "does pure W4A4 work?" only. It should
answer:

1. how much of the model can run W4A4 safely;
2. which layers/experts need BF16/FP8 protection;
3. whether a short QAT-lite/Recovery run improves the weak spots;
4. whether MTP/NEXTN can be trained on top of the best W4A4-adapted checkpoint.

This keeps the project from overreacting to a bad all-W4A4 baseline.

## Runtime Implication

R6000 should support layer-level and expert-level precision masks in the
W4A4/NVFP4-v2 manifest. The kernel path can still target native FP4 active MoE
for the majority of layers while falling back on BF16/FP8 for protected blocks.

The correct product is not "pure W4A4 at all costs". The correct product is the
fastest MoE artifact that passes strict quality gates.
