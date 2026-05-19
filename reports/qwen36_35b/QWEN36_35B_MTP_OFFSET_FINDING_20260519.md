# Qwen3.6-35B-A3B MTP Offset Finding - 2026-05-19

## Verdict

The current Qwen3.6-35B-A3B MTP sidecar is not a usable speed lever for Lynn engine serving yet.

The serving wire is coherent, but the available head behaves like an offset-1 mirror head while speculative decode needs an offset-2 lookahead head. As a result, accept-rate collapses to random-match territory and MTP must remain a gated research branch.

## Spark Smoke Result

| Run | Baseline TPS | Spec K=1 TPS | Accept Rate | Correctness | Notes |
|---|---:|---:|---:|---|---|
| eager-vs-eager | 25.96 | 12.08 | 0.30% | 16/16 token-exact | missing full Config D env |
| Config D baseline | 40.60 | 14.61 | 0.30% | graph-vs-eager drift artifact | baseline env fixed |

The second run confirms that the Spark Lynn-native W4A16 baseline can return to the expected Config D class. It does not rescue MTP accept-rate.

## Root Cause

The training/calibration path supervised the MTP head against the base model's same-position prediction:

- input: hidden at position N
- label: `lm_head(base_hidden_N)`, which predicts token N+1

That is an offset-1 mirror of the base model's next-token distribution.

Speculative serving for a one-token draft step needs:

- after accepting / verifying token N+1, compare the draft for token N+2
- therefore the head must learn an offset-2 target

The current head predicts N+1 while the verifier is evaluating N+2, so agreement falls to about 0.30%.

## What This Means

- Do not count MTP credit in the 35B TPS roadmap.
- Do not use the current sidecar to justify a promotion decision.
- Keep `engine/mtp_serving.py` and the smoke harness as useful wiring scaffolding.
- Treat the current sidecar as a warm-start/diagnostic artifact only.

## Unlock Path

The next meaningful MTP task is an offset-2 retrain/calibration run:

1. teacher-force token N+1 as the MTP input context
2. label the MTP head with the base model prediction for token N+2
3. rerun the same Spark smoke
4. only continue to K=2 / batched verify after accept-rate is real

Until then, 35B speed work should stay on W4A16 boundary coarsening, native MoE repack, and full-attn graph integration.

