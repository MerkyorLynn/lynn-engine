# Qwen3.6 MTP M17 Canonical K=2 Result - 2026-05-20

## Verdict

`LYNN_MTP_K2_VERIFY_MODE=t1_canonical` restores batched-mode token exactness by routing the `spec_k1_batched` entry point through the canonical two-step T=1 verifier.

This is not a speed win yet. It is the correctness scaffold needed to safely recover K=2 speed layer by layer.

## Source

- Remote Spark JSON: `reports/mtp/remote_spark_20260520/mtp_m17_t1_canonical_smoke_20260520_034507.json`
- Code path: `engine/mtp_serving.py`

## Smoke Result

Spark, Qwen3.6-35B-A3B Lynn-native W4A16 NVFP4, official fused MTP sidecar, 6 prompts, max_new 96:

| Config | Exact | Mean Accept | Mean TPS | Ratio vs Baseline |
|---|---:|---:|---:|---:|
| baseline | 6/6 | n/a | 25.79 | 1.00x |
| shadow | 6/6 | 81.66% | 23.86 | n/a |
| spec_k1 sequential | 6/6 | 74.04% | 20.75 | 0.804x |
| spec_k1_batched with `t1_canonical` | 6/6 | 74.04% | 20.74 | 0.804x |

## Interpretation

The MTP head, concat order, accept/reject bookkeeping, and official sidecar conversion are now proven usable. The previous batched K=2 failure is not a semantic MTP-head problem.

M16 showed that the true K=2 verifier diverges immediately from two canonical T=1 steps. M17 proves that if the batched entry point uses the canonical T=1 verifier, token exactness returns.

Therefore the next acceleration work should be incremental:

1. Keep `t1_canonical` as the correctness oracle.
2. Re-enable K=2 batching layer-by-layer.
3. After each layer group, run the small M16 probe before running full smoke.
4. Only when full smoke remains exact should we measure end-to-end TPS.

The current `t1_canonical` mode should not be promoted for speed because it intentionally performs the sequential two-forward verifier. Its value is that it locks the expected behavior while K=2 kernels are repaired.

## Next Patch Direction

Start from the earliest M16 drift:

- zero-advance repro first bad layer: layer 5, `linear_attention`
- other short prefixes alternate between early `linear_attention` and `full_attention`
- downstream state drift often peaks in conv state around layers 20/38

The likely next candidate is a strict K=2 verifier mode that uses canonical T=1 for linear-attention layers first, while allowing only one carefully selected layer group to use K=2. This turns the current all-or-nothing verifier into a bisectable performance path.
