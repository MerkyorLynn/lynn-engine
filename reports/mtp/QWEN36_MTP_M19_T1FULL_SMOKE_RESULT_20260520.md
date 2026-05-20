# Qwen3.6-35B MTP M19 t1-full-attn Smoke Result

Date: 2026-05-20

## Result

| Config | Accept | Effective TPS | Baseline-loop TPS | Exact |
|---|---:|---:|---:|---:|
| `baseline` |  |  | 26.1076 | 6/6 |
| `shadow` |  |  | 24.1682 | 6/6 |
| `spec_k1` | 0.7513 | 21.0330 |  | 6/6 |
| `spec_k1_batched` | 0.7517 | 19.8045 |  | 2/6 |

## Interpretation

- `LYNN_FULL_ATTN_K2_BACKEND=t1_loop` restores batched MTP accept to the sequential-head range: about 75%.
- This confirms M18: full-attention K=2 batching is the unsafe source; linear-attention K=2 can remain enabled for this bridge.
- `spec_k1_batched` is still not promotable: exactness is 2/6 and effective TPS is below the baseline loop.
- MTP is now a real head/contract success, but serving speed still needs an exact and cheaper K=2 verifier.

## Artifacts

- `reports/mtp/remote_spark_20260520/mtp_smoke_m19_t1full_20260520_110033.json`
