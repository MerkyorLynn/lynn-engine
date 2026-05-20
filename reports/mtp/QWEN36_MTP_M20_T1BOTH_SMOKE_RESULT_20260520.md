# Qwen3.6-35B MTP M20 t1-both Smoke Result

Date: 2026-05-20

## Result

| Config | Accept | Effective TPS | Baseline-loop TPS | Exact |
|---|---:|---:|---:|---:|
| `baseline` |  |  | 25.9218 | 6/6 |
| `shadow` |  |  | 23.8610 | 6/6 |
| `spec_k1` | 0.7513 | 20.9934 |  | 6/6 |
| `spec_k1_batched` | 0.7517 | 19.8386 |  | 2/6 |

## Interpretation

- `LYNN_FULL_ATTN_K2_BACKEND=t1_loop` plus `LYNN_MTP_K2_LINEAR_ATTN_MODE=t1_loop` does not improve over M19.
- Batched accept remains healthy at about 75%, confirming the official head and draft contract are good.
- Batched exactness remains 2/6 and effective TPS remains below baseline, so the remaining bug is not fixed by layer-type T=1 bridge alone.
- Next work should inspect speculative commit/state rollback/output equivalence, not keep sweeping full-attn vs linear-attn knobs.

## Artifact

- `reports/mtp/remote_spark_20260520/mtp_smoke_m20_t1both_20260520_111345.json`
