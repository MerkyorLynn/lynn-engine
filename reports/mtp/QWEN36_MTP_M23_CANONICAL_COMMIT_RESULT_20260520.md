# Qwen3.6-35B MTP M23 Canonical Commit Smoke — 2026-05-20

## Result

M23 tests the opt-in diagnostic pair:

- `LYNN_FULL_ATTN_K2_BACKEND=t1_loop`
- `LYNN_MTP_K2_ACCEPT_SOURCE=canonical_t1`
- `LYNN_MTP_K2_COMMIT_SOURCE=canonical_t1`

This keeps the K=2 draft/verify probe in the loop, but commits the accepted
state using the canonical sequential T=1 chain. It is intentionally slow and is
not a production path.

| config | exact | accept | effective TPS | ratio vs baseline |
|---|---:|---:|---:|---:|
| baseline | 6/6 | — | 25.85 decode TPS | 1.00 |
| shadow | 6/6 | 82.72% shadow | — | — |
| spec_k1 sequential | 6/6 | 77.25% | 21.26 | 0.82 |
| spec_k1_batched + canonical commit | **6/6** | **77.25%** | **7.85** | **0.30** |

## Interpretation

M23 restores batched speculative exactness to 6/6. That rules out the official
MTP head, concat order, full-attn K=2 layer drift, linear-attn K=2, lm_head
all-position dispatch, and reject rollback as the remaining correctness lock.

The residual M19/M20/M22 exact drift lives in the **K=2 accept/commit state**:
accepting directly from the K=2 verifier's `h_k2[:, 1]`, `argmax_at_pos1`, and
mutated decode state is not equivalent to accepting through the canonical T=1
chain. When the accepted state is rebuilt via canonical T=1, token exactness is
fully restored.

This is a correctness breakthrough but not a speed breakthrough yet. The M23
path is slow because it pays for the K=2 verifier and then replays canonical T=1
on accepted events.

## Next Probe

M24 should shrink canonical commit into the minimum necessary state repair:

1. Keep K=2 accept decision for speed accounting.
2. On accept, selectively replace only the committed hidden/next-pending source
   with canonical T=1 values, then test exactness and TPS.
3. If hidden repair alone is insufficient, compare recurrent/conv/KV deltas after
   K=2 accept versus canonical T=1 accept to find the smallest state patch.

Do not promote M23 as default; use it as the correctness bridge for M24.

## Artifact

- `reports/mtp/remote_spark_20260520/mtp_m23_canonical_commit_smoke_20260520_124807.json`
