# Qwen3.6-35B MTP M12 lm_head Opt-In Result · 2026-05-20

## Verdict

M12 fixes the M10/M11 regression by restoring final-token-only `lm_head` as the default and making all-position logits explicit for MTP K=2. The official Qwen3.6 MTP head is now clearly usable in Lynn: shadow and sequential speculative accept are high, but batched K=2 is still not numerically exact enough to promote.

## Metrics

| Config | Exact | Mean prefix | Accept | Effective TPS / decode TPS | TPS ratio vs baseline |
|---|---:|---:|---:|---:|---:|
| baseline | 6/6 | 100.83 | n/a | 26.74 decode | 1.00 |
| shadow | 6/6 | 100.83 | 81.44% | 23.92 decode | n/a |
| spec_k1 sequential | 6/6 | 100.83 | 75.13% | 20.87 effective | 0.780 |
| spec_k1_batched | 2/6 | 42.83 | 73.17% | 19.86 effective | 0.743 |

## Gates

- Sequential correctness: `True`.
- Batched correctness: `False`.
- Batched accept recovered from the M9 15.39% failure to 73.17%, so the remaining blocker is no longer MTP head quality; it is K=2 verifier numerical/semantic drift.

## Next

1. Keep MTP opt-in only.
2. Run a K2-vs-two-T1 diff probe to locate the first layer or boundary where batched verify diverges.
3. Prioritize full-attention K2 and linear/GDN state update equivalence before any TPS promotion run.
