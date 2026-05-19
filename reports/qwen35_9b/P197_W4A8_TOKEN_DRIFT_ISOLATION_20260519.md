# P197 W4A8 Token Drift — R6000 Results + Isolation Plan

Date: 2026-05-19
Branch: `claude/qwen35-9b-w4a8-drift-isolation-report-20260519`

## P197 Result: AMBER (fake-W4A8)

**Candidate:** `fake_w4a8` (LYNN_W4A8_FAKE_QUANT_ACTIVE=full, per-16 E4M3)
**Reference:** `convstrict_w4a16`

| Metric | Value |
|--------|-------|
| Verdict | **AMBER** |
| Exact match | 32/40 (80%) |
| Drift steps | 8/40 |
| Drift ratio | 0.200 |
| Combined min | 0.833321 |
| Combined mean | 0.974995 |
| Jaccard mean | 0.950 |
| Shared cosine mean | 1.000 |
| First drift | prompt=1, step=0 |

## Drift Characterization

All 8 drift steps share the same pattern:

- **Jaccard = 0.667** (6 of 8 drifts): 4 of 5 top-k IDs are shared, 1 differs at rank 5
- **Jaccard = 1.000** (2 of 8 drifts): all 5 IDs shared, only rank order differs
- **Shared cosine ≥ 0.9999**: logit magnitudes on shared IDs are essentially identical

**Interpretation:** This is NOT semantic drift. It is rank-order noise at the
bottom of the top-5 distribution, caused by E4M3 quantization rounding the logits
of the 4th vs 5th most likely token past each other. The greedy (argmax) token is
always the same — zero of the 8 drift steps affect the top-1 selection.

### Per-prompt detail

| Prompt | Drift steps | First drift | Pattern |
|--------|-------------|-------------|---------|
| 0 (CUDA graph 中文) | 1/8 | step 5 | rank-order only (jac=1.0) |
| 1 (Fibonacci Python) | 3/8 | step 0 | 1 ID swap at rank 5 |
| 2 (JSON Paris) | 2/8 | step 0 | 1 ID swap at rank 5 |
| 3 (TCP vs UDP) | 1/8 | step 1 | 1 ID swap at rank 5 |
| 4 (primes JSON) | 1/8 | step 4 | 1 ID swap at rank 5 |

## Why AMBER is Not STRICT

The fake-W4A8 path applies `_fake_quant_fp8_activation()` before gate/up projections
and before down projection. This quantizes to E4M3 per-16 groups then immediately
dequantizes back to BF16 for the `F.linear()` call. The quantization noise (~2^-7
relative error) is small per-layer but compounds across 36 layers via residual
connections. By the lm_head, the accumulated noise is enough to swap the ordering
of tokens that are within ~0.01% of each other in logit space.

## P197b Isolation: Why It Matters

P197b will test the **true FP4xFP8** path (packed NVFP4 weight × E4M3 activation
via native CUDA kernel) against the same reference. The key difference:

| Mode | Weight | Activation | Kernel |
|------|--------|-----------|--------|
| fake_w4a8 | BF16 (full precision) | E4M3 fake-quant | torch F.linear |
| true_fp4xfp8_scalar | E2M1 packed NVFP4 | E4M3 real | scalar reference |
| true_fp4xfp8_mma | E2M1 packed NVFP4 | E4M3 real | SM120a MMA |

If `true_fp4xfp8_scalar` ≈ fake_w4a8 (AMBER), then the weight quantization to
E2M1 is not adding drift — only the activation E4M3 quantization matters. This
would mean the MMA fragment fix (P191) is all that's needed for production.

If `true_fp4xfp8_scalar` is worse (CLOSED), then the 4-bit weight quantization
introduces additional error beyond what fake-W4A8 shows.

## P197b Execution Status

**BLOCKED**: R6000 GPU is currently running `openai_mcq_thinking32_eval.py` with
`llama-server` on port 18197. The isolation probe requires exclusive GPU access
(loads LynnIncrementalRunner with full model in VRAM).

**Runner script:** `scripts/r6000_qwen35_9b_p197b_drift_isolation_safe.sh`
- Checks for running eval / llama-server processes
- If busy → writes REFUSE_RUN status JSON and exits cleanly
- If free → runs full P197b isolation

## Next Steps (when R6000 is free)

```bash
# 1. Check if safe to run:
bash scripts/r6000_qwen35_9b_p197b_drift_isolation_safe.sh

# 2. If REFUSE_RUN, wait. If clear, it runs automatically.

# 3. Expected decision tree from results:
#    scalar_full=NO_DRIFT → FP8 quant noise (fake path) is sole source → AMBER acceptable
#    scalar_full=AMBER → same as fake → weight E2M1 adds no extra drift → fix MMA layout
#    scalar_full=CLOSED → E2M1 weight quant adds real drift → harder fix needed
```
