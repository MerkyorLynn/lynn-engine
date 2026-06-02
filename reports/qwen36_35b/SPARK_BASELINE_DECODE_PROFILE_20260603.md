# Spark NVFP4 35B baseline decode — profile + optimization levers (2026-06-03)

Goal (user-picked direction): speed up the **baseline** NVFP4 decode on Spark
(routing-independent; no MTP, no FP4 MMA, no 5090) — 36 → toward llama.cpp 69.77.

## Profile (LYNN_MTP_PROFILE, 64-tok decode, production fast-path + bh=4)
Instrumented decode = **28.4 ms/tok (35.18 TPS)** — note: the profile's per-section
cuda-sync adds overhead, so this is ~12% slower than the clean 40.06 TPS; the
*proportions* are what matter.

| section | ms/tok | share |
|---|---:|---:|
| all instrumented attention (qkv 2.2, o_proj 2.33, rope 0.5, sdpa 0.36, gate 0.23, cache 0.17) | **5.8** | **20%** |
| **remainder** (MoE + lm_head + linear-attn + norms + glue + launch overhead) | **22.6** | **80%** |

Key structural facts:
- `count=630 / 64 ≈ 10` full-attn layers/token → the model is **hybrid**:
  ~10 full-attention + **~38 Gated-DeltaNet linear-attention** layers.
- MoE GEMV ≈ 6.4 ms (config sweep: 134 µs × 48 layers).
- So the 22.6 ms remainder ≈ MoE 6.4 + lm_head (~1.5) + ~38 linear-attn + glue +
  **launch overhead across ~140 tiny kernels/token**. The code itself says it:
  *"decode is dominated by many tiny launches across 30 linear-attn layers"*.

**The bottleneck is launch overhead + many tiny kernels, not big-GEMM compute.**

## Levers (measured)
1. **bh=4 down-kernel config — BANKED: 36 → 40 TPS (+11%)** (`LYNN_MOE_DOWN_BLOCK_HIDDEN=4`, commit `b4657ca`/`c3e4cbc`). The R6000-locked down config was suboptimal on Spark.
2. **Linear-attn opt-in flags — +2.7%, coherent** (`LYNN_LINEAR_ATTN_GQA_RECURRENT=1` + `LYNN_LINEAR_ATTN_RECURRENT_FROM_OUTCONV=1`): 35.18 → 36.12 instrumented, output correct. Trims the ~38 linear-attn layers' q/k materialization + recurrent prep. Stackable.
3. **Reusable decode graph — the big lever for launch overhead** (`LYNN_REUSABLE_DECODE_GRAPH=1`, M3): capture-once / replay-many eliminates the ~140 per-token kernel launches (FlashRT puts "all kernel outputs into CUDA Graph"). **Under test on NVFP4** — the open question is whether the `fixed_triton` NVFP4 MoE dispatch is host-sync-free enough to capture (the FP8 path needed `LYNN_FP8_MOE_GRAPH_SAFE`; NVFP4 `fixed_triton` passes `expert_ids` as a tensor, so it may already capture).

## Next
- If the reusable graph captures NVFP4 + stays coherent → measure the launch-overhead win (likely the largest single lever for the 22.6 ms remainder).
- If it errors on a host-sync → make the NVFP4 MoE dispatch graph-safe (P1-style fixed-K, no `torch.unique/.tolist`), then capture.
- Stack: bh=4 (+11%) × graph (?) × linear-attn flags (+2.7%) → the realistic Spark ceiling (~44+ TPS).
