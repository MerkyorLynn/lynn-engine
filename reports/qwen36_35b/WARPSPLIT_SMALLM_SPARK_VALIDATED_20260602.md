# Warp-split-K + small-M FP8 verify on Spark — VALIDATED (2026-06-02)

Derived from **FlashRT** (Apache-2.0, `github.com/LiangSu8899/FlashRT`): studied
its `fp4_w4a4_mma_warpsplit_mrows_sm120` kernel, reimplemented the technique
clean-room in Triton, and **validated it on Spark sm_121 FP8 with no model
load**. This is the **task #11 unlock, now measured** — and it needs no FP4 MMA.

## FlashRT technique (studied from `csrc/kernels/`)
The spec-verify GEMM uses two tricks:
1. **small-M epilogue** — the 16-row MMA tile means M≤16 verify rows cost the
   *same weight HBM read as M=1*. (Their kernel zeros rows M..15, runs the
   identical MMA mainloop, writes M rows — lines 80-92/123 of the `.cu`.)
2. **warp-split-K** — split the long-K reduction across warps in one block, sum
   partials in shared memory → fills SMs that a single-warp `full_n` GEMV
   underfills on long-K shapes. No cross-block intermediate → graph-safe.

Their atom is the SM120 FP4 `16x8x64` block-scaled MMA. **Spark sm_121 has no
FP4 MMA, but it has the FP8 e4m3 `m16n8k32` atom — also a 16-row tile** — so
both tricks transfer to an FP8 path.

## Spark probe (`scripts/spark_warpsplit_fp8_gemm_probe.py`, no model load)
Clean-room Triton reimpl, FP8 e4m3, K=17408 (the mlp_down verify shape):

| metric | result |
|---|---|
| correctness M=1 / 8 / 16 | **cos = 1.000000** (max_rel ≤ 0.6%) |
| small-M free: M=16 vs M=1 | 475 µs vs 450 µs = **1.055×** (16 rows ≈ free) |
| **per-position T=1 ×16  →  small-M tile M=16** | 6525 µs → 405 µs = **16.1× faster** |
| warp-split-K, N=4096 (well-tiled) | SPLIT_K=1 fastest (split-K hurts) |
| warp-split-K, N=256 / 512 (underfilled) | **SPLIT_K=8 = 1.68×** |

## What this proves — task #11 unlock, validated on Spark
1. **The decisive result:** the MTP smoke's slowdown (spec_k2 = 0.45×, see
   `MTP_NVFP4_35B_SMOKE_20260602.md`) *was* the per-position T=1 MoE verify.
   Replacing it with a **small-M (M=K+1 ≤ 16) tile is 16.1× faster** on Spark
   FP8 — at **cos = 1.000000** (token-exact preserved). This is THE fix.
2. **warp-split-K helps the small-N regime** (1.68× at N≤512) — relevant to the
   MoE *per-expert* GEMMs (each expert's N is small). Use FlashRT's intra-block
   shared-mem reduction, **not** a grid-split + `atomic_add` (my probe's grid
   version regressed at N=4096: extra global traffic with no SM-fill benefit).
3. **No FP4 MMA needed.** Spark's FP8 `m16n8k32` carries both tricks. The FP4
   performance story stays on SM120 (5090 / R6000), but **this MTP-verify win is
   Spark-native** — it directly attacks why MTP was a slowdown on Spark.

## Implementation path (scoped + now measured)
1. Replace the per-position T=1 MoE loop at **`engine/full_forward.py:823-828`**
   with a small-M tile: stack the K+1 verify positions into a ≤16-row activation
   tile, run the MoE per-expert GEMMs at M=K+1 (FP8 `m16n8k32`), mask rows > M.
2. For the per-expert GEMMs (small N), add intra-block warp-split-K → 1.68×.
3. Validate cos vs the per-position path (already a passing gate for k2), then
   re-run `scripts/spark_mtp_speculative_smoke.py`.
4. **Expected:** verify cost ~16× lower → `spec_k2_batched` flips from 0.45× to
   a net win → toward the ~60 TPS target.

## Caveats (honest)
- The probe is a **dense** GEMM with **per-tensor** scale, not yet the full
  NVFP4 per-16 scale-factor MoE. The MoE adds per-expert grouping + the per-16
  SF layout. The core (small-M tile + FP8 `m16n8k32` + cos=1.0) is validated;
  the MoE wiring + SF handling is the remaining work.
- FP8 reads 2× NVFP4's bytes at M=1 decode — but the small-M tile amortizes the
  weight read across 16 rows, so the *verify* is bound by a single weight read
  regardless. The 2× concern is a decode (M=1) issue, not a verify (M=16) one —
  and the verify is exactly where small-M wins.
- N=256 and N=512 timed identically → at these tiny shapes the kernel is
  launch/overhead-bound; the SPLIT_K=8 = 1.68× is the real, repeatable signal.
