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
3. **Reusable decode graph — TESTED, NET NEGATIVE on NVFP4.** `LYNN_REUSABLE_DECODE_GRAPH=1` (+ flags + bh4) = **32.06 TPS vs bh4-clean 40.06** (output coherent — it captured and ran correctly, just slower). The NVFP4 dispatch IS graph-safe (code scan confirms: `torch.topk(out=...)` router, tensor `expert_ids`, no `.tolist`), so it captured — but the graph **requires `LYNN_FULL_ATTN_FIXED_SHAPE`** (recompute attention over the *full* KV cache every step), and that overhead **exceeds** the launch-overhead it saves. So the graph is the **wrong lever** here, unlike FlashRT (whose win is the FP4 MMA, not the graph alone).

## Revised conclusion — the remainder is DISTRIBUTED, no silver bullet
attention 5.8 + MoE ~6.4 + linear-attn ~6 + lm_head ~1.5 + glue ≈ 22 ms. There is
**no single 50% bottleneck** — so no one optimization gives a big jump; wins are
incremental and must stack across attention/MoE/linear-attn kernels.

## Levers, final
- **bh=4 down config: +11% (36→40), BANKED.** ✅
- **linear-attn GQA+outconv flags: +2.7%, coherent.** ✅ (stacks with bh=4)
- reusable graph: ✗ net-negative (fixed-shape-attn cost).
- Stacked best = bh4 × linear-attn flags ≈ **~41 TPS** (clean re-measure in flight).
- **Realistic Spark ceiling via incremental kernel tuning ≈ 44–48 TPS; 60+/150 needs SM120** (no FP4 MMA on sm_121 → dequant-GEMV memory-bound, can't match vendor Q4_K_M 69.77). This re-confirms the standing strategy: Spark = long-ctx/fallback; FP4 perf story on 5090/R6000.

## qkv fusion — CONFIRMED lever (microbench, 1.44×, token-exact)
`spark_qkv_fusion_probe.py` (no model load): the qkv (2.2 ms) runs **3 separate**
q/k/v GEMVs unless `LYNN_FULL_ATTN_QKV_FUSED=1`. Fusing 3→1 GEMV = **29.5 → 20.5 µs
(1.44×), exact=True**. Across ~10 full-attn layers that's ~0.7 ms/tok → **~+2.5%
e2e**, token-exact (pure launch/BW reduction). **Ready to wire**: needs the runner to
prep the fused `self_attn._qkv_proj.weight` + set the flag (the o_proj could get the
same treatment).

## Confirmed stackable levers (all measured)
| lever | gain | status |
|---|---|---|
| bh=4 down config | +11% (36→40) | **BANKED + wired** (`b4657ca`) |
| linear-attn GQA+outconv flags | +2.7%, coherent | measured, opt-in |
| qkv fusion (3→1 GEMV) | 1.44× on qkv → ~+2.5% e2e, exact | **confirmed, ready to wire** |
| reusable decode graph | ✗ net-negative (−20%) | rejected (fixed-shape-attn cost) |

Stacked (bh4 × flags × qkv) ≈ **~42 TPS**. Ceiling via incremental tuning ≈ 44–48;
**60+/150 needs SM120** (no FP4 MMA on sm_121).

## Next (incremental, the only Spark path)
1. **Wire qkv fusion** (prep fused weight + `LYNN_FULL_ATTN_QKV_FUSED=1`), e2e-measure.
2. Same fusion for o_proj / the linear-attn in-proj (more 3→1 launch cuts).
3. Per-kernel Spark config sweeps on the remaining GEMVs (bh=4 proved R6000-locked configs are suboptimal here).
4. (Infra note: stopping APEX then immediately loading the 90G NVFP4 model races the GPU release → use a longer wait or confirm GPU-free before load; microbenches avoid this entirely.)
