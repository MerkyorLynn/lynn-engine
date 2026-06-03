# Spark NVFP4 35B baseline decode — profile + optimization levers (2026-06-03)

Goal (user-picked direction): speed up the **baseline** NVFP4 decode on Spark
(routing-independent; no MTP, no FP4 MMA, no 5090) — 36 → toward llama.cpp 69.77.

## ⚠️ CORRECTION (2026-06-03) — there is NO "Spark hardware ceiling at 40"
An earlier draft framed "~40 TPS ceiling, 60+ needs SM120" as a *hardware* limit.
**That was wrong** and is retracted: **llama.cpp Q4_K_M does 69.77 on the SAME Spark,
no FP4 MMA, no MTP** — so the hardware plainly does ≥70 on the dequant path. 40 is
**our engine's current-kernel** number, not Spark's.

### Why llama.cpp does 70 and we do 40 — quantified (measured)
- **Spark memory BW (measured): ~240 GB/s** (copy 243 / read 230; `spark_decode_tps`-style probe).
- 35B-A3B active ≈ ~3B params/token → **all-4-bit weights ≈ 1.5 GB/tok → BW-bound ceiling ≈ 160 TPS.**
- **llama.cpp 70 TPS** moves ~1.6 GB (all-4-bit Q4_K) → ~112 GB/s = **~47% of BW**.
- **us 40 TPS** move ~2.2 GB → ~88 GB/s = **~37% of BW**. We read MORE because our
  **full-attn q/k/v/o projections are BF16** (the qkv-prep `torch.cat`s BF16 tensors;
  only the MoE experts + linear-attn in-proj are 4-bit). llama.cpp keeps everything 4-bit.
- **Neither is BW-saturated** (both < 50% of 240 GB/s) → at M=1 decode both are
  **latency/overhead-bound, not memory-bound**. So 40 has ~2.7× BW headroom.

**The 1.75× gap decomposes:** ≈**1.5× from extra bytes** (our BF16 attn-proj + any BF16
lm_head vs llama.cpp's all-4-bit) + ≈**1.2× from kernel/launch overhead** (our Triton
"~140 tiny launches/token" vs ggml's fused CUDA). **Both are Spark-side software fixes,
not hardware.** SM120/FP4-MMA is only for the *100–150* tier, never for 70.

**∴ Realistic Spark target = ~70 (llama.cpp-proven), via: (1) quantize the full-attn
q/k/v/o (and lm_head) to 4-bit/packed NVFP4 — kill the BF16 traffic; (2) cut kernel
launch overhead to match ggml.** The "ceiling 40-48 / 60 needs SM120" lines below are
SUPERSEDED by this correction.

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
- **Our *current-kernel* number ≈ 40–41 TPS — NOT a hardware ceiling** (see CORRECTION up top: llama.cpp = 69.77 on the same Spark, no FP4 MMA). 40→70 is 4-bit attn-proj + tighter kernels, all Spark-side; SM120 only for the 100–150 FP4 tier.

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

Stacked (bh4 × flags) ≈ **~40–41 TPS = our current-kernel number, NOT the hardware
ceiling** (llama.cpp 69.77 on the same Spark — see CORRECTION up top).

## Next (incremental, the only Spark path)
1. **Wire qkv fusion** (prep fused weight + `LYNN_FULL_ATTN_QKV_FUSED=1`), e2e-measure.
2. Same fusion for o_proj / the linear-attn in-proj (more 3→1 launch cuts).
3. Per-kernel Spark config sweeps on the remaining GEMVs (bh=4 proved R6000-locked configs are suboptimal here).
4. (Infra note: stopping APEX then immediately loading the 90G NVFP4 model races the GPU release → use a longer wait or confirm GPU-free before load; microbenches avoid this entirely.)

## UPDATE — qkv fusion e2e A/B: NEGLIGIBLE (+0.3%), and profiling overstated headroom
Wired it (it was an existing opt-in flag, no code change) and ran a **clean same-
process A/B** (qkv OFF→ON, same prompt/warmup, `spark_qkv_fusion_e2e_ab.py`):
- **A (off) 38.82 TPS → B (on, 10 layers fused) 38.93 = +0.3%.** Coherent both;
  not bit-identical (fused GEMV's accumulation order flips a few greedy argmaxes —
  same math, quality-neutral).
- The isolated microbench's **1.44× did NOT translate**: the profiler's per-section
  `cuda-sync` **inflated** the section times. The "qkv = 2.2 ms" was a profiling
  artifact; the real qkv is small, so fusing saves ~90 µs/tok ≈ 0.3%.

**Sharper conclusion:** the only e2e-real win is **bh=4 (+11%, 36→40, measured clean)**.
qkv fusion +0.3% (skip). The linear-attn flags' +2.7% was *instrumented* and likely
smaller clean. **The profiled 22.6 ms "remainder" overstated headroom** — under clean
(unprofiled) decode there's much less to squeeze *within our current kernel
architecture* (~40–41). **But that is OUR-kernel, NOT Spark's ceiling** — see the
CORRECTION up top: llama.cpp 69.77 on the same hardware proves ~70 is reachable; the
gap is BF16 attn-traffic + kernel overhead, both Spark-side. Two lessons: (1) trust
**clean e2e A/B**, not profiled-section deltas; (2) **always benchmark against
llama.cpp e2e** so a kernel-efficiency gap is never again mistaken for a hardware limit.
