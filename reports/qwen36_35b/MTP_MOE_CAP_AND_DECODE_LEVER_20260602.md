# MTP on diverse-routing MoE is byte-capped — redirect to decode warp-split (2026-06-02)

> **⚠️ CORRECTION (2026-06-03):** any "Spark ceiling ~44 / 60+ needs SM120" framing
> below is **WRONG** — llama.cpp Q4_K_M does **69.77 on the same Spark (no FP4 MMA, no
> MTP)**, so ~70 IS reachable on Spark (measured BW ~240 GB/s; we're at ~37%, not
> saturated). The gap is BF16 attn-traffic + kernel overhead, both Spark-side software;
> SM120 is only for the 100–150 FP4-MMA tier. See
> `SPARK_BASELINE_DECODE_PROFILE_20260603.md` → "Why llama.cpp does 70 and we do 40".
> (The MTP-on-MoE *byte-cap* conclusion below is still correct and unaffected.)

## Question
Can the validated small-M / warp-split-K technique (16× on dense, cos=1.0 — see
`WARPSPLIT_SMALLM_SPARK_VALIDATED_20260602.md`) flip the MTP smoke's spec_k2
(0.45×, `MTP_NVFP4_35B_SMOKE_20260602.md`) to a net win toward 60 TPS on 35B-A3B?

## Probe (`scripts/spark_moe_verify_grouped_probe.py`, no model load, real shapes)
E=256, K=8, hidden=2048, gate_up=1024, inter=512; realistic random top-8
routing; BF16 (W4A16); per-position (current path) vs grouped gather+bmm:

| M (=K_draft+1) | unique experts | per-position | grouped | speedup | cos |
|---|---|---|---|---|---|
| 3 (k2) | **24 / 24** | 1672 µs | 2025 µs | 0.83× | 1.00000 |
| 5 (k4) | 38 / 40 | 2742 µs | 3307 µs | 0.83× | 1.00000 |
| 9 (k8) | 64 / 72 | 4919 µs | 5782 µs | 0.85× | 1.00000 |

## Finding — MTP cannot amortize MoE routed experts (structural, not a bug)
1. **Diverse routing.** At M verify positions the (pos,expert) pairs are ~all
   unique (24/24 at M=3). The verify must read ~M·K *distinct* expert weights —
   **M× a single decode's expert bytes**. At M=1 decode the MoE is HBM-bound, so
   the verify costs ~M× a decode *regardless of kernel*. No amortization exists:
   you must read every expert the M tokens use.
2. **Grouping is even slightly slower** (0.83×) via torch gather+bmm (the weight
   gather copies ~M·K·4 MB). A real grouped-GEMM kernel avoids the copy but still
   reads M× the bytes → at best ≈ per-position. **cos = 1.0** (the math is right;
   the economics aren't).
3. **Cross-check.** llama.cpp APEX (this exact model, `draft-mtp`, 63% accept) =
   **79 tok/s** vs Q4_K_M **69.77** = **+13%**, not 1.6×. MTP on this MoE is a
   ~+13% lever precisely because routed experts dominate decode and can't
   amortize. The earlier "MTP = 1.6×" framing was a **dense-model** number
   (FlashRT's Qwen3.6-27B is dense; every verify row shares one weight → 16×).

**Conclusion:** spec_k2 = 0.45× is not a fixable kernel bug — it is the verify
reading M× the experts. MTP's real ceiling on 35B-A3B is ~+13%, from amortizing
the *non-expert* dense parts (attention [already batched in k2], lm_head, norms,
shared expert). The small-M tile helps THOSE (validated 16×) and is worth keeping
for the dense verify parts — but it cannot rescue the routed experts.

## Redirect — the real 36 → 60 lever is BASELINE DECODE, not MTP
Accelerate the baseline NVFP4 decode (36–39 TPS) itself. The FlashRT technique
that transfers is **warp-split-K on the M=1 expert GEMVs** (their M=1 decode
+8.1%): each expert GEMV (`[1,2048]@[2048,1024]` gate_up; `[1,512]@[512,2048]`
down) at M=1 underfills the SMs; splitting the K reduction across warps —
**intra-block, shared-memory reduce** (FlashRT's way; *not* my probe's
grid-split + `atomic_add`, which hurt at large N) — fills them. My warp-split
probe already showed 1.68× on small-N shapes. This lever is:
- **Spark-native** (FP8/BF16 dequant path, no FP4 MMA);
- **routing-independent** (helps every decode, not just verify);
- **stackable** with the modest MTP +13% and the reusable decode graph.

## Next (the actual breakthrough build)
1. Add intra-block warp-split-K (shared-mem reduction) to the packed NVFP4 decode
   gate_up + down Triton kernels in `engine/moe_packed_nvfp4.py`, gated by env.
2. Micro-bench at the real expert GEMV shapes (gate_up K=2048 N=1024; down K=512
   N=2048) vs the current kernels; validate cos=1.0.
3. If >1.2×: wire into resident decode, re-run the e2e TPS smoke; target 36 → 50–60.

## UPDATE — warp-split-K does NOT apply to our decode kernels either (code-read)
Reading `triton_kernels/nvfp4_moe.py`: the gate_up + down decode kernels are
**not** single-warp `full_n` GEMVs (FlashRT's underfilled baseline). They are
**already tiled** grid = (inter/hidden-blocks × experts): gate_up = 64
inter-blocks × 8 = **512 programs**; down = 256 hidden-blocks × 8 = **2048
programs** — plenty to fill the GB10 SMs. The SMs are **not underfilled**, so
warp-split-K (FlashRT's fix for the underfilled `full_n` case) gives no
occupancy win here. FlashRT's M=1 +8% was over a *bad* baseline; ours is already
well-tiled. So the redirect above is **also capped** on Spark.

## Honest conclusion — the 36→69 gap is the FP4-MMA-less dequant-GEMV (arch limit)
The kernels dequant NVFP4→BF16 and **scalar-accumulate** (no tensor-core
`tl.dot`: M=1 has no 16-row tile to fill, and Spark sm_121 has no FP4/4-bit MMA).
They are memory-bound on the packed reads + the dequant. The gap to llama.cpp
Q4_K_M (69.77) is **vendor kernel efficiency for 4-bit dequant GEMV on hardware
without 4-bit MMA** — a fundamental Spark limit, not a missing trick.

This **reinforces the standing strategy**: Spark = long-context / fallback /
multi-service host; the FP4-MMA performance story (150 tok/s, FlashRT-proven on a
single RTX 5090) belongs on **SM120** (RTX 5090 ~$2k / the R6000 we lost). On
Spark the realistic wins are modest + stackable: MTP **+13%**, reusable decode
graph **+10%**, and **Spark-specific kernel re-tuning** — the gate_up/down
`BLOCK_*` + `num_warps` are locked to the R6000-best config (see the
`LYNN_MOE_FAST_FIXED` guard), which may not be Spark-optimal. That config sweep
is the one cheap, unexplored decode knob; everything else needs SM120 hardware.

## Config sweep RESULT — a real Spark-specific decode win (measured)
`scripts/spark_moe_decode_config_sweep.py` swept the real gate_up + down kernels
at real shapes on Spark (no model load):

| kernel | R6000-locked | Spark-best | gain |
|---|---|---|---|
| gate_up | 95.0 µs (bi=8 bh=256 nw=4) | 92.6 µs (bh=128) | 1.03× |
| **down** | 55.7 µs (bh=8 bi=512 nw=8) | **41.3 µs (bh=4)** | **1.35×** |

The down kernel's `block_hidden=8` (R6000-best) is **suboptimal on Spark;
`block_hidden=4` is 1.35×**. gate_up is already near-optimal. Combined routed
GEMV 150.7 → 134 µs = **1.12×**. This is a real, zero-code (config) decode win —
wired behind `LYNN_MOE_DOWN_BLOCK_HIDDEN=4` (default 8 unchanged) + the Spark-best
tuple added to the `LYNN_MOE_FAST_FIXED` guard (commit `b4657ca`).

**E2e VALIDATED** (same smoke + conditions as the baseline, APEX stopped):
baseline decode **36.10 → 40.06 TPS = +11%** — better than the ~+5–10% estimate
(the down kernel is a bigger decode fraction than assumed). spec_k1-seq stays
token-exact (bh=4 is a reduction *tile size*; float accumulation order differs
slightly from bh=8 but the math is equivalent). MTP configs stay slowdowns
(capped, as shown above). **New Spark realistic ceiling:** 40 × 1.10 (reusable
graph) ≈ **44 TPS**; MTP doesn't stack (slowdown on MoE). **60+/150 still needs
SM120** (FlashRT-proven 150 on a single RTX 5090).

## Probes that produced this (all on Spark sm_121, committed)
- `spark_warpsplit_fp8_gemm_probe.py` — small-M 16× (dense), warp-split 1.68×
  (small-N **underfilled**), cos=1.0. The technique works *where it applies*.
- `spark_moe_verify_grouped_probe.py` — MoE routed verify can't amortize (0.83×,
  24/24 unique). The MTP-on-MoE ceiling.
- Code read of `triton_kernels/nvfp4_moe.py` — our decode kernels are already
  well-tiled → warp-split-K n/a → the gap is the arch-level dequant-GEMV limit.
