# MTP on diverse-routing MoE is byte-capped — redirect to decode warp-split (2026-06-02)

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

## Probes that produced this (all on Spark sm_121, committed)
- `spark_warpsplit_fp8_gemm_probe.py` — small-M 16× (dense), warp-split 1.68×
  (small-N), cos=1.0. The technique works.
- `spark_moe_verify_grouped_probe.py` — MoE routed verify can't amortize (0.83×,
  24/24 unique). The MTP-on-MoE ceiling. The redirect.
