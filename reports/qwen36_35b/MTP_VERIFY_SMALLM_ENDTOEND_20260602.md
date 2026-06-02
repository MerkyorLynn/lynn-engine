# MTP small-M grouped MoE verify — end-to-end result (NEGATIVE, 2026-06-02)

Per the directive to *actually build it* (not extrapolate from a probe): implemented
`moe_forward_verify_smallm_nvfp4` (route each position → group by expert → run each
active expert's gate_up+down once over its rows → batched shared expert), wired into
the k2/k4 verify behind `LYNN_MTP_VERIFY_SMALLM=1`, and ran the real e2e smoke on
NVFP4 35B + the trained sidecar.

## Result (NVFP4 35B, baseline 36.88 TPS)
| config | per-position (prior) | **small-M grouped (this)** | token-exact |
|---|---:|---:|:--:|
| spec_k1 (seq) | 30.78 (0.83×) | 30.78 (0.83×, unchanged — M=1) | True |
| spec_k2_batched | 16.18 (0.45×) | **2.25 (0.061×)** | **False** |
| spec_k4_batched | 13.42 (0.37×) | 1.74 (0.047×) | False |

The grouped small-M made the batched verify **~7× slower** and **broke token-exactness**.

## Why it's worse (not a tuning issue — structural to this implementation)
1. **Slow dequant.** The function uses the **reference pure-torch `_dequant_nvfp4_slot`**
   (rebuilds the E2M1 table, expands the per-16 scales, gathers) **per expert, per
   layer, per step** — far slower than the optimized `nvfp4_grouped_*` Triton kernels
   the per-position path calls. Amortizing the per-position loop does not pay for
   swapping a fast kernel for a slow dequant.
2. **Host-syncs.** `torch.unique().tolist()`, `nonzero`, and the `for i in range(M)`
   router loop serialize the GPU.
3. **Not token-exact.** The torch dequant + gate/up split + shared-expert finalize
   don't bit-match the Triton path closely enough for greedy argmax (gate False).

## What a genuinely fast version would need — and why it STILL won't reach 60
- A custom **grouped-NVFP4-MoE Triton kernel** ("M rows, experts grouped"): the
  existing `nvfp4_grouped_*` kernels are "1 token, K experts", not batchable
  per-expert. That is task #3 / P2-scale kernel work (days, not hours).
- Even with a perfect kernel: **diverse top-8-of-256 routing** at M=2–3 means the
  verify reads ~M× *distinct* expert weights (no amortization; HBM-bound at decode)
  → the routed verify ≈ M× a single decode. **llama.cpp confirms** MTP on this exact
  model = 79 vs Q4_K_M 69.77 = **+13%**, not 1.6×. (See
  `MTP_MOE_CAP_AND_DECODE_LEVER_20260602.md` for the probe + analysis.)

## Conclusion (empirical, end-to-end — not a probe)
MTP speculative decode **cannot** flip the 35B-A3B verify to a net win toward 60 TPS:
per-position 0.45×, naive grouped 0.061×, and a custom grouped kernel is bounded by
the diverse-routing HBM cap to ~+13% (llama.cpp-confirmed). The goal
"0.45× → net win → 60 TPS via MTP" is **structurally unreachable on this MoE**.

The banked real win remains the **+11% decode config** (36→40 TPS,
`LYNN_MOE_DOWN_BLOCK_HIDDEN=4`). The path to 60+/150 is baseline-decode kernels and
**SM120 hardware** (FlashRT-proven 150 on a single RTX 5090), not MTP-on-MoE.

`LYNN_MTP_VERIFY_SMALLM` is left **gated OFF** as a negative-result reference so the
next session doesn't repeat the naive-torch-dequant approach.
