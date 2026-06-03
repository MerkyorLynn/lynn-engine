# Stage 6 Phase-0 — DECODE byte-path trace + FP4-kernel efficiency (the real lever)

**LEAD = Claude (main session). You are an assistant CLI. OUTPUT TO STDOUT ONLY — do NOT modify
any repo file, do NOT git-commit. LEAD integrates.**

## Context
- HW: DGX Spark GB10 **sm_121, NO FP4 MMA**, 240 GB/s, CUDA 13 / torch 2.9.1.
- Model: `Qwen3.6-35B-A3B` Lynn-native **W4A16 NVFP4**. Repo: this dir (`lynn-engine`).
- Goal: close decode **~45 → ~70 TPS** (llama.cpp Q4_K_M = 69.77 on the same card). Quality must stay
  token-exact / cos≈1.
- RC decode config (env, all ON): `LYNN_MOE_IMPL=packed_nvfp4`, `LYNN_NATIVE_ACTIVE_MOE_BACKEND=triton`,
  `LYNN_NATIVE_GATEUP_BACKEND=triton_fast_decode`, `LYNN_NATIVE_DOWN_BACKEND=triton`, `LYNN_MOE_FAST_FIXED=1`,
  `LYNN_PACKED_DECODE_BACKEND=native_fast_2d`, `LYNN_NVFP4_BF16_OUT=1`, `LYNN_NATIVE_FP4_LM_HEAD=1`
  + RMSNORM/FULL_ATTN/SHARED_EXPERT/GBETA fused flags. Baseline ≈ **44.7 TPS, ~22 ms/token**.

## What we already know (evidence-lock, 2026-06-03)
- Resident weights = **87.2 GiB** = BF16 **64.7** + packed-NVFP4(uint8) **15.0** + FP32 **7.5**.
  `release_decode_bf16_shadows()` drops **60 GiB** (87→27). The 60 GiB BF16 is a dequant duplicate of the 15 GiB packed.
- MoE decode entry `engine/moe_packed_nvfp4.py::moe_forward_decode_packed_nvfp4`: the gate_up
  (`triton_fast_decode` → `nvfp4_grouped_gate_up_silu_fast_decode`) and down (`triton` →
  `nvfp4_grouped_down_weighted_sum`) read **`mlp.experts._gate_up_packed` / `_down_packed` (PACKED FP4)** + scales.
  Attn/dense projections use `engine/nvfp4_runtime.py::forward_native_fast_2d` → `torch._scaled_mm` on packed FP4.
- So DECODE *appears* to already read FP4. BUT a post-`release` `generate()` errored
  `KeyError('mlp.experts.1.gate_proj.weight')` — and that call does a fresh **PREFILL** first, so the error is
  very likely PREFILL (per-expert BF16), NOT decode. This must be disambiguated.

## RESOLVE (the deliverable — be precise, cite file:line)
1. **DECODE per-token weight bytes.** For the RC decode path (single-token, `_decode_layer` / `_decode_layer_fast`
   → packed paths), enumerate EVERY weight tensor read per token and tag FP4-packed vs BF16-resident. Give a
   bytes/token estimate split FP4 vs BF16 (note MoE is top-8/256 sparse; attn/linear-attn/lm_head are dense).
   Is ANY BF16 weight on the steady-state decode read path? (If yes → that's the Stage-6 target. If no → decode
   is already FP4 and the wall is kernel-efficiency/dispatch, not shadow.)
2. **FP4 kernel efficiency.** Find `nvfp4_grouped_gate_up_silu_fast_decode` and `nvfp4_grouped_down_weighted_sum`
   (grep; likely `engine/moe_packed_nvfp4.py` or `engine/triton_kernels/` or `triton_kernels/`). Do they read the
   4-bit packed once + **dequant in register/smem** + bf16 accumulate, with **NO full BF16 weight materialized in
   HBM**? Or do they write/read a BF16 temp (wasted bandwidth)? Same for `forward_native_fast_2d` / `_scaled_mm`
   (does the sm_121 CUTLASS NVFP4 path materialize a wide temp?).
3. **PREFILL BF16.** Where does prefill read per-expert BF16 `mlp.experts.N.gate_proj.weight`? Is there a packed
   prefill MoE? (Confirms the 60 GiB shadow is prefill-only → droppable for decode-only serving = a memory win.)
4. **REAL Stage-6 lever (ranked).** Given 1–3, what is the highest-ROI path to 45→70?
   (a) decode already FP4 → attack dispatch (reusable decode CUDA graph / fewer launches);
   (b) FP4 kernels waste HBM on a BF16 temp → write a true read-4bit/register-dequant/bf16-GEMV kernel;
   (c) decode still reads some BF16 → eliminate it.
   Pick with evidence; if (b/c), provide the kernel.

## Constraints
- sm_121 has no FP4 MMA → any kernel must dequant 4-bit → bf16 in registers, then bf16 GEMV (M=1 decode).
- Must be numerically token-exact / cos≈1 vs the current RC decode. Gate any new path behind a NEW env flag
  (e.g. `LYNN_MOE_EXPERT_FP4_GEMV=1`), default OFF, byte-identical fallback.
- Verification is LEAD's job on Spark (you do NOT have the GPU). Make claims checkable (file:line, formulas).

## Output
Markdown to STDOUT: the trace answering 1–4, with file:line citations, a bytes/token table, and (if 4b/4c)
the kernel + integration point. NO file edits.
