# Campaign — close the Spark decode 40→70 gap by cutting kernel-launch overhead (2026-06-03)

## Target (evidence-locked, not a guess)
- **40 TPS (us) vs 69.77 (llama.cpp Q4_K_M)** on the SAME Spark sm_121 — no FP4 MMA, no MTP.
- **Root cause = per-op/launch overhead, NOT bytes** (triangulated): we run at ~37% of the
  **measured 240 GB/s** BW; three traffic levers are dead (reusable graph −20%, qkv-fusion
  +0.3%, packed-4bit-attn −0.6%). The cost is **~140 tiny Triton launches/token** (per-expert
  MoE, per-layer linear-attn, per-proj attn) vs llama.cpp/ggml's fused, low-dispatch CUDA.
- BW-bound ceiling ≈ **160 TPS** → 70 is very reachable; it is a kernel/software problem.

## Two attack paths
- **A. Kernel fusion** — merge per-layer/per-expert tiny launches into fewer, fatter kernels.
- **B. Incremental-KV CUDA graph** — capture the decode graph *without* forcing full-attn
  fixed-shape (that fixed-shape recompute is why the existing reusable graph went net-negative).

## Team + roles (LEAD = Claude)
- **LEAD (me):** decompose → dispatch → **integrate + VERIFY on Spark** (compile/run/profile/
  token-exact/e2e-A-B) → decide next. The CLIs propose; I am the only one who lands+verifies on the GPU.
- **codex** (gpt-5.5): `codex exec -C <repo> -c model_reasoning_effort=high "<task>"` →
  CUDA/Triton **kernel writing + structural analysis**. Use `high` (xhigh stream-disconnects on
  long tasks across the proxy). Emits code/plan to stdout; LEAD integrates (never auto-lands on GPU).
- **Lynn/Flash:** CPU/glue/profiling-harness scripts ONLY (measured weak on CUDA kernels:
  Triton broadcast bug, `cudaAtomicAdd` confab).
- **claude CLI / Agent tool:** independent code review + adversarial verification of codex's kernels.

## Phases + gates
- **Phase 0 — diagnosis (LEAD):** clean launch profile (nsys + torch profiler) on a 35B NVFP4
  decode. Output: launches/token, per-kernel time, launch-gap (dispatch) overhead, ranked
  launch-sources. *(The earlier profile_section breakdown was cuda-sync-inflated — do NOT reuse it.)*
  GATE → a ranked, quantified launch-source list.
- **Phase 1 — first fusion (codex writes, LEAD verifies):** attack the #1 launch source.
  GATE → token-exact vs baseline + net e2e TPS gain (clean A/B on Spark).
- **Phase 2+** — next launch source; then the incremental-KV graph (path B).

## Hard rules (no shortcuts)
1. Every kernel change: **token-exact (or cos≈1 + coherent) AND clean e2e A/B** on Spark.
2. Trust **clean e2e A/B**, never profiled-section deltas (cuda-sync-inflated) or isolated microbenches.
3. Always benchmark vs **llama.cpp 69.77** e2e — the proven hardware-achievable number.
4. Spark load-race: confirm GPU-free before loading the 90 GB NVFP4 model.

## Status / log
- 2026-06-03: campaign opened. Team: codex ✅ (gpt-5.5), Lynn/Flash ✅ (glue), claude ✅;
  codebuddy ✗ (not installed).
- **codex** delivered a 6-target ranked fusion analysis (`codex_fusion_analysis.md`); its top
  pick was the MoE router (high-savings, HIGH risk). **LEAD overrode → RMSNorm first** (similar
  savings, low risk, prove-the-loop).
- **✅ PHASE 1 WIN — RMSNorm fusion = +8.7% (38.72 → 42.08 TPS), clean same-process A/B.**
  The Triton RMSNorm kernel already existed (`triton_kernels/rmsnorm.py`); wired `_rms_norm`
  (full_forward.py) to it with a cached `(1.0+weight)` offset → 1 launch vs ~6-8 eager, over
  80 layer-norms/token. Gate `LYNN_RMSNORM_FUSED=1`. Coherent; not bit-identical (Triton
  reduction order vs torch `mean`, ~1e-7, quality-neutral — same as every kernel swap here).
  **This is the FIRST lever to move e2e TPS** (the 3 traffic levers gave ~0) → confirms the
  diagnosis (launch overhead) AND the approach. Stacked best now ≈ **42 TPS** (bh4 + flags + fused norm).
- **Next targets** (codex-scoped, descending risk-adjusted ROI): #3 shared-expert finalize fusion,
  #6 full-attn cache-write + gated o_proj (low risk), #5 linear-attn micro-op fusion, then the
  HIGH-risk #2 router + #4 active-MoE-boundary fusion (strict expert-id / cos parity gates).
- Method holds: codex writes → LEAD verifies token-coherent + clean e2e A/B on Spark vs llama.cpp 69.77.
- **✅ STAGE 1 — full-attn fusion (claude-internal): TOKEN-EXACT + +0.6% (42.28→42.54).**
  claude-internal self-implemented `triton_kernels/full_attn_fused.py` (qk_norm_rope+K/V
  cache-write → 1 launch; gate-sigmoid+transpose fold → 1 launch) + wired
  `incremental_decode.py` behind `LYNN_FULL_ATTN_FUSED=1`. A/B token-exact=True (its
  bit-exact design held). Small (~60 launches, only 10 full-attn layers) but free + exact +
  stacks. **Key learning: the NORMS were ~half the launch overhead** (RMSNorm +8.7%); the
  rest is distributed across smaller clusters → next-biggest is the shared expert.
- **✅ STAGE 2 — shared-expert fusion (claude-internal): +2.8% (42.38→43.54), coherent.**
  fused gate_up→SwiGLU→down→gate→add, gated `LYNN_SHARED_EXPERT_FUSED=1`, BF16-only, fallback.
- **SCOREBOARD (4 verified launch-cuts): 36 → 43.5 TPS (+21%), 62% to llama.cpp 70.**
  bh4 +11% / RMSNorm +8.7% / full-attn +0.6% (exact) / shared-expert +2.8%. All gated/committed.
- **✅ RC QUALITY GATE — the fused stack is BEHAVIORALLY IDENTICAL to baseline.**
  `scripts/spark_rc_quality_regression.py`: baseline (3 fusions OFF) vs fused (ON), same-process,
  greedy, 40 diverse prompts (structured / V9 math / GPQA / tool-call / long-form). Result:
  **40/40 identical-greedy outputs** (struct 12/12, v9 8/8, gpqa 10/10, tool 8/8, long 2/2);
  every per-suite score identical OFF-vs-ON; TPS 39.1→42.4 (+8.6%) re-confirmed in-run.
  `RC_QUALITY_PRESERVED=True` — the non-bit-identical reduction-order deltas do NOT flip greedy
  tokens across the battery → ZERO silent capability drop. *Caveat:* absolute struct 0/12 + v9 0/8
  are no_think/template harness artifacts (EQUAL for both configs) — this run proves EQUIVALENCE,
  not absolute model quality; a thinking-on absolute eval (served endpoint) is a separate task.
  Ran in docker `lynn-eval-base:cu13` w/ `PYTHONNOUSERSITE=1` (mounted ~/.local hf-hub 1.12 shadowed transformers).
- **▶ STAGE 3 (next, per user-set priority): linear-attn micro-op fusion** (38 layers, LOWER
  risk than router). Router DEFERRED behind hard gates (expert-id/top-k/logits-cos parity +
  token-divergence first-error + MMLU/GPQA/V8/V9 smoke). Stage-3 target 43.5→**47-50 stable**;
  70 is the campaign target, NOT the Stage-3 acceptance line. Stacked best = **43.5 TPS**
  (RC-validated — promotable to default candidate).
- **STAGE 3 step 1 — DIAGNOSIS DONE (code-level, no GPU).** Traced `decode_linear_attn`
  under full base config (`INPROJ_FUSED_NATIVE_FP4` + `GQA_RECURRENT` + `FROM_OUTCONV` on).
  Per linear-attn layer the launches are: in_proj GEMM (1) + out_proj GEMM (1) — necessary
  matmuls; conv-update (1, already Triton-fused); recurrent from_outconv (1, already fused);
  RMSNormGated (1, already Triton). **The ONLY still-separate cluster = `beta=b.sigmoid()`
  + `g=neg_exp_A_log*softplus(a.float()+dt_bias.float())`** ≈ 4 tiny elementwise launches/layer
  × ~38 layers ≈ **100-150 launches/token** (dispatch ≈ 1ms/tok ≈ ~4.6% of the 23ms/tok budget).
  - **STAGE 3 SPEC (the kernel to write — implement→verify):** the kernel
    `triton_kernels/gated_delta.py::_recurrent_fused_prepare_kernel` (+ gqa variant) ALREADY
    `tl.exp(g)` and loads `beta` raw → extend its signature to take **raw `a_ptr, b_ptr`
    + `dt_bias`, `neg_exp_A_log`** and compute, per-head inside the kernel:
    `beta = tl.sigmoid(b)`, `g = neg_exp_A_log * softplus(a + dt_bias)` (softplus =
    `log1p(exp(x))`), then the existing `exp(g)`. Eliminates the outside elementwise launches.
    Gate `LYNN_LINEAR_ATTN_FUSE_GBETA=1`, fallback to the current pre-computed g/beta path.
    Math is identical → expect token-exact (cos≈1). Free micro-win: cache `dt_bias.float()`.
    **Expected +3-5%** (smaller than RMSNorm's +8.7% — fewer/tinier ops, as the front-loaded
    pattern predicts) → toward the 47-50 target. Verify: token-coherent + e2e A/B + re-run RC.
    Tooling: claude-internal writes the kernel → LEAD verifies on Spark (codex still gated).
- **✅ STAGE 3 — g/beta-fold (claude-internal): +2.6% (42.93→44.03), RC-VALIDATED.**
  New `_recurrent_fused_prepare_from_outconv_gqa_gbeta_kernel` computes beta/g per-head in
  the recurrent kernel from raw a/b/dt_bias/neg_exp_A_log. Gate `LYNN_LINEAR_ATTN_FUSE_GBETA=1`,
  default OFF, fallback byte-identical. A/B coherent; TOKEN_EXACT=False (the predicted
  bf16-vs-fp32 sigmoid / softplus=log1p(exp) last-bit — numeric, not logic). **RC re-run with
  g/beta in the toggle set: 40/40 identical-greedy (struct 12/12 · v9 8/8 · gpqa 10/10 · tool
  8/8 · long 2/2), all per-suite scores identical baseline-vs-fused, RC_QUALITY_PRESERVED=True,
  in-run TPS 39.8→44.1.** → late-token divergence is reproducibility nuance, not capability,
  same verdict as RMSNorm/shared-expert. Commit 65418ec (draft) + this result.
- **SCOREBOARD (5 RC-validated launch-cuts): 36 → 44 TPS (+22%), ~63% to llama.cpp 70.**
  bh4 +11% / RMSNorm +8.7% / full-attn +0.6% (exact) / shared-expert +2.8% / g/beta +2.6%.
  The full fused stack is RC-validated (behaviorally identical to baseline) → promotable default.
- **▶ STAGE 4 (next):** remaining linear-attn elementwise (z-reshape / conv-pre / out_proj
  adjacency) is thin per-layer; re-profile to find the next real cluster before writing. The
  HIGH-risk router stays DEFERRED behind hard gates (expert-id/top-k/logits-cos parity +
  token-divergence first-error + MMLU/GPQA/V8/V9). Target 47-50 stable; 70 = campaign target.
- **✅ STAGE 4 step 1 — DECODE LAUNCH CENSUS (empirical, full 5-fusion stack).**
  `scripts/spark_decode_launch_profile.py` (delta-of-16-vs-32-tokens, cancels prefill/warmup):
  **≈ 1527 CUDA launches / decode token** (the earlier "~140" guess was an order-of-magnitude
  low). With BW at 37% of 240 GB/s, ~half the token time is launch/dispatch, not bytes.
  Top clusters /tok: **aten::copy_ 230** · mm 110 + gemvx 68+57 (≈235 projection GEMMs, M=1) ·
  _rmsnorm 92 (= norm sites, already 1-launch-each) · elementwise 90 + add 80 + direct_copy 79 ·
  MoE grouped kernels 45-each (gate_up/down/topk/softmax/shared) · quantize_fp4 35 · cutlass
  NVFP4 gemm 34 · conv 34 · **g/beta kernel 34 (confirmed live)**.
  **Strategic conclusions:** (1) MoE/router is ALREADY grouped (45/layer, NOT per-expert×8) →
  the high-risk router is MOOT for launch count; don't touch it. (2) norms are already 1-launch
  each → can't cut count without norm+matmul fusion or a graph. (3) point-fusion has hit
  diminishing returns — the remaining mass is DISTRIBUTED copies/elementwise/matmuls, not a
  single fusable cluster. → **Stage-4 real choices: (A) copy-hunt the 230/tok aten::copy_
  (tractable, incremental) or (B) CUDA-graph the decode (structural; collapses 1527 launches;
  the M3 reusable-graph historically gave +10% → ~48, hits 47-50; HARD — full-attn variable
  KV-shape is the known blocker that sank prior attempts). Awaiting user direction.**
