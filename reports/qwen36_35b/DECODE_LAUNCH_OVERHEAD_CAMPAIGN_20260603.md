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
- **✅ STAGE 4A — copy-hunt (claude-internal): bf16-out +3.3% + o_proj-nocopy, RC-VALIDATED.**
  Root cause of the #1 copy cluster: `native_fast_2d` did `_scaled_mm`→fp16 then `.float()`→fp32,
  then `_linear .to(bf16)` = **2 copies/projection**. `LYNN_NVFP4_BF16_OUT=1` → `_scaled_mm
  out_dtype=bfloat16` directly (**WORKS on sm_121, no fallback**) → both copies gone. Plus
  `LYNN_DECODE_OPROJ_NOCOPY` (drop redundant T=1 o_proj `.contiguous()`). A/B 43.94→45.37 (+3.3%);
  RC with both flags in the toggle = **40/40 identical-greedy, scores identical,
  RC_QUALITY_PRESERVED=True, in-run 39.6→44.2**.
- **SCOREBOARD — Stage 4A complete: 36 → ~45 TPS (+26%), 7 flags, all RC-validated.**
  bh4 / RMSNorm / full-attn / shared-expert / g/beta / bf16-out / o_proj-nocopy.
- **DECISION (user): finish A, SKIP B (CUDA-graph).** Why: NVFP4 is structurally capped on
  sm_121 — no FP4 MMA, and the BF16 dequant-shadow makes decode read ~2× the bytes of
  llama.cpp's hand-fused 4-bit-read+dequant-in-register Q4_K_M kernel (memory wall ≈40; that's
  why baseline sat at 38.96). 70 is Q4_K_M territory, unreachable for NVFP4 here. B's +10% isn't
  worth multi-day for ~48. → **NEXT = evaluate MTP, not the graph.**
- **▶ STAGE 5 (mainline next): MTP eval (task #11) — wiring ALREADY EXISTS, it's RUN-not-build.**
  Assets confirmed: trained sidecars on Spark `models/mtp_sidecars/qwen36-35b-a3b-mtp/mtp.safetensors`
  (+ a newer `qwen36-35b-a3b-mtp-official-lynn-fused/` variant). Engine MTP infra present in the
  NVFP4 resident runner: `engine/mtp_sidecar.py` (loader: load_mtp_sidecar / mtp_logits /
  mtp_hidden_and_logits), `engine/mtp_serving.py` (`speculative_step_k1` / `speculative_step_kn_batched`),
  resident_runner flags `LYNN_MTP_SIDECAR` / `LYNN_MTP_VERIFY` / `LYNN_MTP_SHADOW_VERIFY`.
  **EVAL RECIPE (fresh context):** on the ~45-TPS RC-validated NVFP4 stack, set
  `LYNN_MTP_SIDECAR=<path>` + K≥2 step mode + `LYNN_MTP_VERIFY=1`, measure accept-rate + TPS gain +
  RC quality; pick sidecar variant (base vs official-lynn-fused). Expect **~+13% byte-capped**
  (llama.cpp APEX proves it on this exact model: 79 vs 69.77) → ~45→~51, clears 47-50, NO kernel risk.
- **▶ STAGE 5 RESULT (6/3) — NOT a win this round, but NOT判负 (fixable alignment).** Ran
  `scripts/spark_mtp_ab.py` on the full ~45 RC stack, both sidecar variants (`qwen36-35b-a3b-mtp`
  + `-official-lynn-fused`): **`TOKEN_EXACT=True` (verify/reject/rollback wiring correct), but
  accept = 2/82 ≈ 2.4% → effective ~20 TPS (regression).** Diagnosis: NOT an engine-correctness
  bug — a draft-head↔serving ALIGNMENT bug. hidden source confirmed pre-final-norm (correct);
  both sidecars same 2.4% (systematic, not variant) → suspect = **offset mismatch** (trained
  contract offset=2; serving may apply offset-1 → drafts ~always rejected, 2.4% = "basically
  random"). NOT判负 because **llama.cpp APEX-MTP serves the SAME head+model at +13% / 60%+ accept**
  → head is good, +13% is reachable; we only need serving alignment.
  - **FIX ENTRY (next, focused/fresh ctx):** `engine/mtp_sidecar.py::mtp_logits` offset contract
    vs the serving loop's draft placement. First add a draft-vs-actual ±1-offset probe to confirm
    the shift direction, then correct it (likely config-level, no retrain). Cross-check positioning
    against the working llama.cpp APEX-MTP serving. If fixed → ~45→~51.
- **✅ STAGE 5 CORRECTION (6/3, SUPERSEDES the above — accept was never the bug; OFFSET HYPOTHESIS FALSE).**
  Three probes on the same ~45 RC stack (engine md5 == HEAD):
  - `spark_mtp_offset_align_probe.py`: serving's draft contract (`_mtp_draft_logits`: base_hidden=h_p,
    embed(x_{p+1}), `cat([embed,hidden])`) predicts **x_{p+2} at 91.5–91.7% top-1** (rank median 0).
    Draft head is excellent; the "serving offset-1 vs trained offset=2" suspicion was WRONG — the in-use
    sidecar matches serving's contract. (The A100 `a100_mtp_iterative_train.py` `[hidden,embed]`/offset-1
    contract is a different, unused training line — a red herring.)
  - `spark_mtp_block_verify_probe.py`: the **T=2 block verifier is 100% correct** (pos-0 b0==true 36/36,
    hidden cos=1.0000, lm_head all-positions fine).
  - `spark_mtp_verify_config_sweep.py` (7 configs, one load): **accept is high everywhere, NOT 2.4%** —
    seq_k1 / k1b_default / k1b_lin_t1loop = **88.2%**, k2_default (= the old A/B config) = **76.0%**,
    k1b_fast (true-batched) = **97.0%**; all TOKEN_EXACT except k1b_fast. The earlier `2/82≈2.4%` did
    not reproduce → stale/narrow measurement, not an offset root cause.
  - **THE REAL BLOCKER IS SPEED, not accept.** The eager speculative loop costs **~680 ms / event
    (2 tokens)** — BF16 MTP-draft MoE + non-cheap batched verify + snapshot/restore + dispatch/sync —
    so even at ~90% accept it is a **net LOSS** vs the 44.39 baseline. (`decode_tps` reads 0 for the spec
    path = a metric gap; per-event wall-clock is the truth, ≈ 3 TPS effective.) `k1b_fast`
    (`LYNN_FULL_ATTN_K2_BACKEND=k2` + `LYNN_MTP_VERIFY_SMALLM=1`) targets a true-batched verify but is
    **NOT token-exact** (kernel drift — the M12 `decode_full_attn_k2` / smallm-MoE T≥2 numerics).
  - **REVISED VERDICT:** MTP correctness ✅ + accept ✅ (head is good), but MTP is **not a ready 45→51
    shortcut** in this eager runtime. Realizing it needs (a) a low-overhead speculative runtime (graph the
    draft+verify, drop per-step state clones) or (b) token-exact true-batched verify kernels — **both
    overlap Stage 6.** Article framing stays ~45 TPS; MTP = "accept proven, runtime too heavy, speed
    unrealized". → mainline proceeds to STAGE 6.
- **▶ STAGE 6 (endgame, only if MTP tops out — user-directed moat):** the fused
  **read-4bit + dequant-in-register + bf16-GEMV + zero-shadow + single-launch** kernel. Rationale
  (recomputed): the REAL wall is bandwidth, not launch — baseline 38.96 ≈ BF16-shadow 6GB/tok ÷
  240GB/s; reading true 4-bit (1.7GB/tok) moves the wall ~40→~140, where 70 lives (2-3×, not the
  graph's +10%). Same NVFP4 weights + kernel port to R6000's FP4 MMA = native, faster → Lynn's
  cross-device core (the L1 moat). llama.cpp Q4_K_M is the **MIT-licensed reference blueprint**
  (study the pattern, clean-room for NVFP4 — not AGPL, license-clean). Multi-day real kernel work;
  staged PoC→dense→MoE→fuse, each gated + RC. "啃下来 = Lynn 成 NVFP4 的 llama.cpp."
- **✅ STAGE 6 step 1 — EVIDENCE-LOCK DONE (`spark_stage6_shadow_byte_audit.py`, user-directed before any kernel).**
  Premise CONFIRMED + target SHARPENED. Resident weights on the full RC stack = **87.22 GiB** =
  **BF16 64.72 + packed-NVFP4(uint8) 15.00 + FP32 7.50**. `release_decode_bf16_shadows()` drops **60 GiB**
  (87→27 GiB; the BF16 shadow is a pure dequant duplicate of the 15 GiB packed).
  - Baseline decode **44.71 TPS → implied ~5.37 GB/token** (≈ the 6 GB BF16-shadow premise — confirmed; NOT pure-FP4 1.7 GB).
  - **Decode DEPENDS on the BF16 shadow:** post-release decode errors `KeyError('mlp.experts.1.gate_proj.weight')`
    → the **MoE active-expert decode reads BF16 expert weights**, not packed FP4. (Attn/dense projections already go
    packed-FP4 via `torch._scaled_mm`/CUTLASS; the MoE experts are the BF16 byte hog = most of the 60 GiB shadow + the ~5 GB/token read.)
  - **LOCKED TARGET:** Stage 6's highest-value first lever = a **read-4bit + register-dequant + bf16-GEMV MoE
    active-expert decode kernel** (gate_up + down) reading the packed NVFP4 experts directly, no BF16 shadow. Cuts the
    dominant weight read ~4× (resident 64.7→15 GiB; per-token MoE read BF16→FP4) → wall ~40→~140, frees ~60 GiB. Attn/dense already packed (secondary).
  - **PHASE 0 PoC (next; claude-internal writes / LEAD verifies on Spark; codex gated):** ONE active expert's gate_up
    (or down) as a Triton read-4bit/dequant-in-register/bf16-GEMV; gate `LYNN_MOE_EXPERT_FP4_GEMV=1`. Gates: (a) cos≈1 /
    token-coherent vs BF16 expert path, (b) isolated microbench faster, (c) runs with the BF16 expert shadow DROPPED
    (proves shadow-free). Then widen all experts → delete shadow → RC battery → e2e TPS vs 44.71 & vs llama.cpp 69.77.
- **✅ STAGE 6 step 2 — DEEP TRACE + 2 PROBES (SUPERSEDE step-1; the step-1 "MoE FP4 GEMV" target was WRONG).**
  Two headless CLIs (codex gpt-5.5 + claude-internal, traces in `reports/stage6/{codex,claude_internal}_decode_trace.md`)
  + two Spark probes, all converging:
  - **The MoE expert decode is ALREADY read-4bit / register-dequant / bf16-GEMV** (`triton_kernels/nvfp4_moe.py:251,352`:
    packed-uint8 load → e2m1 nibble decode in registers → per-16 scale → fp32 accumulate → store bf16 activation only,
    NO BF16 weight temp in HBM). Writing `LYNN_MOE_EXPERT_FP4_GEMV` would re-implement it → ~0 gain. **DO NOT write it.**
  - **`spark_stage6_decode_shadow_free_probe.py`:** prefill once (shadows present) → `release_decode_bf16_shadows()` (−60 GiB)
    → **continue DECODE shadow-free, same state: 42.36 → 43.68 tok/s (1.03×), coherent.** Decode does NOT read the 60 GiB
    BF16 — it's the **PREFILL** routed-expert shadow (`engine/full_forward.py:323-339`); the step-1 `KeyError(mlp.experts.1.gate_proj.weight)`
    was a fresh prefill, not decode. → "delete shadow" = **decode-only MEMORY win (87→27 GiB)**, NOT a TPS lever.
  - **`spark_stage6_packed_decode_sweep.py` (the cheap A/B both CLIs predicted):** routing the genuinely-BF16 decode
    weights (full-attn q/k/v/o, linear-attn out_proj) to packed FP4 via existing flags gives **NO win** —
    `full_attn_fp4` 0.999× / `linear_outproj_fp4` **0.775×** / `all_packed` 0.772× (all coherent). Cutting BF16→FP4 reads
    is neutral-to-NEGATIVE → **decode is latency/launch-bound, NOT bandwidth-bound** (the RC's `PACKED_DECODE=0` was correct).
    Reconciles the implied 5.37 GB/tok vs actual ~2.6 GB weights (half FP4): the gap is launch/latency, not bytes.
  - **REVISED STAGE 6 VERDICT:** the "read-4bit / zero-shadow → 40→140" premise is **empirically FALSE for this decode**.
    read-4bit is already done (MoE); the shadow is prefill-only (memory win); FP4-ing attn is neutral/negative.
    **The only remaining decode lever is dispatch/launch reduction (reusable decode CUDA graph + launch-folding)** — but the
    FP8-revival experience put graph at only **+10%** (→ ~48-50, not 70) and it's the hard full-attn-variable-KV path.
    **Honest ceiling:** matching llama.cpp 69.77 needs ggml-level fused low-dispatch CUDA; on Spark (no FP4 MMA) the engine
    is structurally capped ~45-50. The kernel moat pays off on **R6000 (FP4 MMA)**, not Spark.
  - **NEXT (cheap, decides the graph bet):** measure `LYNN_REUSABLE_DECODE_GRAPH=1` e2e TPS vs 44.68 on the NVFP4 stack
    (infra exists; the +10% was FP8, NVFP4 delta is UNMEASURED). >+15% → graph is worth it; ≈+10% → accept ~45-50 + the
    60 GiB decode-only memory win, and move kernel work to R6000. **No new kernel until this number exists.**
- **✅ STAGE 6 step 3 — FINAL: the graph bet is NET-NEGATIVE; Spark NVFP4 decode is at its STRUCTURAL CEILING.**
  `spark_stage6_reusable_graph_ab.py`: baseline **44.60** → `LYNN_REUSABLE_DECODE_GRAPH=1` **33.46 TPS = 0.750×**, coherent.
  The reusable graph is a **−25% REGRESSION** (not the FP8 +10%): `LYNN_FULL_ATTN_FIXED_SHAPE` reads the full KV cache via
  masked SDPA every token + graph-safe MoE dispatch overhead > the dispatch it saves. So **both levers are dead on Spark:**
  bandwidth (read-4bit already done; FP4-ing attn 0.999×/0.775×) AND dispatch (graph 0.75×).
  - **CAMPAIGN CONCLUSION:** Spark sm_121 NVFP4 35B-A3B decode is **structurally capped ≈ 44-45 TPS**. The 38.96→45 (+26%)
    point-fusion gains were real and are banked (RC-validated); beyond that, there is NO software lever on this HW —
    matching llama.cpp's 69.77 needs ggml-level hand-fused low-dispatch CUDA (a ground-up kernel rewrite) and ultimately
    FP4-MMA silicon (R6000, retired). **The engine decode-speed track on Spark is concluded at ~45.**
  - **BANKABLE Stage-6 deliverable = the 60 GiB decode-only MEMORY win** (release_decode_bf16_shadows after prefill →
    resident 87→27 GiB), which buys KV/long-context/batch headroom on the shared 128 GB Spark — Spark's real value
    (long-ctx 6.77× + multi-service), not raw single-stream decode TPS. (Productizing it needs a packed prefill MoE or
    per-request shadow reload — flagged as a follow-up; not done this round.)
  - Article/README "对标 llama.cpp" framing updated to reflect the ceiling (honest: Spark ~45 is the NVFP4 ceiling; parity
    is an FP4-MMA / ggml-rewrite goal, not a Spark deliverable).
- **✅ STAGE 6 step 4 — BANKED the 60 GiB decode-only MEMORY win as a SERVING capability (option (b), user-chosen).**
  Productized the probe finding (step2): decode never reads the BF16 shadow, so serve at ~27 GiB during decode and rebuild
  the shadow only when a new prefill needs it.
  - **New primitive `LynnIncrementalRunner.reload_decode_bf16_shadows()`** (`engine/resident_runner.py`) — exact inverse of
    `release_decode_bf16_shadows()`. `release` now records what it dropped; `reload` rebuilds those BF16 weights by
    dequantizing the still-resident packed NVFP4 via the decode-proven `moe_packed_nvfp4._dequant_nvfp4_slot` (the same
    primitive the graph-safe v31 decode path uses). **NO disk I/O** — the 15 GiB packed stays resident. (Note: on Spark's
    unified memory you cannot "offload to host" to free the pool; you must drop + rebuild, which is why reload is a dequant.)
  - **Server (`server/openai_http.py`)** rewired from the old one-shot (2nd request 409'd) to the per-request cycle
    **reload → prefill → release → decode**; `/health` + response now expose reload seconds / release GiB / reload count.
  - **Verified on Spark** (`scripts/spark_stage6_shadow_reload_serving_ab.py`, docker `lynn-eval-base:cu13`, APEX stopped):
    resident **88.16 → 28.18 GiB** on release (drops **60.00 GiB**) → **88.18** on reload. **TOKEN-EXACT** on all three
    gates: shadow-free decode == baseline, **reloaded-shadow PREFILL == baseline** (reload is bit-faithful), and the server
    `LynnEngineHandle` path == baseline across 2 sequential requests (no one-shot 409, reload_count=2). **No TPS regression:**
    decode 44.18 (baseline) / 44.50 (req1) / 44.23 (req2) tok/s. **KV headroom is real & reclaimable:** at 28 GiB resident a
    **35 GiB** KV cache (max_seq_len ≈ 1.83M) allocates fine, but at 88 GiB resident only **1.3 GiB** is free → the same
    alloc OOMs. `ALL_PASS=True`.
  - **COST (the price of option (b)):** per-request reload ≈ **24 s** (60 GiB FP4→BF16 dequant, unoptimized elementwise).
    So this mode fits **decode-only / single long-prompt** serving where freeing 60 GiB during a long generation (KV /
    co-resident headroom) is worth a one-time ~24 s, **NOT** high-throughput multi-request serving (every post-release
    request pays the reload). For zero-reload serving (resident 27 GiB always, no prefill peak), the path remains
    **option (a): a packed-NVFP4 prefill MoE** (decode kernels are M=1-only; prefill M>1 needs a new batched kernel) — still
    the documented follow-up.
  - **CAVEAT (honest):** option (b) frees 60 GiB **only during decode**; it does **not** lower the prefill peak (shadows are
    reloaded before prefill, and KV is pre-allocated at max_seq_len), so it does **not** by itself let a longer-context
    *prefill* fit. That requires option (a) or a growable/lazy KV cache.
- **✅ STAGE 6 step 5 — P0.1 packed-prefill / zero-reload smoke PASSED (correctness + memory proof; not performance path).**
  `scripts/spark_stage6_packed_prefill_no_reload_smoke.py` now gates the next path after option (b): release the 60 GiB
  grouped-MoE BF16 shadow, do **not** call `reload_decode_bf16_shadows()`, prefill from resident packed NVFP4, and assert
  token-exact output + low peak memory.
  - **Attempt #1 (`LYNN_PACKED_PREFILL_SLOW_MODE=decode_kernel`) failed honestly:** memory was clean (resident 28.18 GiB,
    peak 28.30 GiB, reload_calls=0) but token state diverged: baseline ids `[20, 198, 1409, 27102]` vs probe
    `[20, 271, 248068, 198]`. Replaying the T=1 decode MoE kernel is fast, but not BF16-prefill state-coherent across
    decode steps. It remains a diagnostic only.
  - **Attempt #2 (`LYNN_PACKED_PREFILL_SLOW_MODE=stream_bf16`) passed:** current-layer packed NVFP4 is temporarily
    dequantized to BF16, the original BF16 MoE math is run, and the temp dies before the next layer. Spark result:
    baseline resident **88.18 GiB**, release **60.00 GiB / 80 tensors**, resident after release **28.18 GiB**, probe peak
    **40.28 GiB** (no hidden 88 GiB reload), reload_calls **0**, token-exact ids `[20, 198, 1409, 27102]`, decode
    **42.47 TPS** vs baseline **42.97 TPS**, `ALL_PASS=True`.
  - **Cost / meaning:** proof prefill is **20.75 s** for the tiny `2+2=` smoke. This is not shippable throughput; it proves
    the zero-resident-shadow invariant and gives P2 a correctness oracle. This promoted P0.2 resident inventory, now
    recorded below, then P1 batched packed projections and P2 grouped M>1 packed MoE prefill kernels.
- **✅ STAGE 6 step 6 — P0.2 resident BF16 inventory PASSED (kernel-order gate; no speed claim).**
  `scripts/spark_stage6_p02_resident_inventory.py` ran on Spark after stopping idle APEX, docker status **0**, then APEX
  was restored on `:18098` and `/health` returned `{"status":"ok"}`.
  - **Load / release:** resident after load **88.16 GiB**; BF16 total before release **64.72 GiB**; release dropped
    **60.00 GiB / 80 tensors**; resident after release **28.16 GiB**.
  - **After-release BF16 inventory:** total **4.72 GiB**: `linear_attn.projection` **1.884 GiB**,
    `outside.embed` **0.947 GiB**, `outside.lm_head` **0.947 GiB**, `full_attn.projection` **0.508 GiB**,
    `moe.shared_expert` **0.391 GiB**, `moe.router` **0.039 GiB**, norms negligible.
  - **Packed alias finding:** normal inventory found **0.0 GiB** packed-alias candidates, so the next memory reductions
    require explicit packed-prefill / packed-lookup paths rather than simply releasing an existing alias.
  - **Decision:** P1 should lead with batched packed projection prefill plus embed/lm_head semantics; P2 remains the
    grouped M>1 packed MoE prefill kernel that replaces the **20.75 s** `stream_bf16` proof and avoids the **23-24 s**
    full reload. Router is not first-order leverage.
- **✅ STAGE 6 step 7 — P1 single dense projection packed-NVFP4 PoC PASSED.**
  `scripts/spark_stage6_p1_dense_projection_poc.py` ran a real
  `model.language_model.layers.0.linear_attn.in_proj_qkv.weight` projection on Spark without constructing the full
  runner or stopping APEX. APEX stayed online and `/health` remained `{"status":"ok"}`.
  - **Artifact correction:** planning assumed E4M3 scale; the real Lynn-native 35B artifact stores this projection's
    scale as **FP16**. Shape: packed `uint8[8192,1024]`, scale `float16[8192,128]`, global scale `float32[]`.
  - **Bytes:** BF16 shadow **32.00 MiB** vs packed+scale+global **10.00 MiB** (**3.20x** smaller); timed packed args
    **10.04 MiB**.
  - **Numeric:** packed Triton vs FP32 dequant oracle `cos=1.000000000`, `rel_l2=1.387e-07`, `max_abs=7.153e-07`,
    argmax match. Vs BF16 shadow `cos=0.999998123`, `rel_l2=1.938e-03`, argmax match.
  - **No hidden BF16 shadow:** timed packed benchmark ran after deleting FP32/BF16 reference weights; peak allocation
    **0.0177 GiB**, below the single projection's **0.03125 GiB** BF16 shadow size.
  - **Microbench:** packed Triton **160.29 us** vs BF16 `F.linear` **190.03 us**, **1.186x** speedup. This banks the
    single-projection contract, not full dense-path integration. Next gate: P1-A batched/M>1 packed projections.
- **⚠️ STAGE 6 step 8 — P1-A naive batched projection bridge REJECTED for performance (numeric/no-shadow PASS).**
  `scripts/spark_stage6_p1a_batched_projection_poc.py` added a single-launch bridge over
  `tokens x output-row-blocks` for the same real `linear_attn.in_proj_qkv` projection. It intentionally did not reuse
  activation tiles across tokens.
  - **Numeric:** M=1/4/16/64 all passed vs FP32 dequant oracle (`cos=1.000000000`, rel_l2 <= `3.161e-07`, argmax
    match).
  - **No hidden BF16 shadow:** timed packed benchmark ran after deleting FP32/BF16 refs; peak **0.0200 GiB**, below the
    projection BF16 shadow (**0.03125 GiB**).
  - **Perf:** M=1 packed **159.27 us** vs BF16 **181.80 us** (**1.141x**), but M>1 fails badly: M=4 **0.260x**,
    M=16 **0.067x**, M=64 **0.016x** vs BF16. BF16 `F.linear` amortizes the batch; this bridge does not.
  - **Decision:** keep it as a correctness/regression probe only. Do not wire into resident_runner. Next P1-A attempt must
    be a true tiled `BLOCK_T x BLOCK_OUT x BLOCK_K` packed projection kernel.
- **⚠️ STAGE 6 step 9 — P1-A tiled scalar projection bridge also REJECTED for Spark performance.**
  `nvfp4_tiled_batched_matmul_packed()` added a real `BLOCK_T x BLOCK_OUT x BLOCK_K` style bridge and swept
  `BLOCK_T={8,16,32}`, `BLOCK_OUT={16,32,64}`, `BLOCK_K=128` for M=16/64.
  - **Numeric/no-shadow:** all nine configs passed vs FP32 dequant oracle; representative best-shape
    `cos=0.999999979`, `rel_l2≈2.6e-04`, argmax match, timed packed peak **0.0200 GiB** after deleting BF16/FP32 refs.
  - **Perf vs naive:** tiled is real progress over the naive bridge, up to **25.93x** faster.
  - **Perf vs BF16:** still fails the promotion gate. Best M=16 is **209.11 us** vs BF16 **156.60 us** (**0.749x**);
    best M=64 is **421.43 us** vs BF16 **151.17 us** (**0.359x**).
  - **Decision:** scalar-dequant dense M>1 packed prefill is closed on Spark. Do not wire into resident_runner. Dense M>1
    needs native FP4-MMA/CUTLASS-style kernels if pursued; Stage 6 should move to P2 grouped MoE prefill or native runtime
    work.
- **✅ STAGE 6 step 10 — P2 grouped MoE prefill census PASSED; first real kernel target locked.**
  `scripts/spark_stage6_p2_grouped_moe_prefill_census.py` ran one real layer's routed MoE with BF16 resident shadow,
  then deleted `mlp.experts.gate_up_proj` / `down_proj` and compared the two packed proof paths.
  - **Bytes:** one-layer grouped expert BF16 shadow **1.500 GiB** vs packed expert tensors **0.563 GiB** (**2.667x**);
    after deleting the BF16 shadow, memory was **0.641 GiB** for the single-layer harness.
  - **Numeric:** `stream_bf16` is exact vs BF16 (`rel_l2=0`, argmax match); `smallm` verifier is tight but not exact
    (`cos≈0.9999975`, `rel_l2≈0.002`, argmax match).
  - **Latency:** BF16 prefill **4.16/6.17/12.30/21.45 ms** for M=1/4/16/64. `stream_bf16` is **487.86-506.22 ms**
    per layer, explaining the **~20.75 s** 40-layer no-reload proof. `smallm` is **9.96/39.49/128.43/260.65 ms**:
    **1.94-48.97x** faster than stream, but still **0.082-0.418x** of BF16.
  - **Memory:** `stream_bf16` peaks **12.64 GiB** in the one-layer harness because it materializes wide dequant
    temporaries; `smallm` peaks **0.70 GiB** after BF16 shadow deletion.
  - **Decision:** P2 should target the routed expert inner loop with prefill router semantics preserved:
    `h_flat[M,2048] + expert_ids[M,8] + routing_weights[M,8] + packed gate/up/down -> moe_out[M,2048]`. Do not
    promote `stream_bf16` or `smallm`; use them as numeric/memory oracles.
- **⚠️ STAGE 6 step 11 — P2-A single-expert packed gate/up component is valid but NOT a performance win.**
  `nvfp4_prefill_gate_up_silu_one_expert()` added the smallest routed-MoE prefill slice: one expert, M>1 hidden rows,
  packed NVFP4 gate/up weights, fused `silu(gate) * up`, BF16 intermediate output.
  - **Bytes:** one expert's BF16 gate/up shadow **4.00 MiB** vs packed **1.50 MiB** (**2.667x** smaller).
  - **Numeric:** main run passes component numeric gate (`cos≈0.99999`, rel_l2 `0.0038-0.0045`, argmax match for
    M=1/4/16/64). A sweep with another expert had M=64 argmax mismatches, so this is not RC-quality.
  - **No hidden BF16 gate/up:** packed bench ran after deleting `mlp.experts.gate_up_proj`; peak stayed **0.953 GiB**.
  - **Perf:** main run packed **82.77/82.72/82.67/240.33 us** vs BF16 **16.29/18.81/19.22/19.10 us** for
    M=1/4/16/64 (**0.197/0.227/0.233/0.079x**). Small tile sweep best M=64 is **0.115x**; larger tiles OOR on Spark.
  - **Decision:** keep as a component probe only. Do not wire into `resident_runner`. Next P2-B should measure routed
    gate/up grouping over unique experts; beating BF16 likely needs native FP4-MMA/CUTLASS-style kernels.
- **✅ STAGE 6 step 12 — P2-B routed gate/up grouping lower-bound PASSED; P2 remains viable as no-reload service path.**
  `scripts/spark_stage6_p2b_routed_gateup_grouping_poc.py` preserved the prefill router, precomputed token/slot groups
  by unique expert, then called the P2-A packed gate/up component once per unique expert.
  - **Route shape:** M=16 has **95** unique experts over 128 route slots; M=64 has **207** unique experts over 512 route
    slots. Largest M64 group has only **10** rows, so this is a high-dispatch shape.
  - **Numeric/no-shadow:** packed grouped gate/up vs BF16 grouped gate/up `cos≈0.999992`, rel_l2≈`0.004`, argmax match;
    packed bench ran after deleting `mlp.experts.gate_up_proj`, peak **0.955 GiB**.
  - **Latency vs BF16 gate/up:** M=16 **9.34 ms** vs BF16 **4.14 ms** (**0.443x**); M=64 **20.00 ms** vs BF16
    **8.47 ms** (**0.423x**). Not a BF16-speed win.
  - **Latency vs no-reload proof:** M=64 gate/up lower-bound **20.00 ms/layer** is far below P2 census `stream_bf16`
    **506 ms/layer**. Down projection will add work, but P2 remains plausible as a way to remove the **23-24 s**
    per-request reload.
  - **Decision:** continue to P2-C routed down projection and full routed output accounting. Do not integrate server-side
    until full one-layer routed MoE beats `stream_bf16` and stays memory-clean.
- **✅ STAGE 6 step 13 — P2-C active routed MoE lower-bound PASSED; no-reload path remains viable.**
  `scripts/spark_stage6_p2c_active_moe_lower_bound_poc.py` composed P2-B packed gate/up grouping with the existing
  packed down weighted-sum kernel. Routes were precomputed; shared expert, router timing, residual, and norms are out of
  scope.
  - **Bytes:** one-layer active BF16 expert shadow **1.500 GiB** vs packed **0.563 GiB** (**2.667x** smaller); after
    deleting BF16 active shadows, resident in the harness was **0.641 GiB**.
  - **Numeric/no-shadow:** M=16/64 active routed output vs BF16 active output `cos≈0.999981`, rel_l2≈`0.006`, argmax
    match; packed peak **0.642 GiB** after deleting `gate_up_proj` and `down_proj`.
  - **Latency vs BF16 active:** M=16 **10.13 ms** vs BF16 **6.42 ms** (**0.633x**); M=64 **23.83 ms** vs BF16
    **13.34 ms** (**0.560x**). Not a BF16-speed win.
  - **Latency vs no-reload proof:** M=64 **23.83 ms/layer** vs P2 census `stream_bf16` **506.22 ms/layer** (~**21x**
    faster). This is the important service signal: P2 can plausibly remove the **23-24 s reload** even if it is slower
    than resident BF16 prefill.
  - **Decision:** continue to P2-D shared/router-inclusive one-layer harness. Do not integrate server-side until full
    one-layer MoE beats `stream_bf16` and remains memory-clean.
- **⚠️ STAGE 6 step 14 — P2-D router/shared-inclusive one-layer hybrid is correct and memory-clean, but not a speed promotion.**
  `scripts/spark_stage6_p2d_one_layer_moe_hybrid_poc.py` adds router linear/top-k/softmax/eager grouping back into the
  timed path, keeps shared expert on the existing BF16 prefill path, deletes active BF16 expert shadows, and then runs
  packed active experts from NVFP4.
  - **Bytes:** one-layer active BF16 expert shadow **1.500 GiB** vs packed active **0.563 GiB**; shared BF16 is only
    **0.006 GiB**, so shared is not the current memory bottleneck. After deleting active BF16 shadows, harness resident
    was **0.641 GiB**.
  - **Numeric/no-shadow:** M=16/64 hybrid output vs full BF16 MoE `cos≈0.999997`, rel_l2≈`0.0023`, argmax match; packed
    peak stayed **0.642 GiB** after deleting `gate_up_proj` and `down_proj`.
  - **Latency vs BF16 full MoE:** M=16 **12.43 ms** vs BF16 **11.68 ms** (**0.940x**); M=64 **29.21 ms** vs BF16
    **21.65 ms** (**0.741x**). Not a BF16-speed win.
  - **Component costs exposed:** M=64 router/grouping **5.41 ms**, packed active precomputed **23.99 ms**, BF16 shared
    **0.046 ms**. The next bottlenecks are eager route/group scheduling and packed active latency, not shared expert.
  - **Latency vs no-reload proof:** M=64 **29.21 ms/layer** vs `stream_bf16` **506.22 ms/layer** (~**17x** faster).
    P2 still matters for removing the **23-24 s reload**, but it is not ready for serving integration.
  - **Decision:** bank P2-D as correctness/memory evidence. Next P2-E should reduce route/grouping overhead and packed
    active scheduling cost; do not wire into `resident_runner` until the one-layer hybrid has a clean speed story.
- **✅ STAGE 6 step 15 — P2-E scheduler/active retune PASSED; first one-layer packed MoE hybrid beats BF16 full MoE.**
  `scripts/spark_stage6_p2e_scheduler_active_retune.py` tested two low-risk changes before production edits:
  `argsort + unique_consecutive` route grouping and packed gate/up `block_inter=8`.
  - **Scheduler:** M=64 route/grouping improved **5.02 ms → 0.61 ms** (**8.17x**) by replacing per-expert mask scans
    with one sort + consecutive grouping. M=16 improved **2.32 ms → 0.34 ms** (**6.83x**).
  - **Packed active retune:** M=64 packed active improved **24.48 ms → 20.12 ms** with
    `block_t=32, block_inter=8, block_hidden=128, num_warps=4`. `block_inter=32` hit Spark shared-memory OOR
    (**114688 > 101376 bytes**). Scratch-buffer reuse was rejected (**0.963x/0.974x** for M16/M64).
  - **Hybrid vs BF16 full MoE:** with sort scheduler + `block_inter=8`, M=16 **8.87 ms** vs BF16 **11.58 ms**
    (**1.306x**); M=64 **20.25 ms** vs BF16 **21.41 ms** (**1.057x**).
  - **Numeric/no-shadow:** hybrid sort vs BF16 full MoE `cos≈0.999997`, rel_l2≈`0.0023`, argmax match; active BF16
    expert shadows were deleted and resident stayed **~0.641 GiB**.
  - **Decision:** bank P2-E and continue to P2-F: move the P2-E combination from harness composition to a
    flag-gated one-layer prefill replacement. This is still Python/Triton scheduling; the long-term llama.cpp chase
    remains CUDA C++ / CUTLASS-style grouped kernels plus C++ hot-path runtime when paired with FP4-MMA hardware.
- **✅ STAGE 6 step 16 — P2-F opt-in engine path PASSED; `p2e_hybrid` is no longer just a harness.**
  `engine/full_forward.py` now supports `LYNN_PACKED_PREFILL_SLOW_MODE=p2e_hybrid` with
  `LYNN_PACKED_PREFILL_P2E_LAYERS=<list>` and P2-E tile envs. Default remains off; non-selected layers fall back to
  `stream_bf16`.
  - **Engine-dispatch verification:** `scripts/spark_stage6_p2f_one_layer_replacement_verify.py` ran `_moe_forward()`
    itself, not a harness-only composition, after deleting `mlp.experts.gate_up_proj` and `mlp.experts.down_proj`.
  - **Numeric/no-shadow:** M=16/64 P2E vs BF16 full MoE `cos≈0.999997`, rel_l2≈`0.0023`, argmax match; active BF16
    shadows absent.
  - **Latency:** M=16 **8.23 ms** vs BF16 **11.63 ms** (**1.412x**) and `stream_bf16` **494.66 ms** (**60.08x**);
    M=64 **20.23 ms** vs BF16 **21.10 ms** (**1.043x**) and `stream_bf16` **504.93 ms** (**24.96x**).
  - **Memory:** P2E peak **0.641-0.643 GiB** vs `stream_bf16` peak **12.64 GiB** in the one-layer harness.
  - **Decision:** bank P2-F. Next P2-G should select multiple/all MoE layers with `p2e_hybrid` and measure cross-layer
    numeric drift, memory, and latency before any server default or multi-request promotion.
- **✅ STAGE 6 step 17 — P2-G 4-layer MoE smoke PASSED; P2E survives consecutive layers.**
  `scripts/spark_stage6_p2g_multilayer_moe_smoke.py` chains layers 0-3 with residual addition on synthetic hidden states
  (`h = h + MoE(h)`) to avoid the invalid raw-MoE collapse seen in the first diagnostic attempt.
  - **Bytes:** 4-layer active BF16 shadow **6.000 GiB** vs packed active **2.250 GiB**; after deleting active BF16
    shadows, harness resident was **2.525 GiB**.
  - **Numeric/no-shadow:** stream remains exact vs BF16; P2E vs BF16 after 4 layers has M=16 `cos=0.999999869`,
    rel_l2=`5.11e-4`, max_abs=`0.00390625`; M=64 `cos=0.999999840`, rel_l2=`5.65e-4`, max_abs=`0.0078125`;
    argmax matches for both.
  - **Latency:** M=16 **34.20 ms** vs BF16 **47.23 ms** (**1.381x**) and stream **2388.18 ms** (**69.82x**);
    M=64 **80.97 ms** vs BF16 **85.12 ms** (**1.051x**) and stream **2432.64 ms** (**30.04x**).
  - **Memory:** P2E peak **2.526-2.527 GiB** vs stream peak **14.525-14.526 GiB**.
  - **Decision:** bank P2-G. Next P2-H should move from MoE-only synthetic smoke to full transformer prefill with selected
    MoE layers on `p2e_hybrid`, then measure token/hidden agreement, memory, and latency before all-layer/server
    promotion.
- **✅ STAGE 6 step 18 — P2-H selected-layer full prefill PASSED; P2E is inside `_prefill_layer`.**
  `scripts/spark_stage6_p2h_selected_layer_prefill_smoke.py` runs the full engine prefill layer path: RMSNorm,
  linear/full attention cache population, residuals, and MoE FFN. Active BF16 expert shadows are deleted before packed
  modes run.
  - **Coverage:** full-attn layer 3 at T=16/64, linear-attn layer 0 at T=16, and mixed layers 0-3 at T=16 all passed.
  - **Numeric/no-shadow:** stream remains exact vs BF16. P2E vs BF16: full-attn T64 `cos=0.999999681`, rel_l2=`7.99e-4`;
    linear-attn T16 `cos=0.999999834`, rel_l2=`5.76e-4`; mixed 0-3 T16 `cos=0.999983027`, rel_l2=`5.83e-3`;
    argmax matches in all banked runs.
  - **Latency:** mixed 0-3 T16 **45.97 ms** vs BF16 **58.57 ms** (**1.274x**) and stream **2394.57 ms** (**52.09x**);
    full-attn T64 **20.51 ms** vs BF16 **20.75 ms** (**1.012x**) and stream **941.14 ms** (**45.90x**).
  - **Memory:** mixed 0-3 T16 P2E peak **2.606 GiB** after deleting **6.000 GiB** BF16 active shadow; stream peak
    **14.585 GiB**.
  - **Caveat:** this is synthetic-hidden selected-layer prefill, not tokenized full-model e2e. Mixed T64 is not banked;
    the old torch-only linear-attn prefill path remains a separate P2-J trace/kernel target.
  - **Decision:** bank P2-H. Next gates: P2-I expand selected MoE layers beyond the first four; P2-J isolate/replace
    the linear-attn prefill wall before server promotion.
- **✅ STAGE 6 step 19 — P2-I 8-layer selected-MoE expansion PASSED.**
  `scripts/spark_stage6_p2h_selected_layer_prefill_smoke.py` was re-run with `--layers 0-7 --seq-lens 16`, expanding
  selected full-prefill coverage from four layers to eight layers (linear, linear, linear, full repeated twice).
  - **Bytes:** 8-layer active BF16 shadow **12.000 GiB** vs packed active **4.500 GiB**; after deleting active BF16
    shadows, harness resident was **5.041 GiB**.
  - **Numeric/no-shadow:** stream remains exact vs BF16; P2E vs BF16 has `cos=0.999948277`, rel_l2=`1.017e-2`,
    max_abs=`0.0625`, argmax match.
  - **Latency:** T=16 **88.96 ms** vs BF16 **113.82 ms** (**1.279x**) and stream **4154.68 ms** (**46.70x**).
  - **Memory:** P2E peak **5.123 GiB** vs stream peak **17.102 GiB**.
  - **Decision:** bank P2-I. The remaining pre-server risk is now P2-J: trace/replace the old torch-only linear-attn
    prefill wall before scaling to all selected MoE layers or removing reload in serving.
- **✅ STAGE 6 step 20 — P2-J linear-attn prefill trace PASSED; next native target identified.**
  `scripts/spark_stage6_p2j_linear_attn_prefill_trace.py` isolates one BF16 linear-attention layer and times each
  prefill segment while checking exact output/state/conv agreement against `prefill_linear_attn()`.
  - **Numeric:** T=16/64/128/256/512 all exact vs `prefill_linear_attn` for output, recurrent state, and conv state.
  - **Dominant segment:** `chunk_gated_delta_with_state` is **71-76%** of traced wall time: T16 **2.56 ms**,
    T64 **2.42 ms**, T128 **2.68 ms**, T256 **3.47 ms**, T512 **4.08 ms**.
  - **Not the wall:** QKV/out projection, depthwise conv, and RMSNormGated are all small by comparison; the next-largest
    non-chunk segment is only **0.165-0.393 ms** across T16..512.
  - **Decision:** bank P2-J. Next gate P2-K should target a native/fused gated-delta prefill kernel, not more projection
    bridge tuning. Server promotion remains blocked until P2-K/P2-L and RC quality pass.
- **⚠️ STAGE 6 step 21 — P2-KA gated-delta native recurrent-loop PoC NUMERIC PASS / SPEED FAIL.**
  `scripts/spark_stage6_p2k_gated_delta_native_loop_poc.py` tries the cheapest native reuse path: loop over prefill tokens
  and call the existing single-token Triton decode recurrent kernel (`recurrent_gated_delta_fused_prepare_gqa`).
  - **Numeric:** T=16/64/128/256/512 all pass vs `chunk_gated_delta_with_state` for output/state with min cosine
    **0.999989555** and argmax match.
  - **Latency:** short T16/T64 look faster because the torch chunk reference has fixed overhead, but the one-launch-per-token
    shape falls over as T grows: T128 **4.05 ms vs chunk 2.46 ms (0.608x)**, T256 **8.03 ms vs 3.40 ms (0.424x)**,
    T512 **15.62 ms vs 4.16 ms (0.266x)**.
  - **Decision:** bank P2-KA as a rejected implementation path. The decode recurrent kernel is useful as a math oracle, but
    prefill needs P2-KB: a true chunk/block-level gated-delta kernel that processes multiple tokens per launch. Server
    promotion remains blocked until P2-KB/P2-L and RC quality pass.
- **✅ STAGE 6 step 22 — P2-KB gated-delta block Triton kernel CORE PASSED.**
  `triton_kernels/gated_delta.py::recurrent_gated_delta_block_gqa()` moves the P2-KA recurrent loop inside one Triton
  launch while keeping the gated-delta core isolated from projection/conv/g-beta fusion.
  - **Numeric:** T=16/64/128/256/512 all pass vs `_chunk_gated_delta_with_state` with min cosine **0.999989555**,
    max rel_l2 **0.004794770**, and argmax match. It also matches the P2-KA host-loop oracle to near machine precision.
  - **Latency:** T512 block kernel **1.16 ms**, vs P2-KA host loop **16.28 ms** (**14.04x**) and torch chunk reference
    **4.55 ms** (**3.92x**). T256 is **0.62 ms** vs host loop **8.89 ms** (**14.31x**).
  - **Decision:** bank P2-KB as a core-kernel pass. Next gate P2-L should wire it into `prefill_linear_attn` behind an
    opt-in flag and rerun selected-layer/full-prefill smoke before any server/default promotion.
- **✅ STAGE 6 step 23 — P2-L `prefill_linear_attn` block-kernel opt-in integration PASSED.**
  `LYNN_LINEAR_ATTN_PREFILL_BLOCK_GQA=1` now routes `prefill_linear_attn()` through
  `recurrent_gated_delta_block_gqa()` while leaving the default path unchanged.
  - **Numeric:** T=16/64/128/256/512 output, recurrent state, and conv state all pass with argmax match. Min output/state/conv
    cosine is **0.999983974**; max rel_l2 **0.005845026**.
  - **Latency:** T512 **2.18 ms** vs reference **5.58 ms** (**2.56x**); T128 **0.90 ms** vs **3.37 ms** (**3.74x**).
  - **Decision:** bank P2-L as an opt-in layer-level pass. Next gate P2-M should rerun selected-layer/full-prefill smoke with
    the block linear-attn flag plus existing P2-E MoE opt-in before server/default promotion.
