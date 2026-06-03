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
