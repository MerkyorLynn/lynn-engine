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
