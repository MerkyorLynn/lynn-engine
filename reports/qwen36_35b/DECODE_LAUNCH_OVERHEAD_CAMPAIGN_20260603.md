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
  codebuddy ✗ (not installed). Phase 0 kicking off + codex dispatched on launch-structure analysis.
