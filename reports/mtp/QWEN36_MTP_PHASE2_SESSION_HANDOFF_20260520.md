# Lynn engine Spark MTP / FP8 session handoff · 2026-05-20

## TL;DR

* **MTP Phase 1 closed**: M21–M26 ruled out lm_head dispatch (M21), KV cache restore (M22), per-output commit repair (M24), captured-tensor state (M25), and CUDA allocator (M26) as sole sources of K=2 batched residue. Sequential `spec_k1` 6/6 exact at 26.4 TPS ships as the correctness-clean K-step path. Batched `spec_k1_batched` stays opt-in at 3/6 — root cause is process-side state outside captured tensors (autotune cache / native FP4 lm_head workspace / runner lazy buffers), not lever-able in single-session budget.
* **Phase 2 FP8 opened**: offline NVFP4 → FP8 E4M3 repack tool v0 ships ([`scripts/spark_pack_w4a8_fp8.py`](../../scripts/spark_pack_w4a8_fp8.py), self-test PASS cos > 0.999 across 4 shapes × 2 granularities). The Triton fused gate/up kernel (Phase 2 step 2) and resident_runner integration (step 3) are multi-day work and **deferred to next session**.
* **TPS target gap remains**: Lynn-native W4A16 NVFP4 baseline 38.96 TPS (graph mode, already beats SGLang BF16 30.14). Closing the gap to SGLang FP8 + MTP 60–70 TPS requires the Phase 2 Triton kernel + integration; not achievable in one session.

## What landed tonight

### Phase 1 — MTP K=2 batched correctness chase (closed)

Probes / engine opt-in switches (all default OFF):

| Probe | Commit | Finding |
|---|---|---|
| M18 layer-type sweep | [`32731a9`](https://github.com/MerkyorLynn/lynn-engine/commit/32731a9) | `LYNN_FULL_ATTN_K2_BACKEND=t1_loop` necessary + sufficient at layer-level (M18 strict diff) |
| M21 lm_head + accept_source | [`432a29e`](https://github.com/MerkyorLynn/lynn-engine/commit/432a29e) | `LYNN_MTP_K2_LM_HEAD_MODE=bf16` and `LYNN_MTP_K2_ACCEPT_SOURCE=canonical_t1` perturb accept signal ~1pp each, neither fixes exact alone |
| M22 reject rollback | [`819cdf6`](https://github.com/MerkyorLynn/lynn-engine/commit/819cdf6) | `LYNN_MTP_K2_REJECT_ROLLBACK=full_state` byte-identical no-op — KV not the bug |
| M23 canonical commit | [`cf0a187`](https://github.com/MerkyorLynn/lynn-engine/commit/cf0a187) | Full canonical T=1 chain commit reaches 6/6 at 7.85 TPS — oracle but slow |
| M24 commit repair bisect | [`56f8a6a`](https://github.com/MerkyorLynn/lynn-engine/commit/56f8a6a) | `hidden` byte-identical to default → K=2 outputs bit-equal to canonical; only `state`/`full_canonical` reach ≥5/6 |
| M25 state-delta probe | [`c4c20c8`](https://github.com/MerkyorLynn/lynn-engine/commit/c4c20c8) | Captured tensor state (KV + recurrent + conv + seq_len) **byte-equal** between K=2 and canonical T=1 commit — bug is process-side |
| M26 allocator probe | [`3d697ad`](https://github.com/MerkyorLynn/lynn-engine/commit/3d697ad) | Dropping `expandable_segments` byte-identical — CUDA allocator not the bug |

### Phase 2 — Spark FP8 path (opened)

| Step | Commit | Status |
|---|---|---|
| 1. Offline NVFP4 → FP8 repack tool v0 | [`e370cbc`](https://github.com/MerkyorLynn/lynn-engine/commit/e370cbc) + [`ca77375`](https://github.com/MerkyorLynn/lynn-engine/commit/ca77375) | Self-test PASS cos > 0.999 |
| 2. Fused gate/up Triton kernel | — | **DEFERRED** (estimated 1-2 days) |
| 3. resident_runner sm_121 integration | — | **DEFERRED** (depends on step 2) |
| 4. CUDA graph wrapper | — | **DEFERRED** (stretch goal) |

## Final TPS table (Spark `--max-new 64` apples-to-apples eager + shadow off where applicable)

| Config | Exact | Effective TPS | Ratio vs graph baseline (38.96) | Ship status |
|---|---:|---:|---:|---|
| **baseline_greedy (graph, production)** | 6/6 | **38.96** | **1.00** | ✅ current Lynn ship |
| spec_k1_sequential | 6/6 | 26.4 | 0.68 | ✅ opt-in correctness oracle |
| spec_k1_batched_default | 3/6 | 24.5 | 0.63 | ❌ NOT promoted |
| baseline_greedy (eager) | 6/6 | ~32 | 0.82 | (reference for spec comparison) |
| spec_k1_batched + canonical commit (M23 oracle) | 6/6 | 7.85 | 0.20 | ❌ NOT promoted (too slow) |
| (reference) SGLang BF16 single-stream | 6/6 | 30.14 | 0.77 | per memory |
| (target) SGLang FP8 + MTP single-stream | 6/6 | **60–70** | 1.54–1.80 | per memory |

## Honest assessment of the 60–70 TPS target

* The gap from Lynn 38.96 → SGLang FP8 + MTP 60–70 is ~1.54–1.80×.
* MTP alone cannot close it on Spark — spec is eager-only ("graph captures cannot be rolled back on draft reject"), and eager spec TPS (~25) is below graph baseline (38.96).
* The fundamental hardware lever Lynn isn't using yet is **Spark sm_121 native FP8 MMA** (162 TFLOPS peak, 1.64× BF16). SGLang FP8 uses it; Lynn's NVFP4 path goes through dequant-to-BF16 and never touches the FP8 MMA unit.
* The Phase 2 design (offline FP8 repack + fused gate/up Triton kernel + resident_runner integration) is the path. The offline repack is straightforward (Phase 2 step 1 done tonight). The Triton kernel + integration is **multi-day engineering**, not a single-session probe.

## Recommended next session

1. **Phase 2 step 2** — Triton fused gate/up FP8 kernel
   * Input: BF16 activation, FP8 E4M3 weight (concatenated [gate; up]), per-row weight scale
   * Internal: per-block activation cast to FP8, `tl.dot` FP8×FP8 → F32 accum, scale apply, SwiGLU, output BF16
   * Verify: cos > 0.999 vs BF16 reference; perf sweep block sizes; autotune
2. **Phase 2 step 3** — `engine/resident_runner.py` sm_121 detection, FP8 path branch when `_dense_ffn_forward` is called and sidecar `mlp._fp8.gate_up_proj.weight_fp8` keys exist
3. **Phase 2 step 1.5** — extend repack tool to full Lynn-native model dir (manifest-driven, all layers/experts at once; produce a sidecar `lynn-w4a8-fp8` model dir alongside the existing W4A16 one)
4. **End-to-end measure** — TPS regression vs graph baseline 38.96; target 50+ for promotion, stretch 60+

## Branch state

- Branch: `claude/mtp-k2-strict-diag-20260520`
- Pushed: HEAD = `ca77375` on origin/main (also serves as the diagnostic branch)
- Worktree: `/Users/lynn/Downloads/Lynn/worktrees/codex-main-overnight`

## Cross-reference

- Phase 1 closing: [`reports/mtp/QWEN36_MTP_PHASE1_CLOSING_REPORT_20260520.md`](QWEN36_MTP_PHASE1_CLOSING_REPORT_20260520.md)
- Strategy memory: `project_lynn_engine_t1_only_kernel_contract_20260519`, `reference_spark_fp8_w4a8_design_strategy_20260519`
- Spark baseline TPS: `project_spark_single_stream_tps_baseline_20260518`
- SGLang FP8 + MTP target: `reference_4090_llamacpp.md:58`, `reference_dgx_spark_llm_candidates_0501.md:94`
