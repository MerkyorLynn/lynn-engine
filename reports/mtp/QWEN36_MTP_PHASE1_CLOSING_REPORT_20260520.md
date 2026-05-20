# Qwen3.6-35B MTP Phase 1 Closing Report · 2026-05-20

## Decision

**Phase 1 MTP K=2 batched correctness chase is CLOSED.** After M21→M26
(six diagnostic probes + four engine opt-in switches), the residual
batched-vs-baseline exact-match gap is **3/6** and cannot be fixed
within tonight's diagnostic budget. Sequential `spec_k1` already ships
6/6 exact and remains the correctness-clean K-step speculative path.
Phase 2 (Spark FP8 fused gate/up kernel) is the next-ROI investment to
reach the SGLang FP8 + MTP 60–70 TPS target.

## Phase 1 deliverables (kept opt-in, not promoted to default)

All switches default-OFF; runner behavior is unchanged.

| Knob | Effect | Status |
|---|---|---|
| `LYNN_FULL_ATTN_K2_BACKEND=t1_loop` | Replace `decode_full_attn_k2` with 2× `decode_full_attn` (T=1) | M18 layer-strict + M19 production |
| `LYNN_MTP_K2_LINEAR_ATTN_MODE=t1_loop` | Replace `decode_linear_attn_k2` with explicit per-position chain | M18 redundant — linear-attn k2 already strict |
| `LYNN_MTP_K2_VERIFY_MODE=t1_canonical` | Short-circuit batched to `speculative_step_k1` | M14–M17 oracle path |
| `LYNN_MTP_K2_LM_HEAD_MODE=bf16` | Force BF16 F.linear for K=2 lm_head | M21 diag (no exact change) |
| `LYNN_MTP_K2_ACCEPT_SOURCE=canonical_t1` | Use shadow T=1 chain argmax for accept decision | M21 diag (no exact change) |
| `LYNN_MTP_K2_REJECT_ROLLBACK=full_state` | Replace lean restore with `runner._restore_state` on REJECT | M22 diag (byte-identical no-op) |
| `LYNN_MTP_K2_COMMIT_SOURCE=canonical_t1` | M23 full canonical T=1 commit on accept | 6/6 oracle (7.85 TPS) |
| `LYNN_MTP_K2_COMMIT_REPAIR={hidden,next_pending,hidden_next_pending,state,full_canonical}` | M24 fine-grained accept-commit bisect | only `state`/`full_canonical` reach ≥5/6 |

## Result table (Spark `--max-new 64` apples-to-apples eager + shadow off)

| Config | Exact | Mean prefix | Accept | Effective TPS | Ratio vs eager baseline |
|---|---:|---:|---:|---:|---:|
| baseline_greedy (eager) | 6/6 | 64.0 | — | ~32 | 1.00 |
| **baseline_greedy (graph, production)** | **6/6** | **128+** | — | **38.96** | **(reference ship)** |
| spec_k1_sequential | **6/6** | 57.3 | 79.18% | 26.4 | 0.83 |
| spec_k1_batched_default | 3/6 | 44.0 | 74.85% | 24.5 | 0.77 |
| +M22 fullrollback | 3/6 (byte-identical) | 44.0 | 74.85% | 24.1 | 0.76 |
| +M24 hidden | 3/6 (byte-identical) | 44.0 | 74.85% | 14.0 | 0.44 |
| +M24 next_pending | 3/6 | 44.0 | 74.99% | 14.0 | 0.44 |
| +M24 hidden_next_pending | 3/6 | 44.0 | 74.99% | 13.9 | 0.44 |
| +M24 state | 5/6 | 54.8 | 78.58% | 14.4 | 0.45 |
| +M24 full_canonical (=M23) | 6/6 | 57.3 | 79.18% | 14.4 | 0.45 |
| +M26 no expandable_segments | 3/6 (byte-identical) | 44.0 | 74.85% | 24.6 | 0.77 |

Raw artifact: [`reports/mtp/mtp_m26_no_expandable_20260520_134725.json`](mtp_m26_no_expandable_20260520_134725.json).

## What we proved (elimination chain)

1. **M18**: K=2 layer outputs are bit-strict under `LYNN_FULL_ATTN_K2_BACKEND=t1_loop`. ✅
2. **M21**: lm_head per-row dispatch and K=2 accept argmax are NOT the sole bug (each only shifts accept rate ~1pp; exact stays at 0/6 at `--max-new 128`).
3. **M22**: KV cache restore on REJECT path is a byte-identical no-op. KV residue is NOT the bug.
4. **M24** (`hidden` mode is byte-identical to default): K=2 `next_base_hidden` is bit-equal to canonical T=1 `h_after_draft`. M18 strict diff holds multi-round; outputs are NOT the bug.
5. **M25** (direct state-delta probe): K=2-committed state and canonical-T=1-committed state are **byte-equal** in every captured tensor (KV, recurrent, conv, seq_len) for two distinct prompts. ✅ Captured state is NOT the bug.
6. **M26** (drop `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`): byte-identical to default M22/M24 numbers. CUDA allocator is NOT the bug.

The residual is therefore **process-side state outside captured tensors AND independent of the CUDA caching allocator**. Remaining candidates: Triton autotune cache, native FP4 lm_head workspace, runner-level lazy buffers, or stream/event ordering. None of these were tonight's lever.

## Why this isn't worth chasing further now

- Even with **full canonical commit** (M23 / M24 full_canonical, the *only* 6/6 batched mode), effective TPS is 14.4 — **63% slower than eager baseline (24.5) and 73% slower than graph baseline (38.96)**. The batched path is correctness-clean only at speed cost; it is never a TPS win.
- Sequential `spec_k1` is already 6/6 exact at 26.4 TPS — also slower than the graph baseline. **No K-step speculative path on Spark currently beats the graph baseline.**
- The Spark target is SGLang FP8 + MTP single-stream **60–70 TPS** (memory `reference_dgx_spark_llm_candidates_0501`). To close that gap Lynn needs to switch the matmul path from W4A16-dequant-to-BF16 to W4A8 with native FP8 MMA + fused kernels (memory `reference_spark_fp8_w4a8_design_strategy_20260519`). MTP correctness is orthogonal to that.

## Phase 2 — opening

**Goal**: single-stream 30–50 TPS on Spark W4A8 + FP8 MMA + fused gate/up. Stretch goal 60–70 to match SGLang FP8 + MTP.

**Task list** (per `reference_spark_fp8_w4a8_design_strategy_20260519`):

1. **Offline NVFP4 → FP8 repack tool** — `scripts/spark_pack_w4a8_fp8.py`
   - Input: Lynn-native NVFP4 packed weights (E2M1 + per-16 BF16 scale)
   - Output: FP8 E4M3 + per-row/per-tensor scales, col-major layout
   - Verify: dequant cos > 0.999 vs original NVFP4
2. **Fused gate/up Triton kernel** — `triton_kernels/spark_fp8_gate_up_fused.py`
   - BF16 act × FP8 weight (concatenated [gate; up]) → split + SwiGLU → output BF16 for down_proj
   - Targets `cuBLASLt FP8 GEMM` sweet spot via larger N
3. **Resident decode loop integration** — `engine/resident_runner.py`
   - sm_121 detect → FP8 fused path on
   - sm_120a R6000 → keep FP4 native (not affected)
   - Other → BF16 dequant fallback
4. **Optional CUDA graph wrapper** (stretch)

Sequential MTP and graph baseline both remain as the existing
correctness-clean paths during Phase 2 work.

## Cross-reference

- M18 layer sweep: [`32731a9`](https://github.com/MerkyorLynn/lynn-engine/commit/32731a9)
- M21 lm_head + accept_source bisect: [`432a29e`](https://github.com/MerkyorLynn/lynn-engine/commit/432a29e)
- M22 reject rollback (no-op): [`819cdf6`](https://github.com/MerkyorLynn/lynn-engine/commit/819cdf6)
- M23 canonical commit (6/6 oracle): [`cf0a187`](https://github.com/MerkyorLynn/lynn-engine/commit/cf0a187)
- M24 commit repair bisect: [`56f8a6a`](https://github.com/MerkyorLynn/lynn-engine/commit/56f8a6a)
- M25 state-delta probe: [`c4c20c8`](https://github.com/MerkyorLynn/lynn-engine/commit/c4c20c8)
- M26 (this report)
- Strategy memory: `project_lynn_engine_t1_only_kernel_contract_20260519`, `reference_spark_fp8_w4a8_design_strategy_20260519`
