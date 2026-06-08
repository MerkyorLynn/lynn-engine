# Lynn Engine × NVFP4 — Full Benchmark Appendix (Qwen3.6-35B-A3B)

> Raw data backing the blog posts. Two Blackwell-class GPUs, same model, all numbers tagged with hardware + condition + source. `✅` = first-party measured / direct-checked.
> Model: **Qwen3.6-35B-A3B** (MoE, 3B-active). Hardware: **DGX Spark (GB10, sm_121, no FP4-MMA, LPDDR5X ~240 GB/s)** + **RTX PRO 6000 Blackwell (sm_120, has FP4-MMA)**. Dates: 2026-06-03 → 06-05.

---

## 1. Spark (sm_121) — single-stream decode, self-built NVFP4 engine

| metric | value |
|---|---|
| Start → end of optimization | **~36 → ~45 TPS** (+26% in-session; +16% vs 38.96 W4A16 anchor) |
| Quality (token-exact) | 40/40 prompts byte-identical; MMLU **84.40** / GPQA-Diamond **49.49** |
| Decode bottleneck | **launch/dispatch-bound: ~1527 kernel launches/token, only 37% of 240 GB/s** (NOT bandwidth-bound — the "read-4bit→bandwidth→70" thesis was probed and **falsified**) |
| Same-card llama.cpp Q4_K_M | **69.77 TPS** (plain) / **79** (APEX-MTP config) → self-built is **~1.5× slower** |

**5 RC-validated kernel fusions (the +26%):**

| fusion | gain |
|---|---:|
| Fused RMSNorm (92 sites, 6-8→1 launch) | +8.7% |
| NVFP4 bf16-out copy-elision (no cast chain) | +3.3% |
| Shared-expert fusion (gate_up→SwiGLU→down→gate→add) | +2.8% |
| Linear-attn g/beta-fold (register-local per-head) | +2.6% |
| Full-attention fusion (qk-norm+rope+KV+gate) | +0.6% |

**Verdict:** research asset, not Spark's best backend. Spark has no FP4-MMA → NVFP4 is meaningless there; Q4_K_M wins.

---

## 2. R6000 (sm_120) — FP4-MMA's home turf, grouped-MoE A/B → NO-GO (2026-06-04)

Real full-active-MoE **prefill** boundary (P138 fixture):

| kernel | latency | vs packed P3 |
|---|---:|---:|
| **packed-NVFP4 P3 (current best Lynn)** | **0.0925 ms** | 1.00× |
| new grouped-MoE FP4-MMA candidate | **0.2668 ms** | **2.88× SLOWER** ❌ |
| W4A16 dequant baseline | 0.602 ms | (only beats this — doesn't count) |

Numeric: max-abs-err 0.0137, rel_l2 0.211, cosine 0.978. **In FP4-MMA's own best scenario (prefill), our new kernel lost ~3× to our own old packed kernel.** Private grouped-MoE FP4-MMA kernel line closed.

**Two physics reasons it can't win:**
1. FP4-MMA tensor cores accelerate **compute (prefill / batched GEMM)**, not single-token decode. Upstream llama.cpp measured: native FP4 → **prefill +45%, decode flat**.
2. The target model's MoE is **W4A16** (✅ `nvidia/Qwen3.6-35B-A3B-NVFP4` = `MIXED_PRECISION`: MoE `W4A16_NVFP4`, attn/KV FP8) → **no FP4×FP4 activation path exists**, even on B200. We wrote a W4A4 kernel for a use case that doesn't exist.

---

## 3. R6000 (sm_120) — single-stream baselines (reference)

| engine | metric | value |
|---|---|---:|
| llama.cpp Q4_K_M | single-stream decode tok/s | **~207** |
| Lynn self-built NVFP4 W4A16 | single-stream strict tok/s | **~108** |
| vLLM NVFP4 Marlin | c=1 *serving* request | 175.09 |

⚠️ Discipline: single-stream **decode** (207) ≠ c=1 **serving** (175, request latency in a batching loop). Do **not** compare to Spark 38.96/42.85 (different hardware).

---

## 4. R6000 (sm_120) — "join the ecosystem": vLLM forced-Marlin NVFP4 serving ✅

vLLM 0.22.0, `inferRouter/Qwen3.6-35B-A3B-NVFP4` (vLLM-ready variant), `VLLM_MOE_FORCE_MARLIN=1`, Triton attn, FP8-E4M3 KV.

**Concurrency sweep (1024/256):**

| c | output tok/s | total tok/s | mean TTFT ms | p99 TTFT ms | mean TPOT ms |
|---:|---:|---:|---:|---:|---:|
| 1 | 175.09 | 875.44 | 88.45 | 91.52 | 5.39 |
| 8 | 809.78 | 4048.91 | 304.44 | 504.78 | 8.70 |
| 16 | 1288.91 | 6444.57 | 462.92 | 665.29 | 10.63 |
| 32 | 1872.41 | 9362.03 | 648.50 | 1009.78 | 14.58 |
| 16 (1024/1024) | 1466.87 | 2933.75 | 442.10 | 522.06 | 10.48 |

**Final soak (2026-06-05, release-grade, 0 failed):** c16 mean **1234**, c32 mean **1849**, **c64 mean 2434** (2442/2427), c8 512/128 interactive **832 / TTFT 190 ms**.

**Product soak (280 real-prompt requests, 0 failed):**

| workload | c | output tok/s | mean TTFT ms | p99 TTFT ms |
|---|---:|---:|---:|---:|
| short QA | 4 | 536.59 | 76.82 | 154.50 |
| mixed interactive | 8 | **938.10** | 84.71 | 113.08 |
| mixed serving | 16 | 1512.20 | 124.29 | 212.51 |
| mixed serving | 32 | **2215.56** | 148.01 | 176.37 |
| long-context | 8 | 814.76 | 400.10 | 651.95 |

**NVFP4 vs FP8 (same machine / same harness A/B):**

| c (1024/256) | NVFP4 tok/s | FP8 tok/s | NVFP4/FP8 |
|---:|---:|---:|---:|
| 1 | 175.09 | 131.10 | **1.336×** |
| 8 | 809.78 | 635.16 | **1.275×** |
| 16 | 1288.91 | 1131.46 | **1.139×** |
| 32 | 1872.41 | 1584.80 | **1.181×** |
| 16 (1024/1024) | 1466.87 | 1237.79 | **1.185×** |

(NVFP4 = vLLM-ready W4A16 variant; FP8 = official Qwen FP8. Same-machine route A/B, not strict same-checkpoint.) `atomic_add` rejected (neutral/negative). Low-latency variant c16 1024/1024 = **1506 / TTFT 398 ms**.

---

## 5. SGLang on sm_120 — NO-GO (tonight)

- SGLang 0.5.8 NVFP4 MoE on sm_120 = NaN/broken; 0.5.12 needs newer FlashInfer/kernel wheels; available wheels only ship SM90/SM100 ops.
- vLLM forced-Marlin **1289** (c16) vs SGLang best **~900** → vLLM **~34% faster**. Retest only with a known-good SM120 container.
- External ref (NVIDIA forum, Qwen3.5-397B, 4× R6000): Marlin W4A16 **50.5 tok/s** baseline winner; FlashInfer-CUTLASS 26-41 (brittle); SGLang 0.5.8 NaN; vLLM native CUTLASS ~5 (broken).

---

## 6. Speculative decode (MTP / DFlash) — lossless but no single-stream win

- MTP head is **strong**: W4A16 chat-forced-prefix accept **0.894** (top2 0.988, implied 1.89×).
- But strict token-exact commit path **loses to greedy**: Spark spec-k1 **23.71** vs greedy **34.78** (0.68×); the verifier/commit loop eats the accept gain.
- Root cause = head/serving-path mismatch: MTP head trained for native-FP4 activations, run on dequant path → mispredict. Spark accept **60.6%** ≈ vLLM-marlin **59%** (two dequant paths hit the same ~60%).
- Community bf16-grafted matched head (27B) → **87% accept, 1.74×**. Lynn's 60.6% matches this failure signature → "matched-head" is a model-artifact task, not a config sweep.
- DFlash: vLLM code paths present, draft weights absent → parked fallback, no speed claim.

---

## 7. Ecosystem state — NVFP4 is commoditized (verified 2026-06-04)

| project | status | evidence |
|---|---|---|
| llama.cpp GGUF NVFP4 type | ✅ merged | `GGML_TYPE_NVFP4=40`, PR #19769 (2026-03-11) |
| llama.cpp Blackwell FP4-MMA prefill | ✅ merged | PR #22196 (2026-04-28), `e2m1×e2m1` PTX, gate `BLACKWELL_MMA_AVAILABLE` (sm_120+) |
| llama.cpp grouped-MoE FP4 tensor-core | 🔴 closed gap | Issue #18250 "SM120 native NVFP4 MoE kernels" = **closed as not planned** |
| NVIDIA modelopt checkpoint | ✅ published | `nvidia/Qwen3.6-35B-A3B-NVFP4` (W4A16 MoE, FP8 attn) |
| vLLM NVFP4 MoE | ✅ merged | `CompressedTensorsW4A4Nvfp4MoEMethod` + PR #28892 (23.5-81.3% faster vs cutlass) |
| SGLang NVFP4 | ✅ in docs | issue #7994; official quant docs list NVFP4 |

**Conclusion:** format + weights + serving kernels all shipped by NVIDIA + OSS → no moat in a self-built NVFP4 engine. FP4-MMA helps **prefill + batched serving**, never single-stream decode.

---

## 8. Reproducibility caveats / flags

1. **Official `nvidia/...-NVFP4` failed to load in vLLM 0.22** (quantized `lm_head` shape mismatch) → used `inferRouter/...-NVFP4` (BF16 lm_head). Valid serving-route probe, not exact-official-checkpoint.
2. c16 1024/256 variance: synthetic bench 1289 vs soak rep1 1178 (~8%, workload distribution). c64 across runs 2434 vs 2431 (<0.1%, consistent).
3. FP8 c=1 TTFT anomaly 700 ms = Triton-sampler fallback overhead (FlashInfer broken on this SM120 stack), not fundamental to FP8.
4. Known failure boundaries on this stack: FLASH_ATTN+fp8_e4m3 KV (invalid); FP8-FlashInfer (needs known-good container); native FlashInfer FP4 MoE (segfault).
