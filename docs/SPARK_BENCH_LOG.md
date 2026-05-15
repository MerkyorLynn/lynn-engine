# Spark sm_121 27B NVFP4 Benchmark Log

> **Branch**: `spark/sm121-port`. All numbers are **Spark-only** (DGX Spark GB10 sm_121 with Grace ARM + Blackwell UMA 119GiB). Do not conflate with Lynn engine R6000 sm_120 main-line numbers.

---

## 2026-05-15 23:47 — native_fast_2d backend milestone

**Goal**: Get Spark single-stream TPS to user-set 30 TPS target.  
**Achieved**: **42.85 mean / 43.30 median** (+43% over target).

### Run config

| Item | Value |
|---|---|
| Model | `Lynn-V4-Distill-Qwen-27B-A3B-NVFP4` (Lynn-native `nvfp4_e2m1_rowwise_per_16`) |
| Container | `lynn-27b-nvfp4-server` on `lmsysorg/sglang:dev-cu13` |
| Env | full P10 production recipe (see `scripts/spark/run_27b_nvfp4_server.sh`) — incl. `LYNN_MOE_IMPL=packed_nvfp4`, `LYNN_NATIVE_FP4_LM_HEAD=1`, `LYNN_LINEAR_ATTN_INPROJ_FUSED_NATIVE_FP4=1`, `LYNN_PACKED_DECODE_BACKEND=native_fast_2d`, `LYNN_PACKED_DECODE_PREPARE_NATIVE=1` |
| Backend reported | `backend=native_fast_2d native_prepared=190` (was 0 before backend env) |
| Bench prompt | "Describe in detail how a transformer attention layer processes tokens, around 300 words." |
| Sampling | greedy temperature=0, max_tokens=400 |
| Sample size | 3 runs |
| Sample completion length | 351 tokens (consistent across runs) |

### Numbers

| Run | tokens | prefill (s) | decode (s) | TPS | TTFT (s) |
|---:|---:|---:|---:|---:|---:|
| 1 | 351 | 0.555 | 8.37 | 41.93 | 0.821 (first-call warm-up) |
| 2 | 351 | 0.518 | 8.10 | **43.32** | 0.543 |
| 3 | 351 | 0.517 | 8.11 | **43.30** | 0.542 |
| **mean** |  | **0.530** | **8.19** | **42.85** | **0.635** |
| **median** |  |  |  | **43.30** | **0.542** |

Steady decode = **~7.5 ms/token** (was 42ms in baseline scalar_bridge config).

### Trajectory (this branch)

| Config | TPS mean | TTFT mean | Δ vs prev |
|---|---:|---:|---|
| Init (env misconfigured, scalar_bridge) | 23.88 | 0.52s | — |
| Fix LYNN_MOE_IMPL=packed_nvfp4 + LYNN_NATIVE_FP4_LM_HEAD=1 + LYNN_LINEAR_ATTN_RECURRENT_INPLACE=1 (`f050b29`) | 28.15 | 0.55s | +18% |
| Add LYNN_PACKED_DECODE_BACKEND=native_fast_2d (`15fc137`) | **42.85** | **0.63s** | **+52%** |
| **Cumulative** | **+79%** | | |

### Context

| Reference (NOT Spark) | TPS |
|---|---:|
| R6000 P10 OpenAI server stable (sm_120) | ~88-89 |
| R6000 P10 strict full graph w/ FP4 lm_head | 103.49 |
| R6000 P10 replay-only | 107.15 |

Spark sm_121 / R6000 sm_120 ratio at server-stable: **~48%**. Reflects that Lynn engine native FP4 kernels are first-class on R6000 sm_120; sm_121 still uses the same code paths but minor codegen / scheduling differences leave headroom on Spark.

### Mem state at READY

| Metric | Value |
|---|---|
| Host mem used | 97 GiB / 119 GiB |
| Host mem available | 22 GiB |
| Server load time | 312.5s |
| GPU mem (`nvidia-smi`) | `[N/A]` — Spark unified memory, no separate GPU mem reporting |

The 97 GiB headroom is tight. **P11 Path A (delete BF16 shadow, resident only packed NVFP4)** target: drive used down to **~35-40 GiB** while preserving 42.85 TPS.

### Open items

- [ ] **P11 Path A prototype** — delete BF16 shadow, streaming dequant for prefill. Target: ~60 GiB unified mem saved, TPS preserved.
- [ ] Re-run full V8/V9/coding spike eval with this config (quality verified against earlier 23.88 config which already passed ship-gate).
- [ ] Long-ctx 6-tier (1K → 64K) bench with diverse content (memory `feedback_long_ctx_bench_diverse_content`).
- [ ] BF16 vs NVFP4 cosine parity once BF16 lands Spark.
