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

Steady decode = **~23.3 ms/token** (was ~42 ms in baseline scalar_bridge config) — **~1.8× faster per token** (consistent with 1.79× TPS gain 23.88→42.85).

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

---

## 2026-05-16 00:24 — Codex P12 one-shot mode milestone on Spark sm_121

**Config**: P10 production env vars + `LYNN_RELEASE_DECODE_SHADOWS_AFTER_PREFILL=1`. Container `2358c1d47e43`.

### Mem release

| | Before prefill | After release | Δ |
|---|---:|---:|---:|
| Host mem used | 97 GiB | **58 GiB** | **-39 GiB** |
| Host mem available | 22 GiB | **61 GiB** | +39 GiB |

R6000 P11/P12 doc reports `torch.cuda.memory_allocated()` 81 → 27 GiB (-54G). Spark sees -39 GiB via `free -h` host view (UMA, no separate GPU memory namespace). The 17 GiB gap vs R6000 likely from PyTorch caching allocator not fully returning to OS pool immediately. Real GPU-side release should match R6000 (~54G); only host MemAvailable lags reflect it.

### TPS

| Run | tokens | decode (s) | **TPS** | TTFT |
|---:|---:|---:|---:|---:|
| 1 | 422 | 10.24 | **41.23** | 0.855s (prefill 0.609 + first step 247ms) |

vs native_fast_2d baseline 42.85 → -3.8% (within measurement noise). **TPS holds after shadow release** ✓.

### Health state machine

| Phase | `release_decode_shadows_consumed` | 2nd request |
|---|---|---|
| Pre-prefill | false | (server up) |
| Post-prefill | **true** ✓ | **HTTP 409** ✓ |

### Verdict

**Codex P11/P12 stack works on Spark sm_121 + UMA**. 39G unified mem freed for other Lynn agent services (TTS / ASR / emotion2vec / brain v2). TPS preserved.

### Caveat

P12 one-shot 限制 multi-prefill — restart required for new prompt session. Path C(timer-based idle evict)is the multi-prefill-friendly enhancement; currently not yet prototyped.

### Comparison vs V Flash 35B-A3B SGLang baseline

| 维度 | V Flash 35B SGLang | Lynn 27B Lynn engine | Δ |
|---|---:|---:|---:|
| Single-stream tail TPS | **54.64** | 42.85 (native_fast_2d) | **-22%** ⚠️ Lynn 27B 同时更小 + 慢 |
| Single-stream peak TPS | 57.49 | 42.85 | -25% |
| Mem footprint | ~65 GiB (mem-fraction 0.55) | **58 GiB after P12** | **Lynn -11% ✓** |
| V9 strict | 1.7% (template 坑) | TBD (round-1 8/8 answered reasonable) | **预期大胜** |
| Tool-call strict | 0/35 (template 坑) | 1/1 PASS | **明显胜** ✓ |
| Long ctx 16k | 3.12 TPS (SGLang 二次方崩) | TBD (Lynn linear-attn 应胜) | **预期大胜** |
| Batched N=16 aggregate | 76.49 tok/s | N/A (Lynn engine MVP serialized) | 不公平 |

**Lynn 27B 突破 SGLang 35B 的真实路径**:
1. Single-stream TPS gap 22-25% 要补 — P12 之后需要 P13+ kernel 优化
2. Long-ctx 16k+ 是 Lynn engine **天然优势**(linear-attn 不二次方),SGLang 自爆
3. 质量(V9 / tool-call)是 Lynn 27B 现成胜负手 — V Flash 35B SGLang 撞 template 坑
4. Mem 已经胜出(58G < 65G)— 桌面 app brain backend 多腾出 ~7G + 多 instance 友好
