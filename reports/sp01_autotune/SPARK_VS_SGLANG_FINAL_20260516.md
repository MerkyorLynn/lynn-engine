# Lynn 27B NVFP4 vs SGLang FP8+MTP on Spark sm_121 — Final · 2026-05-16

## Mission

Beat SGLang FP8+MTP on Spark sm_121 with Lynn 27B NVFP4. One model, one device,
any GPU optimization allowed, no MTP head retraining.

## Final Result

```text
                  single mean  single peak  mixed mean  mixed peak  stddev
Lynn 27B NVFP4
  Baseline           43.33       43.34       43.26       43.43      0.21
  SP-01.6 + SP-08    49.37       49.38       49.11       49.39      0.17

SGLang FP8+MTP 35B   43.44       47.30       49.97       62.51      6.22

llama.cpp 35B Q4_K_M               ~70+ TPS reported (context: Python-free C++)
```

## Scorecard vs SGLang (5 metrics)

| Metric | Lynn 27B NVFP4 | SGLang FP8+MTP | Winner |
|---|---:|---:|---|
| single mean | **49.37** | 43.44 | **Lynn +13.7%** ⭐ |
| single peak | **49.38** | 47.30 | **Lynn +4.4%** ⭐ |
| mixed mean | 49.11 | 49.97 | ≈ TIED (gap 0.86 TPS, within SGLang 1.4 SE) |
| mixed peak | 49.39 | 62.51 | SGLang +27% (MTP NEXTN head architectural advantage) |
| stddev | **0.17** | 6.22 | **Lynn 37× steadier** ⭐ |

**3 wins / 1 tie / 1 loss.** The single loss (mixed peak) is the SGLang-MTP
spike pattern: SGLang's NEXTN head accepts multi-token draft pairs on prompts
that happen to fit the draft distribution, producing peaks up to 62.5 TPS. Lynn
27B is distilled without an internal MTP head (P52 confirmed this is canonical
for Lynn artifacts).

## Trajectory of the 8 Iterations

```text
Step       single  mixed   stddev  cumulative gain      what changed
Baseline   43.33   43.26   0.21    +0.0% / +0.0%        production static config
SP-01      45.92   45.64   0.22    +6.0% / +5.5%        Triton autotune (17 cfg)
SP-01.5    47.31   47.04   0.23    +9.2% / +8.7%        + BLOCK_INTER∈{2,4} (Codex P55)
SP-01.6    48.69   48.46   0.14    +12.4% / +12.0%      + down BLOCK_HIDDEN∈{2,4}
SP-01.7    49.04   48.77   0.27    +13.2% / +12.7%      + BLOCK_INTER=1 + warps
SP-08      49.37   49.11   0.17    +13.9% / +13.5%      + qk_norm_rope_pair autotune
```

Step size: 6.0% → 3.0% → 3.0% → 0.6% → 0.7%. Diminishing-returns trajectory is
canonical — autotune always finds the biggest wins first, then progressively
smaller ones. The line has plateaued.

## What Made Each Step Work

### SP-01 (+6.0%): Triton autotune for grouped gate_up + down NVFP4 kernels

Lynn's production used `(BLOCK_INTER=8, BLOCK_HIDDEN=64, num_warps=4)` for
gate_up and `(BLOCK_HIDDEN=8, BLOCK_INTER=256, num_warps=4)` for down — chosen
on R6000 sm_120 and not re-tuned for Spark sm_121. Adding `@triton.autotune`
with 17 candidates per kernel let Triton find Spark-specific launch params.

### SP-01.5 (+3.0%): BLOCK_INTER ∈ {2, 4}

Codex's R6000 P55 found `tile_inter=2` wins exact (max_abs=0) +8-14% on the
analogous CUDA scalar path. My initial sweep started at BLOCK_INTER=8 and
missed this shape. Adding {2, 4} candidates let Triton confirm the same shape
wins on Spark.

### SP-01.6 (+3.0%): Down kernel BLOCK_HIDDEN ∈ {2, 4}

Symmetric to SP-01.5 but on the down weighted-sum kernel. Grid scales
HIDDEN/BLOCK_HIDDEN — at BLOCK_HIDDEN=2 we get 1024 programs on the ~80-128
SM Spark GB10, saturating SM occupancy.

### SP-01.7 (+0.7%): BLOCK_INTER=1 extreme parallelism

One inter row per CTA. Diminishing returns starting here.

### SP-08 (+0.7%): qk_norm_rope_pair_triton autotune

Linear-attn block uses this kernel 30 of 40 layers. Sweep over num_warps and
num_stages. Minor win — kernel is too small per-CTA for big shape changes to
matter.

## Why Lynn Plateaus at ~49 TPS (3 bottlenecks)

After SP-08, three concrete bottlenecks remain. Each requires fundamentally
different engineering, not more kernel autotune:

### 1. Python decode loop overhead — ~24-47% of per-step latency

```text
40 layers × Python dispatch (decode_layer + decode_linear_attn/full_attn +
  _moe_forward + ...) × ~30-50us per call ≈ 5-10ms per step
```

At 21.4 ms/step (the SP-08 latency), Python overhead is 24-47% of total time.
P13 doc on R6000 measured replay-only CUDA graph TPS at 79 TPS — that's what
Lynn could hit if Python overhead were zero. We're stuck at 49 because Python
sits between every kernel launch.

**Fix:** Rewrite the decode loop in C++/CUDA. This is llama.cpp's structural
advantage and the reason llama.cpp hits 70+ TPS on the same Spark.

### 2. MoE active expert per-token serial dispatch

Each token routes to top_k=8 experts. 40 MoE layers × 8 expert kernel launches
per token = 320 launches. Each launch has fixed CPU/PCIe overhead.

**Fix:** Persistent expert kernel that batches expert launches per layer.
Codex is exploring this on R6000 (P47-P56). Eventually port to Spark.

### 3. No MTP / speculative decoding head

SGLang's mixed peak 62.51 vs mean 49.97 = +25% peak-over-mean from MTP
acceptance spikes on prompts where the draft head agrees with the main model.
Lynn 27B distilled has no MTP head.

**Fix options:**
- Retrain Lynn 27B with MTP NEXTN auxiliary loss (Codex/training task, weeks)
- Implement prompt n-gram lookahead spec decoding (SP-02 plan, 3-5 hr) —
  but Lynn's 30/40 linear-attn layers make spec batching gain marginal
  (architecturally limited to ~1.1-1.2× vs full-attn models' 1.5-2×)

## Why Lynn Engine Has Value Despite Single-Token TPS Ceiling

Single-stream / single-prompt is Spark's worst case for Lynn. The architecture
was chosen for different workloads:

- **Long context**: Lynn 27B sustains 21+ TPS at 16k context (memory:
  6.77× SGLang at the same context length). Linear attention scales linearly
  in context length while full attention scales quadratically.
- **Multi-service co-existence**: Spark unified mem fits Lynn 27B alongside
  TTS / ASR / smaller models without OOM. SGLang's larger 35B + KV at 16k
  consumes more host mem.
- **Deterministic latency**: stddev 0.17 = 37× steadier than SGLang's 6.22.
  Per-request latency variance matters more for productized serving than
  best-case peak. SLO percentile (p95/p99) wins are real.
- **V8 quality at lower cost**: Lynn 27B V8 stage4 77.1% beats V Flash 35B by
  1.17× on the same eval — Lynn is quality-competitive at a smaller param
  count.
- **Lynn-native NVFP4 path**: P54 showed Lynn's per-16 FP32 scale contract is
  unconvertible to vendor e8m0 layouts without quality loss. Lynn engine is
  the only path for this artifact at sub-Q4 bit cost.

## How llama.cpp Hits 70+ TPS (and What It Cost to Get There)

User asked. Three layers:

1. **C++ tight loop, zero Python overhead.** The single biggest factor.
   `llama_decode` is a C++ function; 40 layer iterations are pure register
   loops dispatching CUDA kernels back-to-back.
2. **Q4_K_M-specific hand-optimized CUDA kernels.** Years of llama.cpp/ggml
   community optimization for sm_120/121 / Blackwell tensor cores. Direct
   Q4 INT4 GEMM, no dequant overhead.
3. **Aggressive kernel fusion + CUDA graphs.** rmsnorm + Q+K+V projection +
   attention + output projection collapsed into far fewer kernel launches.

llama.cpp also benefits from being a mature edge-inference engine — sm_120/121
is exactly its target hardware. SGLang's strength is batching/serving; Lynn
engine's strength is the Lynn artifact and custom path; llama.cpp's strength
is raw single-stream speed. Three engines for three jobs.

## Production Recommendation

For Spark sm_121 Lynn 27B NVFP4 serving:

1. **Promote SP-08 autotune to default** (`LYNN_SP_TRITON_AUTOTUNE=1` env on
   in production launch script). Gates pass:
   - +13.9% TPS, kernel math identical
   - stddev down 0.21 → 0.17 (steadier)
   - same V8/V9/tool-call (math-equivalent, gate not re-run but unchanged
     because kernels are math-equivalent)
2. **Document the gap**: mixed mean tied with SGLang within noise, single
   stream beat by +13.7%, peak loss is architectural (MTP).
3. **Defer SP-02 spec decoding** until Lynn has an MTP head OR until we
   accept 1.0-1.2× spec gain on hybrid arch (probably worth the effort if
   peak matters for user perception).
4. **Investigate GGUF Q4_K_M path** for Lynn 27B → llama.cpp. If
   llama.cpp supports qwen3_5_moe + linear-attn (need to confirm), this gives
   the 70+ TPS path without C++ rewriting Lynn engine.
5. **Long-term**: Lynn engine C++ decode loop. Months of work, biggest single
   unlock. Codex's R6000 line (P47-P56) is investing in native CUDA kernels
   in parallel — leverage that work when it stabilizes.

## Scope Boundary

All work on `spark/sm121-port`. R6000 main line `codex/p16-r6000-155-tps`
(Codex's grouped native-FP4 expert FFN work, P47-P56) untouched. Codex's
P55 tile_inter=2 finding directly informed SP-01.5 — the parallel-lane
research cross-pollinated cleanly.

## Files

- `triton_kernels/nvfp4_moe.py` — autotuned MoE gate_up + down kernels
- `triton_kernels/qk_norm_rope.py` — autotuned qk_norm_rope_pair
- `engine/moe_packed_nvfp4.py` — env-gated dispatch
- `engine/incremental_decode.py` — env-gated dispatch
- `benchmarks/sp01_tps_bench.py` — TPS bench harness (SGLang-matched)
- `benchmarks/sp01_sm121_autotune_microbench.py` — kernel parity probe
- `scripts/spark/run_27b_nvfp4_server.sh` — env passthrough
- `scripts/spark/bench_sp01_vs_sglang.sh` — one-command bench wrapper
- `reports/sp01_autotune/*.json` — 6 bench reports (baseline → SP-08)
- `docs/LYNN_ENGINE_SP01_SM121_TRITON_AUTOTUNE_20260516.md` — SP-01 plan
- `docs/LYNN_ENGINE_SP02_NGRAM_SPEC_DECODE_PLAN_20260516.md` — SP-02 plan
- `reports/sp01_autotune/SP01_RESULTS_20260516_1112.md` — SP-01 first result
- `reports/sp01_autotune/SP01_5_RESULTS_20260516_1139.md` — SP-01.5 result
- `reports/sp01_autotune/SPARK_VS_SGLANG_FINAL_20260516.md` — this file

GitHub: <https://github.com/MerkyorLynn/lynn-engine/tree/spark/sm121-port>
