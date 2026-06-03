# Lynn Engine

> **🚀 Strategic correction (2026-06-03): Lynn engine is restarted as a parallel mainline, target = benchmark against (rival) llama.cpp — no longer "downgraded to R&D / just use llama.cpp".** The client still ships llama.cpp/GGUF short-term as a **pragmatic default backend**; the engine advances in parallel — same model + hardware vs llama.cpp, endgame = chewing through the **fused 4-bit / zero-shadow kernel** ourselves (single-projection PoC → all dense + drop shadow → MoE grouped experts → fuse to cut launches, gate + RC each step), approaching and even **surpassing it on FP4-MMA cards (R6000 class)**. **If you're going to build an engine, build the kernels yourself.**
>
> **✅ Latest banked Lynn-native core win:** 35B NVFP4 serving now releases the decode-phase BF16 dequant-shadow from resident memory: **88→28 GiB resident (~60 GiB freed)**, token-exact, 0.998× TPS (no regression). `server/openai_http.py` wires the `reload→prefill→release→decode` service cycle. **P0.1/P0.2 packed-prefill gates have now passed:** after releasing the 60 GiB shadow, `stream_bf16` proves token-exact no-reload prefill (peak **40.28 GiB**, proof prefill **20.75 s**); P0.2 resident inventory shows only **4.72 GiB** BF16 after release. **P1 dense PoC passed; P1-A batched bridges rejected. P2 grouped MoE has reached P2-M:** P2-F wires `LYNN_PACKED_PREFILL_SLOW_MODE=p2e_hybrid`; P2-H/P2-I selected full prefill keep passing numeric/no-shadow/speed gates; P2-J identifies the linear-attn wall; P2-KB block Triton kernel passes the core gate; P2-L wires it into opt-in `prefill_linear_attn`; **P2-M selected-layer full prefill passes:** layers 0-3 with P2E MoE + block linear-attn, T16 **39.12ms vs BF16 57.40ms = 1.467x**, T64 **84.26ms vs BF16 93.59ms = 1.111x**, numeric/no-shadow pass. Next gate: **P2-N wider layer coverage + RC/server smoke**. Python is the control/verification plane; catching llama.cpp requires CUDA C++/CUTLASS/native kernels.

> **🆕 2026-06-03 Decode kernel-launch campaign — Spark NVFP4 35B-A3B single-stream 38.96 → ~45 TPS, RC quality-identical.**
> Decode is launch-bound (census: **~1527 CUDA launches/token**, ~40% of token time is CPU-side dispatch). We fused launch clusters + elided copies: **fused RMSNorm (biggest) / shared-expert / linear-attn g/beta-fold / full-attn (token-exact) / NVFP4 `_scaled_mm` bf16-out copy-elision** — **5 RC-validated launch-cuts**, with **40/40 greedy outputs bit-identical to baseline** across structured/V9/GPQA/tool-call/long-form, inheriting **MMLU 84.40 / GPQA-Diamond 49.49**. All gated, default-safe, reversible.
>
> **🎯 On catching llama.cpp (framing corrected by the 6/3 evidence-lock).** Q4_K_M leads at **69.77** (~1.5×) on the same box. **We first assumed the cause was a ~2× bandwidth wall from the BF16 dequant-shadow — that a "read-4bit / zero-shadow" kernel would move it ~40 → ~140. The 6/3 evidence-lock (2 headless-CLI code traces + 4 Spark probes) disproved that premise:** read-4bit is already done (MoE experts run a "packed-4bit → register-dequant → bf16-GEMV" Triton kernel); the 60 GiB BF16 is **prefill-only** (decode drops it entirely and keeps running, TPS unchanged 42.4→43.7); routing attn to FP4 gives **no win / slower** (full-attn 0.999× / linear out_proj 0.775×); the reusable decode CUDA graph is **net-negative 0.75×**. → **decode is launch-bound, and Spark sm_121 (no FP4 MMA) is structurally capped ~45.** The gap to 69.77 is llama.cpp's hand-fused, low-dispatch, highly-mature ggml CUDA — a ground-up kernel rewrite + ultimately FP4-MMA silicon (R6000, retired), **not a Spark deliverable**. Bankable win this round: **decode-only shadow-drop → resident 87→27 GiB** (60 GiB freed for KV / long-context / batch). The cross-device kernel moat (same NVFP4 weights → native on FP4-MMA hardware) still holds, but it pays off only when that hardware is in hand. See [decode launch-overhead campaign](reports/qwen36_35b/DECODE_LAUNCH_OVERHEAD_CAMPAIGN_20260603.md).

> **🆕 2026-05-20 status change (⚠️ superseded by the 6/3 restart correction above) — at the time Lynn engine was downgraded to an "R&D exploration track"; it is now restarted as a parallel mainline benchmarking against llama.cpp.**
> The Lynn client adopts the **llama.cpp** ecosystem as the short-term default local inference backend (Mac Metal / Windows / Linux CUDA + Q4_K_M GGUF).
> Default ship model = **Qwen3.5-9B Q4_K_M-imatrix (5.3 GB)**, thinking-on excl_pf MMLU 90+ / GPQA 80+.
> Historical decision → [5/20 Release Notes](./RELEASE_NOTES_20260520.md). Current authority → [6/3 Restart Notes](./RELEASE_NOTES_20260603.md).
>
> All Lynn engine engineering artefacts kept on `main` (5 parallel CLIs + 7 bug fix trail + 178s repack + 2160-config autotune sweep — all real). Bar to return to mainline: **at same hardware × same model, Lynn engine speed ≥ llama.cpp AND quality has a moat llama.cpp can't provide** — both gates must pass.
>
> ⚠️ The 5/16 status content below is preserved as historical progress notes. **The current status authority is the 2026-06-03 restart section and notes.**

---

> **Custom inference engine for Lynn 27B-A3B NVFP4 on NVIDIA Blackwell.**
> This is a narrow, vertical engine for Lynn's own variable-pruned MoE + NVFP4 artifact. The goal is not to become another general serving framework; the goal is to make one model family fast, understandable, and production-ownable on R6000 / Spark-class Blackwell hardware.

[阅读中文版](README.md) · [Chinese Zhihu June series: Qwen 3.6 35B-A3B custom inference engine](https://zhuanlan.zhihu.com/p/2045562329396400486) · [Strategy](docs/STRATEGY.md) · [Architecture](docs/DESIGN.md) · **[🆕 6/3 Restart Notes](RELEASE_NOTES_20260603.md)** · [P1 dense projection PoC](reports/stage6/P1_DENSE_PROJECTION_POC_20260604.md) · [P1-A tiled sweep](reports/stage6/P1A_TILED_PROJECTION_SWEEP_20260604.md) · [P2 grouped MoE census](reports/stage6/P2_GROUPED_MOE_PREFILL_CENSUS_20260604.md) · [P2-G multi-layer MoE smoke](reports/stage6/P2G_MULTILAYER_MOE_SMOKE_20260604.md) · [P2-H selected-layer prefill smoke](reports/stage6/P2H_SELECTED_LAYER_PREFILL_SMOKE_20260604.md) · [P2-I selected-MoE expansion](reports/stage6/P2I_SELECTED_MOE_EXPANSION_SMOKE_20260604.md) · [P2-J linear-attn trace](reports/stage6/P2J_LINEAR_ATTN_PREFILL_TRACE_20260604.md) · [P2-KA gated-delta loop PoC](reports/stage6/P2KA_GATED_DELTA_NATIVE_LOOP_POC_20260604.md) · [P2-KB block kernel PoC](reports/stage6/P2KB_GATED_DELTA_BLOCK_KERNEL_POC_20260604.md) · [P2-L linear-attn integration](reports/stage6/P2L_LINEAR_ATTN_BLOCK_INTEGRATION_SMOKE_20260604.md) · [P2-M selected-layer smoke](reports/stage6/P2M_SELECTED_LAYER_BLOCK_LINEAR_SMOKE_20260604.md) · [5/20 historical notes](RELEASE_NOTES_20260520.md)

[![commits](https://img.shields.io/github/commit-activity/m/MerkyorLynn/lynn-engine)](https://github.com/MerkyorLynn/lynn-engine/commits/main)
[![license](https://img.shields.io/badge/license-TBD-orange)](.)

## Current status (2026-06-03)

**6/3 strategic correction (supersedes the 5/20 pivot)**: Lynn engine is **restarted as a parallel mainline, target = benchmark against (rival) llama.cpp** — no longer "downgraded to R&D / client defects to llama.cpp". The client still uses llama.cpp/GGUF short-term as a **pragmatic default backend**, but the engine advances in lockstep: same model + hardware vs llama.cpp, endgame = chewing through the fused 4-bit / zero-shadow kernel ourselves, approaching and even surpassing it on FP4-MMA silicon (R6000 class). The 6/3 measured read (Spark sm_121 has no FP4 MMA → decode structurally capped ~45; parity is a ggml-rewrite / FP4-MMA goal) is in the top 6/3 banner and [6/3 Restart Notes](RELEASE_NOTES_20260603.md); the 5/20 decision is historical background only.

### 5/16-5/20 Spark sm_121 W4A8 FP8 Phase 2 — what we learned

5 parallel CLIs + 23 commits driving Wave 2 W4A8 FP8 e2e. The final retry #4 exposed an **architectural signal**: Python overhead dominates decode time. With 30 MoE layers × 8 active experts × per-expert `_scaled_mm` = 240+ kernel launches/token, and CUDA graph capture disabled (because the active-expert dispatch contains host syncs), every launch pays the full Python/CUDA dispatch overhead. This is bug #7 — **not fixable by patching kernels**. It requires vectorised expert dispatch + CUTLASS grouped GEMM + a C++ service loop, which is months of engineering.

### 35B horizontal comparison (Spark sm_121 GB10 single-stream; baseline 2026-05-18, lynn-engine row updated 6/3)

| Path | Model size | Single-stream TPS | MMLU 500 | GPQA Diamond 198 | Note |
|---|---:|---:|---:|---:|---|
| Lynn-native NVFP4 W4A16 / lynn-engine | 23 GB | **38.96 → ~45** | 84.40% | 49.49% | 5/18 base → **6/3 launch-overhead campaign (5 cuts, RC-validated)** |
| **llama.cpp Q4_K_M-imatrix** | **20 GB** | **69.77** | 83.00% | **50.00%** | ~1.55× lynn-engine — 6/3 evidence-lock: Spark NVFP4 decode is structurally capped ~45 (bandwidth + dispatch levers both ruled out); parity needs a ggml-level rewrite / FP4-MMA silicon, **not a Spark deliverable** |
| SGLang BF16 official | 67 GB | 30.14 | 86.40% | 45.45% | reference |
| Lynn W4A8 FP8 (engineering exploration) | 35 GB | — | — | — | architecture incomplete, see RELEASE_NOTES |

**Key finding**: 35B quantised variants have essentially equal GPQA (BF16 / Q4_K_M-imatrix / Lynn NVFP4 all within ±1 pp around 49.5%). The "NVFP4 GPQA advantage" we previously expected does not hold on full samples.

### 9B Q4_K_M-imatrix default ship candidate (thinking-on excl_pf MMLU 90+ / GPQA 80+ / 5GB)

| Dimension | Lynn default ship 9B Q4_K_M-imatrix |
|---|---|
| Model file | **5.3 GB** (Q4_K_M-imatrix GGUF) |
| llama.cpp runtime | 79 MB (C++ binaries + .so) |
| **Total install footprint** | **5.4 GB** |
| MMLU 100 thinking-on excl_pf | **90.00%** (81/90) |
| GPQA Diamond 198 thinking-on excl_pf | **81.71%** |
| Spark sm_121 single-stream TPS | 36.80 |
| Spark sm_121 c=8 concurrent total TPS | **177.54** |
| Mac / Windows / Linux CUDA | All platforms native |

**The killer selling point for ordinary users = "unlimited local tokens"**: 9B running locally, no quota, no API key, no cross-border latency, agent running overnight with no metering.

### Lynn engine restarted mainline

- **Lynn engine is restarted as a parallel mainline**, aiming at same-model same-hardware comparison against llama.cpp instead of remaining demoted to R&D.
- **Banked:** 35B NVFP4 decode-only shadow release, resident 88→28 GiB, token-exact, 0.998× TPS.
- **P0.2 passed:** after release, only 4.72 GiB BF16 resident remains; the next cut should prioritize projections / embedding / lm_head, not router.
- **P1 single projection passed:** `linear_attn.in_proj_qkv` packed Triton matvec passed numeric/no-shadow/microbench gates, 1.186× vs BF16 shadow.
- **P1-A rejected:** both naive and tiled scalar batched bridges passed numeric/no-shadow gates, but both lose to BF16 tensor-core GEMM for M>1; do not wire into serving.
- **P2 census passed:** one-layer MoE confirms `stream_bf16` is the 20s no-reload proof bottleneck; the small-M verifier is memory-clean but slow.
- **P2-A..P2-M evidence:** single-expert gate/up loses to BF16; routed active path reaches M64=23.83ms/layer; P2-F engine opt-in mode M64=20.23ms/layer; P2-H/P2-I selected full prefill keeps numeric/no-shadow/speed passing; P2-J trace shows `chunk_gated_delta_with_state` is 71-76% of linear-attn prefill traced wall; P2-KA rejects token-by-token reuse of the decode recurrent kernel; P2-KB/P2-L wire block linear-attn; P2-M layers 0-3 selected full prefill keeps passing.
- **Next gate:** P2-N wider layer coverage + RC/server smoke; dense/MoE M>1 needs a native FP4-MMA/CUTLASS-style bridge to catch and surpass BF16.
- **Honest Spark sm_121 read:** decode is structurally capped around ~45; chasing 69.77 needs native runtime + fused kernels + eventually FP4-MMA silicon.
- **All Wave 2 commits remain on main, not reverted** — 5 parallel CLIs + 7 bug fix trail + 178s repack + 2160-config autotune sweep are real engineering artefacts.

---

> The 5/15-5/16 R6000 27B Lynn-native NVFP4 engineering record below is preserved as historical progress reference. **Current public data and engine restart framing are governed by this section (2026-06-03) and [RELEASE_NOTES_20260603](RELEASE_NOTES_20260603.md)**.

## 5/15-5/16 R6000 27B Lynn-native NVFP4 engineering record (historical)

5/15 R6000 sm_120a milestone on Lynn-native NVFP4: **107-108 TPS strict default** (P25 512 token = 107.43, P37 + 40/40 prompts all pass strict), three reporting layers:

- **107.23 TPS** serving replay
- **103.44 TPS** strict full path including final lm_head (the real "past 100 TPS" hard number)
- **99.86 TPS** strict full path no FP4 lm_head

Full P3 → P108 timeline (118.73 TPS R6000 ceiling → A100 W4A8 Recovery → SM120a FP4 contract → W4A8 hardware route) preserved in the section below.

## Current status (2026-05-16)

Lynn engine has moved from "Qwen 35B architecture bring-up" to an independent runtime for the **final Lynn 27B NVFP4 base**:

| Item | Status |
|---|---|
| **27B final BF16** | ✅ Recovery step5000 final merged; structural validation PASS; greedy sanity PASS |
| **27B Lynn-native NVFP4** | ✅ 20G artifact produced and transferred to R6000; manifest integrity PASS |
| **Independent loader** | ✅ No vLLM / SGLang / TRT-LLM / llama.cpp dependency; reads safetensors + Lynn quant manifest directly |
| **6-prompt coherent smoke** | ✅ Chinese explanation / Python / RoPE-ALiBi / English arithmetic / tool JSON / long-context prompts all pass |
| **Current R6000 strict full path** | ✅ **118.73 tok/s** with P23 packed NVFP4 MoE + native FP4 lm_head + active MoE retuning |
| **Serving replay ceiling** | ✅ **123.78 tok/s** 40-layer body graph, reproducible |
| **OpenAI server guard** | ✅ strict tool-call PASS,`<think>` loop fail-pattern guard PASS,decode reaches **~99-100 tok/s** once reusable block graphs are enabled |
| **P10 runner graph-slot gate** | ✅ 6 prompts × 3 prefixes = 18/18 strict PASS,runner graph slot 88.8-103.1 tok/s |
| **P11 packed-resident memory gate** | ✅ after prefill, releases **56.47 GiB** BF16 shadow; allocated memory **81.06 → 24.59 GiB** with exact greedy-id match |
| **P12 one-shot + graph-after-release gate** | ✅ OpenAI server first request releases **56.47 GiB**; graph slot after release reaches 79.5-83.8 tok/s with max_abs=0 |
| **P13 graph-slot generate wiring** | ✅ `generate()` opt-in path uses full-token graph slots;multi-prompt gate proves future-window unsafe,next state-refresh slot |
| **P14 state-refresh probe** | ✅ full mutable-state roundtrip costs only **0.79 ms**,far below graph capture at 60-105 ms |
| **P15 runtime config audit** | ✅ disables global `LYNN_PACKED_DECODE`; restores **103.48 strict / 107.23 replay**; disproves packed shared expert path |
| **P85-P88 SM120a FP4 contract** | ✅ blockscaled FP4 MMA / E2M1 shift / CuTe tile layout / real gate-up packed-code tile all pass |
| **P89 per-16 scale contract** | ✅ the current Lynn-native artifact can go directly through split16 per-16 scale native-FP4; single K32 scale folding is unsafe |
| **P90 real split16 gate/up kernel** | ✅ real expert116,8 gate + 8 up rows,K=2048 full dimension PASS,max_abs `2.38e-7` |
| **P91 row-tile sweep** | ✅ 8/16/32/64 rows all PASS;64-row median `0.0443ms`,rows/ms is **8.3×** above the 8-row tile |
| **P92 full gate/up expert** | ✅ full 512 gate + 512 up rows PASS,median `0.0502ms`,max_abs `4.77e-7` |
| **P93 top-k gate/up backend** | ✅ top-k=8 single-launch backend PASS,quantized-reference cosine `0.9999986`;slightly slower than Triton today,not promoted |
| **P94 active MoE composition** | ✅ P93 gate/up + packed down end-to-end PASS,cosine `0.9999986`;almost tied with Triton,not promoted yet |
| **P95 down backend sweep** | ✅ native_down_tile1 is **2.14×** faster than Triton down;all variants pass the contract |
| **P96 native-down composition** | ✅ numeric PASS,but `0.0830ms` vs Triton `0.0810ms`;not promoted because two-stage scheduling eats the down win |
| **P97 interval decomposition** | ✅ P93 gate/up + native_down_tile1 gives a full active-MoE **1.113×** speedup vs baseline,candidate contract PASS |
| **P98 split16 runtime gate** | ❌ backend build PASS,but graph-on capture FAIL;graph-off `new_ids_all_match=false`,median `0.80×`;not promoted |
| **P99 activation quant strategy** | ✅ no hidden runtime activation quantization in production; W4A4 moves into the A100 MTP/retrain + re-quant cycle |
| **P100 native-down runtime gate** | ❌ graph-off retest still has `new_ids_all_match=false`;median only `1.049×`;native-down-only stays off by default |
| **P101 graph-owned state gate** | ❌ P14-C/P35/P101 are all `pass=false`;replay TPS is attractive but the sequence drifts,so it is not a production graph path |
| **P102 mixed-MMA probe** | ❌ sm_120a has no BF16/FP16 × E2M1 MMA;E2M1×E2M1 controls PASS;155+ now requires a W4A4 / activation-aware artifact |
| **P103 W4A8 hardware route** | ✅ FP8(E4M3/E5M2) activation × E2M1 weight raw/blockscaled atoms all PASS;W4A8+MTP is now the near-term mainline |
| **P104/P105 W4A8 quality gates** | ✅/⚠️ active-MoE local gate AMBER and generation gate AMBER;do not promote the current artifact directly;move to A100 Recovery |
| **A100 BF16 transfer/inventory** | ✅ 1026/1026 shards,missing 0;resident BF16 load peaks at **59.10 GiB** on A100-80G |
| **P106 A100 W4A8 Recovery milestone** | ✅ expert-wise foldable alpha overlay reduces 40-layer real-prompt worst active-MoE drift from **3.67% to 1.79%**;the folded overlay artifact revalidates at **1.7836%** vs original BF16 |
| **Next target** | Fold the P106 alpha overlay into down weights,then rerun the 40-layer gate / P105 generation / V8-V9;implement a Lynn-owned MTP/NEXTN head prototype in parallel |

Current primary artifact:

```text
Lynn 27B variable-pruned Recovery step5000
├── BF16 final      ~60G  (reference / eval / fallback)
└── NVFP4 final     ~20G  (Lynn-native runtime artifact)
```

> Note: this NVFP4 artifact is **Lynn-native variable-expert NVFP4**. It is not the public compressed-tensors v8-RTN variant and not GGUF Q4_K_M. Generic frameworks generally cannot load this variable-pruned artifact directly; that is exactly why Lynn engine exists.

## Performance status

| Path | Latency / token | t/s | Status |
|---|---|---|---|
| Phase 2 brute-force | ~300 ms | 2-3 | historical baseline |
| Phase 3.1 incremental decode | ~200 ms | 5 | historical baseline |
| P5/P6 eager Triton path | ~30-33 ms | 30-33 | P6-K/N/O |
| **P6-S resident graph smoke** | **~15-16 ms** | **63-66** | ✅ 50 TPS target cleared |
| **P7 current serving env** | **~14.6-15.0 ms** | **66-68** | ✅ 6-prompt generate PASS |
| **P7/P8 CUDA graph ceiling** | **12.68 ms** | **78.8** | ✅ stable and reproducible |
| P8 torch.compile spike | 12.33 ms | 81.1 | signal only, not product path |
| **P10-M packed NVFP4 MoE** | **10.01 ms** | **99.86** | ✅ strict full path,BF16 lm_head |
| **P10-P native FP4 lm_head** | **9.67 ms** | **103.44** | ✅ strict full path,opt-in |
| **serving replay/body graph** | **9.33 ms** | **107.23** | ✅ 40-layer graph ceiling |
| **P10-U runner graph slot** | **9.69-11.26 ms** | **88.8-103.1** | ✅ 6 prompts × 3 prefixes strict PASS |
| **OpenAI server stable path(pre-P25)** | **~11.2 ms** | **88-89** | ✅ historical stable baseline |
| **P11 session-scoped packed resident** | — | — | ✅ BF16 shadow 81.06→24.59 GiB,exact decode-id match |
| **P12 one-shot + graph after release** | — | **79.5-83.8** | ✅ graph/eager exact match after 56.47 GiB release |
| **P13 graph-slot generate/window** | **12.6-12.8 ms replay / 60-105 ms capture** | **78-79 replay / 8-14 e2e** | ⚠️ current-position strict;future window multi-prompt FAIL |
| **P14 state refresh** | **0.79 ms roundtrip + 12.6 ms replay** | **~70-80 projected** | ✅ copy-cost green light,implementation pending |
| **P15 correct runtime config** | **9.66 ms strict / 9.33 ms replay** | **103.48 / 107.23** | ✅ `LYNN_PACKED_DECODE=0`,shared expert stays BF16 |
| **P16 active-MoE boundary** | **skip-active 5.75 ms replay / non-MoE 4.79 ms replay** | **173.8 / 208.8 upper bound** | 🔬 155 TPS requires a new grouped native-FP4 active expert kernel |
| **P17 Triton FP4 dot_scaled** | **raw gate/up shape 0.0125 ms; e8m0 neutral byte=127** | compute headroom ✅ | 🔬 layout/scale contract mapped,next step is per-16→group32 bridge |
| **P18 scale-contract decision** | **dot_scaled raw 0.018 ms vs scalar 0.050 ms** | speed ✅ / quality ❌ | 🔬 simple e8m0 bridge is not shippable; move to custom per-16 kernel |
| **P19 active block retune** | **8.66 ms strict / 8.32 ms replay** | **115.4 / 120.3** | ✅ quality-safe scheduling gain |
| **P20 unsorted router top-k** | **8.51 ms strict / 8.17 ms replay** | **117.6 / 122.4** | ✅ same expert set,MoE parity PASS |
| **P21 shared gate/up fusion** | **8.50 ms strict / 8.15 ms replay** | **117.7 / 122.7** | ✅ exact BF16 shared path,small gain |
| **P22 MoE warp retune** | **8.46 ms strict / 8.11 ms replay** | **118.3 / 123.3** | ✅ down kernel 8 warps |
| **P23 active MoE accounting** | **8.42 ms strict / 8.08 ms replay** | **118.7 / 123.8** | ✅ int32 expert-id cleanup;router/top-k branch ruled out |
| **P25 OpenAI server graph path** | **~9.95 ms decode / 0.65 s prefill** | **~99-100 decode / 87.7 wall @512 tok** | ✅ service path crosses 100 decode TPS |
| **P24/P26 Triton dead ends** | `tl.dot` gate/up / merged-topk gate/up | — | ❌ numerically OK but slower;not promoted |
| **P27 native CUDA extension smoke** | build/load/launch | add-one 0.0047 ms | ✅ R6000 sm_120 CUDA extension foundation passes |
| **P28 native gate/up contract** | CUDA scalar gate/up | 0.035 ms/layer | ✅ cosine≈1.0 contract pass;not speed-promoted |
| **P29 native down contract** | CUDA scalar down | 0.030 ms/layer | ✅ cosine=1.0,active-MoE contract complete |
| **P30 native active-MoE contract** | CUDA scalar gate/up + down | 0.064-0.068 ms/layer | ✅ full active routed expert path exact |
| **P31 native active-MoE runtime gate** | `LYNN_NATIVE_ACTIVE_MOE_BACKEND=cuda_scalar` | 1.48-1.59x MoE-function speedup | ✅ opt-in exact,default still off |
| **P32-P34 generate gate** | cuda_scalar full generate / graph / allowlist | 121.7 TPS but `!` loop; graph-off/allowlist still greedy drift | ❌ not promoted,fail-loud guard added |
| **P35 sorted-router graph slot** | `LYNN_ROUTER_TOPK_SORTED=1` + full-token graph slot | 12/12 strict PASS,97.7-111.5 TPS replay | ✅ graph-slot line regains parity |
| **P36 decode dispatch cleanup** | runner-fixed MoE/backend dispatch | 100.53 vs 100.55 TPS | ✅ exact,kept by default;not the 155 breakthrough |
| **P37 MoE block retune closed** | layer-28 profile + full generate gate | candidate 94.94 TPS and greedy drift | ❌ block-size line closed;move to native grouped FP4 |
| **P38 multi-layer MoE wall** | layers 2/8/14/20/28/36 | full MoE mean 0.193 ms/layer,active 0.112 ms/shared 0.060 ms | 🔬 no slow layer to harvest;target active expert kernel |
| **P39-P40 fast fixed MoE** | fixed R6000 best MoE config | layer-level 1.079x,generate exact,100.94 TPS median | ✅ small default speedup,not the 155 breakthrough |
| **P42 cuda_scalar retest** | full-attention-only allowlist | 0/3 greedy match,mean 82.49 TPS | ❌ scalar bridge is not a production shortcut |
| **P43 shared expert triage** | fused shared BF16 expert | 0.0609→0.0556 ms/layer | ✅ keep the small win,but too small for the 155 line |
| **P44 shortcut triage** | merged-topk / cross-expert `_scaled_mm` | 0.48x / 0.39x vs Triton | ❌ wrapper-level shortcuts closed;PyTorch `_scaled_mm` composition is not enough |
| **P45 native active-MoE ABI** | one-call CUDA contract | 0.0658 ms vs Triton 0.0583 ms,cosine≥0.99999988 | ✅ ABI foundation complete,not promoted |
| **P46 fused atomic probe** | one-kernel atomic accumulation | 0.1768 ms vs Triton 0.0592 ms | ❌ atomics are too slow and drift slightly;P47 moves to non-atomic grouped kernel |
| **P48-P50 tile-hidden down** | non-atomic CUDA down projection | isolated/decode-state 1.25-1.27x; full decode flips top-1 at step 5 | 🔬 kernel-level win confirmed,but tiny accumulation drift can flip greedy; not promoted |
| **P51 MoE budget ladder** | top-k limit / skip shared expert | best 124.39 TPS but output breaks; coherent top6 only +1.6% | ❌ fewer experts do not reach 155; quality fails first |
| **P52-A/B native FP4 sensitivity** | selected gate/up `_scaled_mm` + scale decomposition | active cosine 0.976; FP8 scale contract min cosine 0.972 | ❌ plain PyTorch `_scaled_mm` composition is not shippable; blocker is the per-16 scale contract |
| **P53 Triton retune review** | E2M1 decode simplification / scale hoist | LUT variant exact but avg 0.936x; scale-hoist JIT too heavy | 🔬 local signal only,no free 15 TPS;not promoted |
| **P54 vendor-layout feasibility** | e8m0/group32 scale search | best upper-bound inter cosine 0.9869-0.9918,all fail | ❌ direct ModelOpt-like scale contract is not enough;mainline stays per-16 grouped kernel |
| **P55 gate/up tile-inter** | CUDA scalar `tile_inter=2` | 1.09-1.14x vs Triton gate/up,max_abs=0 | 🔬 positive tile-shape signal for P56 grouped kernel |
| **P56 gate/up tile runtime** | `LYNN_NATIVE_GATEUP_BACKEND=cuda_tile_inter` | 114.43 TPS median(+15.4%) but `!` loop / greedy mismatch | ❌ runtime not promoted; keep tile-shape signal |
| **P58 graph-off retest** | `cuda_tile_inter` + block graph disabled | 28.43 TPS median,greedy mismatch remains | ❌ not a graph-only bug;scalar tile-inter is not a production bridge |
| **P59 dual NVFP4 dispatch** | metadata-only layout classifier | Lynn-native / vendor / BF16 / unknown packed FP4 fail-loud | ✅ one engine,two explicit NVFP4 artifact families |
| **P85-P87 SM120a FP4 tile contract** | CUTLASS/CuTe blockscaled FP4 MMA | E2M1 `<<2` shift + CuTe fragment layout | ✅ non-uniform synthetic tile `max_abs=0` |
| **P88 real gate/up packed-code tile** | real Lynn 27B layer28 expert116 | production activation FP4 codes + real packed weights,neutral scale | ✅ `max_abs=0`;current packed tensors can feed SM120 MMA |
| **P89 per-16 scale tile contract** | split16 neutral-scale MMA + explicit per-16 scale accumulation | `rel_l2=1.0e-7`,tolerance PASS;best K32 fold rel_l2 0.0227 | ✅ consume the current Lynn-native artifact first;do not wait for official re-quant |
| **P90 split16 gate/up kernel** | real expert116,8 gate + 8 up rows,K=2048 | median 0.0621ms,max_abs `2.38e-7`,rel_l2 `1.53e-7` | ✅ first real full-K native FP4 gate/up row tile PASS |
| **P91 split16 row-tile sweep** | row_count 8/16/32/64 | 64 rows median `0.0443ms`,rows/ms `2892`,all tolerance PASS | ✅ next design point is the 64-row tile |
| **P92 full gate/up expert** | 512 gate + 512 up rows,K=2048 | median `0.0502ms`,rows/ms `20401.7`,max_abs `4.77e-7` | ✅ full active expert gate/up sub-operator PASS |
| **P93 top-k gate/up backend** | top-k=8,one CUDA launch,output `[8,512]` | median `0.0602ms`,quantized-ref cosine `0.9999986`,rel_l2 `0.00167` | ✅ production-shaped gate/up backend contract PASS;slightly slower than Triton so not promoted |
| **P94 active MoE composition** | P93 gate/up + packed down weighted-sum | median `0.0823ms` vs Triton `0.0818ms`,quantized-ref cosine `0.9999986` | ✅ full active-MoE end-to-end contract PASS;speed now needs fused/non-atomic scheduling |
| **P95 down backend sweep** | fixed P93 inter + down variants | native_down_tile1 median `0.0243ms` vs Triton `0.0521ms` | ✅ down half has real 2.14× headroom;P96 should compose native gate/up + native down |
| **P96 native-down composition** | P93 gate/up + native_down_tile1 | median `0.0830ms` vs Triton active `0.0810ms`,quantized-ref cosine `0.9999986` | ✅ numeric PASS / ❌ no speed win;next work must reduce gate/up or fuse scheduling |
| **P97 interval decomposition** | CUDA-event gate/down intervals | best `0.0800ms` vs baseline `0.0891ms`,speedup `1.113×` | ✅ first full active-MoE composition speed win;next gate is full-generate parity |
| **P98 split16 runtime gate** | runtime `split16_fp4 + native_down_tile1` | graph-on capture fails;graph-off greedy mismatch and `0.80×` median TPS | ❌ P93/P97 quantized-activation contract is not production BF16-activation semantics |
| **P99 activation quant strategy** | after P98 redirect | production keeps BF16 activation semantics;activation quant moves to A100 MTP/requant path | ✅ closes the wrong shortcut early,keeps W4A4 as an explicit model contract |
| **P100 native-down runtime gate** | graph-off native-down-only retest | median `1.049×`,but `new_ids_all_match=false` on all prompts | ❌ not graph-only;native-down-only runtime replacement rejected |
| **P101 graph-owned state gate** | P14-C/P35/P101 authoritative state sequence | replay-only 75-85 TPS class,but `same_ids=false`,min cosine `0.6536` | ❌ graph-owned mutable state drifts;do not promote |
| **P102 mixed-MMA probe** | SM120a CuTe atom compile matrix | E2M1×E2M1 controls PASS;BF16/FP16×E2M1 raw/blockscaled all FAIL | ❌ no BF16-activation + FP4-weight shortcut;W4A4 model adaptation is required for 155+ |
| **P103 W4A8 probe** | SM120a FP8 activation × E2M1 weight compile matrix | E4M3/E5M2 × E2M1 raw/blockscaled all PASS | ✅ W4A8 is a viable hardware route;near-term mainline is W4A8+MTP |
| **P104 W4A8 sensitivity** | active-MoE fake-quant E4M3/per16 | gate/up clean;full-active near gate,max rel_l2 ~3.30% on the R6000 sample | ⚠️ W4A8 is trainable,not direct-promotable |
| **P105 W4A8 generation gate** | 6-prompt generation fake-quant | 64-token gate: gateup 4/6 exact,full 5/6 exact,late/local drift | ⚠️ A100 Recovery required before runtime promotion |
| **P106 A100 Recovery** | expert-wise foldable intermediate alpha overlay | real-prompt 40-layer worst active-MoE rel_l2 **3.67% → 1.79%**;folded artifact W4A8-vs-original BF16 **1.7836%** | ✅ folded overlay artifact GREEN;W4A8 artifact contract,not BF16 fallback |
| Long target | <5 ms | >200 | native FP4 / larger fused blocks |

Current best R6000 environment:

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LYNN_PREFILL_WARMUP=1
export LYNN_LINEAR_ATTN_RECURRENT_BACKEND=triton_fused_prepare
export LYNN_LINEAR_ATTN_RECURRENT_INPLACE=1
export LYNN_MOE_IMPL=packed_nvfp4
export LYNN_MOE_GATE_BLOCK_INTER=8
export LYNN_MOE_GATE_BLOCK_HIDDEN=256
export LYNN_MOE_DOWN_BLOCK_HIDDEN=8
export LYNN_MOE_DOWN_BLOCK_INTER=512
export LYNN_MOE_GATE_NUM_WARPS=4
export LYNN_MOE_DOWN_NUM_WARPS=8
export LYNN_QK_NORM_ROPE_BACKEND=triton_pair
export LYNN_RMSNORM_GATED_BACKEND=triton
export LYNN_LINEAR_ATTN_INPROJ_FUSED_NATIVE_FP4=1
export LYNN_NATIVE_FP4_LM_HEAD=1
export LYNN_LINEAR_STATE_UPDATE=inplace
export LYNN_LINEAR_BLOCK_GRAPH=1
export LYNN_LINEAR_BLOCK_GRAPH_REUSE=1
export LYNN_LINEAR_BLOCK_GRAPH_PREWARM=1
export LYNN_PACKED_DECODE=0
export LYNN_PACKED_DECODE_PREPARE_NATIVE=0
export LYNN_PACKED_SHARED_EXPERT=0
```

Measured final step5000 NVFP4:

```text
strict full path:      118.73 tok/s  (P23 int32 expert-id cleanup + P22/P21/P20/P19)
serving replay/body:   123.78 tok/s  (40-layer graph ceiling)
OpenAI server decode:   ~99-100 tok/s (P25 reusable block graph env,512-token wall TPS 87.7)
BF16 lm_head path:     99.86 tok/s
quality smoke:         6/6 coherent + strict tool-call + no-think loop guard PASS
```

The native FP4 lm_head path is currently a deterministic/greedy opt-in optimization:6/6 prompt top-1 match,minimum top-20 overlap 15/20,and minimum logits cosine 0.9924. Sampling-heavy production traffic keeps the BF16 lm_head fallback until a larger parity gate is complete.

CUDA graph note:P10-S records the cross-token drift boundary for full-token
graph families, so future-window full-token graphs still do not enter the
default production path. P25 uses the more conservative reusable linear-block
graph server path and reaches **~99-100 decode TPS** through the
OpenAI-compatible wrapper. The remaining 155 TPS gap is the active expert
kernel, not the HTTP wrapper. See
[`docs/LYNN_ENGINE_P10S_GRAPH_BOUNDARY_20260515.md`](docs/LYNN_ENGINE_P10S_GRAPH_BOUNDARY_20260515.md)
and [`docs/LYNN_ENGINE_P25_SERVER_100TPS_20260516.md`](docs/LYNN_ENGINE_P25_SERVER_100TPS_20260516.md).

P15 config note: do not treat `LYNN_PACKED_DECODE=1` as "more packed means
faster". On R6000 it drops the full graph path from **103.48 tok/s** to
**88.15 tok/s**, because small Q/K/V/O decode projections are slower on the
generic packed native path. The current correct profile packs MoE active
experts, uses native fused linear-attention in-projection, and optionally uses
native FP4 lm_head, but keeps **generic packed decode disabled**. The shared
expert also stays BF16 because packed scalar/native shared paths are slower.
See [`docs/LYNN_ENGINE_P15_RUNTIME_CONFIG_20260516.md`](docs/LYNN_ENGINE_P15_RUNTIME_CONFIG_20260516.md).

P16 155 TPS note: profiling shows the non-MoE path can replay at
**208.8 tok/s**, and skipping active routed experts reaches **173.8 tok/s**.
However, top-k approximation and block-size sweeps do not safely reach 155.
The conclusion is that 155 TPS is not another environment toggle; it requires a
new **grouped native-FP4 active expert kernel**. See
[`docs/LYNN_ENGINE_P16_155TPS_ACTIVE_MOE_20260516.md`](docs/LYNN_ENGINE_P16_155TPS_ACTIVE_MOE_20260516.md).

P17 note: Triton 3.6 `tl.dot_scaled(e2m1)` now passes a raw packed-FP4 layout
probe on R6000. The real gate/up shape `1x2048 @ 2048x8192` takes only
**0.0125 ms**. A follow-up scale probe confirms `rhs_scale=[N,K//32]` and a
synthetic two-sided neutral e8m0 byte of **127**, proving that FP4 tensor-core
compute is not the bottleneck. The next blocker is wiring Lynn's per-16/e4m3
scale contract into e8m0/group32 grouped native dot. See
[`docs/LYNN_ENGINE_P17_TRITON_DOT_SCALED_20260516.md`](docs/LYNN_ENGINE_P17_TRITON_DOT_SCALED_20260516.md).

P18 note: three scale-bridge variants were tested. `dot_scaled` raw gate/up
reaches **0.018 ms** versus the current scalar bridge at about **0.050 ms**, but
quality does not pass: per-16→group32 folding reaches only **0.894** best inter
cosine, BF16→e8m0/group32 re-quant reaches **0.980**, and padded per-16 reaches
**0.936**. The conclusion: 155 TPS still has hardware headroom, but we should
not ship a simple e8m0 bridge that sacrifices quality. The next route is a
**custom per-16 grouped native-FP4 kernel / CUTLASS path** or a stronger
engine-native quant artifact. See
[`docs/LYNN_ENGINE_P18_SCALE_CONTRACT_DECISION_20260516.md`](docs/LYNN_ENGINE_P18_SCALE_CONTRACT_DECISION_20260516.md).

P19 note: without changing the numerical path, active MoE kernel block retuning
moves the R6000 full graph from **103.40/107.13 TPS** to **115.41/120.25 TPS**.
The recommended config is now the default: `gate_hidden=256,down_inter=512`,
with env overrides retained for device-specific tuning. See
[`docs/LYNN_ENGINE_P19_ACTIVE_BLOCK_RETUNE_20260516.md`](docs/LYNN_ENGINE_P19_ACTIVE_BLOCK_RETUNE_20260516.md).

P20 note: router `topk(sorted=False)` has been verified to keep the same expert
set and paired weights. Representative-layer MoE parity has `max_abs=0`, and
the full graph moves to **117.55/122.43 TPS**. See
[`docs/LYNN_ENGINE_P20_ROUTER_TOPK_UNSORTED_20260516.md`](docs/LYNN_ENGINE_P20_ROUTER_TOPK_UNSORTED_20260516.md).

P21 note: the shared expert stays BF16, but gate/up are fused into one BF16
GEMM. Representative-layer parity has `max_abs=0`; the full graph nudges up to
**117.71/122.71 TPS**. See
[`docs/LYNN_ENGINE_P21_SHARED_GATEUP_FUSION_20260516.md`](docs/LYNN_ENGINE_P21_SHARED_GATEUP_FUSION_20260516.md).

P22 note: MoE active kernels now expose `num_warps`. R6000's best profile keeps
gate/up at 4 warps and moves down projection to 8 warps, nudging the full graph
to **118.25/123.25 TPS**. See
[`docs/LYNN_ENGINE_P22_MOE_WARP_RETUNE_20260516.md`](docs/LYNN_ENGINE_P22_MOE_WARP_RETUNE_20260516.md).

P23 note: full-layer active MoE accounting shows no pathological slow layer.
The 40 active routed expert calls are nearly flat at **~0.069 ms/layer**,
router is about **0.049 ms/layer**, and shared expert is about
**0.057 ms/layer**. 1D top-k, manual softmax, and a Triton fused
top-k+softmax probe were ruled out. The only promoted safe cleanup casts
expert ids to int32 once after router top-k, avoiding duplicate gate/up and
down casts, and moves the full graph to **118.73/123.78 TPS**. See
[`docs/LYNN_ENGINE_P23_ACTIVE_MOE_ACCOUNTING_20260516.md`](docs/LYNN_ENGINE_P23_ACTIVE_MOE_ACCOUNTING_20260516.md).

P25 note: once `LYNN_LINEAR_BLOCK_GRAPH=1 / REUSE=1 / PREWARM=1` are enabled,
the OpenAI-compatible server no longer stays on the older 88-89 TPS path. It
stabilizes at **~99-100 decode tok/s**, with a 512-token request reaching
**87.7 wall tok/s**. This proves the service wrapper is not the main remaining
155 blocker; the remaining gap is still the active expert kernel. See
[`docs/LYNN_ENGINE_P25_SERVER_100TPS_20260516.md`](docs/LYNN_ENGINE_P25_SERVER_100TPS_20260516.md).

P24/P26 note: two Triton-only shortcuts are now closed. P24's per-16
dequant→`tl.dot` bridge is numerically good but reaches only **0.0807 ms**,
slower than the production scalar gate/up at **0.0335 ms**. P26's merged-topk
scheduling variant has cosine=1.0, but is **2.06-2.09x slower** across four
representative layers. The conclusion is sharper: 155 TPS needs a
CUDA/CUTLASS-level custom per-16 grouped native-FP4 expert kernel, not more
Triton program-grid rearrangement. See
[`docs/LYNN_ENGINE_P24_TL_DOT_NEGATIVE_20260516.md`](docs/LYNN_ENGINE_P24_TL_DOT_NEGATIVE_20260516.md)
and [`docs/LYNN_ENGINE_P26_MERGED_TOPK_NEGATIVE_20260516.md`](docs/LYNN_ENGINE_P26_MERGED_TOPK_NEGATIVE_20260516.md).

P27 note: the R6000 native CUDA extension build/load/launch gate now passes.
The measured stack is PyTorch **2.10.0+cu128**, CUDA toolkit **12.8**, and
`sm_120`. `torch.utils.cpp_extension.load` successfully compiles and imports a
Lynn-owned CUDA extension; the 1M-float `add_one` smoke kernel reports
`max_abs=0` and averages **0.0047 ms**. This is not a TPS improvement by
itself, but it removes the build-plumbing risk for the next custom per-16
grouped native-FP4 active expert kernel. See
[`docs/LYNN_ENGINE_P27_CUDA_EXTENSION_SMOKE_20260516.md`](docs/LYNN_ENGINE_P27_CUDA_EXTENSION_SMOKE_20260516.md).

P28 note: the first real active-MoE CUDA extension contract now passes.
`gate_up_silu_scalar` consumes the final Lynn 27B grouped packed NVFP4 tensors,
per-16 scales, and top-k expert ids directly, then emits the `[top_k, 512]`
intermediate. Across four representative layers it matches the Triton reference
with `cosine≈1.0 / max_abs≈0`. It is slightly slower (**0.035 ms** vs
**0.034 ms**), so this is a contract pass, not a speed promotion. The next step
is to keep the same C++/CUDA entrypoint and replace the scalar inner loop with
true grouped native-FP4 math. See
[`docs/LYNN_ENGINE_P28_NATIVE_GATEUP_CONTRACT_20260516.md`](docs/LYNN_ENGINE_P28_NATIVE_GATEUP_CONTRACT_20260516.md).

P29 note: the second half of active MoE now has a native CUDA extension contract
too. `down_weighted_sum_scalar` consumes the `[top_k, 512]` intermediate,
routing weights, and down packed/scale/global tensors, then emits `[2048]`.
Across four representative layers it matches the Triton reference with
`cosine=1.0`. It is slower (**0.030 ms** vs **0.026-0.027 ms**), so it is not a
speed promotion, but P28+P29 now provide the complete native active-MoE data
contract. See
[`docs/LYNN_ENGINE_P29_NATIVE_DOWN_CONTRACT_20260516.md`](docs/LYNN_ENGINE_P29_NATIVE_DOWN_CONTRACT_20260516.md).

P30 note: the complete active routed expert path can now be composed from
Lynn-owned CUDA extension calls: CUDA gate/up scalar followed by CUDA down
weighted-sum scalar. Across four representative layers it matches the
production Triton active path with `cosine=1.0 / max_abs≈0`. It is slower
(**0.064-0.068 ms/layer** vs **0.058 ms/layer**), so it is not a runtime
default. But P27-P30 now cover native extension build, gate/up, down, and full
active-MoE data-contract correctness. The next TPS-relevant work is replacing
the scalar inner loops with grouped native-FP4 tensor-core math. See
[`docs/LYNN_ENGINE_P30_NATIVE_ACTIVE_MOE_CONTRACT_20260516.md`](docs/LYNN_ENGINE_P30_NATIVE_ACTIVE_MOE_CONTRACT_20260516.md).

P31 note: after wiring the P30 native active-MoE scalar path into the production
`moe_forward_decode_packed_nvfp4` function, the native backend is **1.48-1.59x
faster** than the default Triton backend at the function boundary, while
remaining exact across four representative layers. This likely removes part of
the Python/Triton wrapper overhead. It is still opt-in and off by default;
promotion requires full-token decode parity, full graph/server TPS measurement,
and tool-call/no-think guard coverage. See
[`docs/LYNN_ENGINE_P31_NATIVE_ACTIVE_MOE_RUNTIME_GATE_20260516.md`](docs/LYNN_ENGINE_P31_NATIVE_ACTIVE_MOE_RUNTIME_GATE_20260516.md).

P32-P34 note: the full-generate gate explicitly rejects promoting
`cuda_scalar` as-is. Full-layer `cuda_scalar` with reusable linear-block graphs
reaches **121.7 tok/s**, but falls into a token-0 / `!` loop from the second
decode token. Disabling graphs restores coherent text but still fails greedy-id
parity. A full-attention-only allowlist avoids the graph failure but still
shows top-1 drift. The runner now fail-louds instead of silently serving the
unsafe graph combination. See
[`docs/LYNN_ENGINE_P32_P34_NATIVE_ACTIVE_MOE_GENERATE_NEGATIVE_20260516.md`](docs/LYNN_ENGINE_P32_P34_NATIVE_ACTIVE_MOE_GENERATE_NEGATIVE_20260516.md).

P35 note: the graph-slot line is not dead, but it needs a stricter router-order
contract than the eager/full-graph path. With `LYNN_ROUTER_TOPK_SORTED=1`, the
full-token graph-slot gate passes **12/12** checks across four prompt types and
three prefix lengths, with `max_abs=0` on every row and **97.7-111.5 tok/s**
replay. The current stable serving path may keep P20 unsorted top-k, but future
graph-slot serving modes should force the sorted-router contract. See
[`docs/LYNN_ENGINE_P35_SORTED_ROUTER_GRAPH_SLOT_20260516.md`](docs/LYNN_ENGINE_P35_SORTED_ROUTER_GRAPH_SLOT_20260516.md).

P36 note: decode-time MoE implementation selection, linear-attention recurrent
backend, and state-update policy are now fixed on the runner instead of being
resolved inside every layer/token call. The R6000 gate reports exact greedy-id
match, with legacy median **100.55 TPS** and fast-dispatch median
**100.53 TPS**. This is a safe engineering cleanup and remains enabled by
default, but it proves the remaining 100 -> 155 TPS gap is not simple Python
dispatch overhead. See
[`docs/LYNN_ENGINE_P36_DECODE_FAST_DISPATCH_20260516.md`](docs/LYNN_ENGINE_P36_DECODE_FAST_DISPATCH_20260516.md).

P37 note: the layer-28 profile splits current MoE latency into router
**0.036 ms**, active packed NVFP4 experts **0.107 ms**, and shared BF16 expert
**0.060 ms**. The isolated block sweep suggested a tiny `down_block_hidden=16`
win, but the full generate gate fell to **94.94 TPS** and drifted on all three
prompts. The conclusion is that more Triton block-size retuning is not the
155 TPS path; the next step is a grouped native-FP4 active expert kernel or a
strict-parity graph-owned serving path. See
[`docs/LYNN_ENGINE_P37_MOE_BLOCK_RETUNE_CLOSED_20260516.md`](docs/LYNN_ENGINE_P37_MOE_BLOCK_RETUNE_CLOSED_20260516.md).

P38 note: P37's single-layer profile is now extended to six sampled layers
(2/8/14/20/28/36). The picture is uniform: full MoE averages
**0.193 ms/layer**, split into router **0.037 ms**, active routed packed NVFP4
experts **0.112 ms**, and shared BF16 expert **0.060 ms**. There is no isolated
slow layer to harvest; the next 155 TPS work should target the active expert
grouped native-FP4 kernel first, with the shared expert second. See
[`docs/LYNN_ENGINE_P38_MOE_MULTILAYER_PROFILE_20260516.md`](docs/LYNN_ENGINE_P38_MOE_MULTILAYER_PROFILE_20260516.md).

P39-P40 note: splitting active MoE shows gate/up at **0.033 ms** and down at
**0.025 ms**, with exact split-vs-combined output. The current R6000 best MoE
config is then promoted into the fixed `LYNN_MOE_FAST_FIXED` path. The
layer-level candidate is **1.079x** faster, the full-generate gate keeps exact
greedy IDs, and the post-promotion default path verifies at **100.79 TPS**
median. This is a safe
small knife, not the 155 TPS breakthrough; the next major step remains a
fused/grouped native-FP4 active expert kernel. See
[`docs/LYNN_ENGINE_P39_P40_FAST_FIXED_MOE_20260516.md`](docs/LYNN_ENGINE_P39_P40_FAST_FIXED_MOE_20260516.md).

P42 note: `cuda_scalar` native active-MoE was retested with a
full-attention-only allowlist to avoid linear-block graph capture. It still
fails full-generate parity: **0/3 greedy matches**, with one prompt dropping to
**48.50 TPS**. Keep `cuda_scalar` as a native-extension contract/diagnostic
backend only; it is not a production speed shortcut. See
[`docs/LYNN_ENGINE_P42_CUDA_SCALAR_RETEST_NEGATIVE_20260516.md`](docs/LYNN_ENGINE_P42_CUDA_SCALAR_RETEST_NEGATIVE_20260516.md).

P43-P44 note: the 155 TPS route is now narrower. P43 shows the fused shared
expert path only improves **0.0609 ms → 0.0556 ms/layer**, too small to be the
main line. P44 closes two more shortcuts: merged-topk scheduling is only about
**0.48x** of the production gate/up path, and cross-expert `torch._scaled_mm`
composition reaches only about **0.39x** mm-only / **0.15x** with activation
quantization while drifting to min cosine **0.97698**. Conclusion: 155 TPS
requires a real grouped/block-diagonal FP4 active expert kernel, not another
wrapper-level rearrangement. See
[`docs/LYNN_ENGINE_P43_P44_ACTIVE_MOE_NATIVE_FP4_TRIAGE_20260516.md`](docs/LYNN_ENGINE_P43_P44_ACTIVE_MOE_NATIVE_FP4_TRIAGE_20260516.md).

P45-P46 note: P45 freezes the native active-MoE one-call CUDA ABI:
`active_moe(hidden,expert_ids,routing_weights,gate_up*,down*) -> out[2048]`.
It matches the two-call scalar reference with min cosine **0.99999988**, but is
still slower than Triton (**0.0658 ms vs 0.0583 ms**), so it is an ABI
foundation only. P46 tests a one-kernel fused-atomic bridge; it is much slower
(**0.1768 ms** vs Triton **0.0592 ms**) and has small accumulation drift.
Conclusion: P47 must be a non-atomic grouped/block-diagonal kernel. See
[`docs/LYNN_ENGINE_P45_NATIVE_ACTIVE_MOE_CONTRACT_20260516.md`](docs/LYNN_ENGINE_P45_NATIVE_ACTIVE_MOE_CONTRACT_20260516.md)
and [`docs/LYNN_ENGINE_P46_FUSED_ATOMIC_NEGATIVE_20260516.md`](docs/LYNN_ENGINE_P46_FUSED_ATOMIC_NEGATIVE_20260516.md).

P48-P50 note: the first non-atomic route targets the down-projection subsegment
with a tile-hidden CUDA kernel. Isolated six-layer microbenchmarks improve
Triton down from **0.02586 ms** to **0.02067 ms** (**1.25x**), and P49 confirms
the same signal on true decode-state MoE inputs (**1.27x**, max rel_l2
**8.74e-05**). But P50 shows complete graph-free decoding still diverges: the
first visible drift appears at step 1 / layer 27 and top-1 flips by step 5. So
P48 is a real kernel-level signal, but tiny accumulation drift can affect greedy
decode; it is not a default runtime path. See
[`docs/LYNN_ENGINE_P48_DOWN_TILE_NONATOMIC_20260516.md`](docs/LYNN_ENGINE_P48_DOWN_TILE_NONATOMIC_20260516.md).

P51 note: to close the "just compute fewer experts" shortcut, P51 adds opt-in
`LYNN_MOE_TOPK_LIMIT` / `LYNN_MOE_SKIP_SHARED` budget profiles. The result is
clear: top6+shared stays coherent but only gains **1.6%**; top4+shared gains
**4.5%** but already shows `<think>` pollution; skipping the shared expert can
reach **124.39 TPS** but breaks output quality. Conclusion: 155 TPS requires a
grouped native-FP4 active expert FFN or an exact graph-owned route, not expert
budget trimming. See
[`docs/LYNN_ENGINE_P51_ACTIVE_MOE_BUDGET_LADDER_20260516.md`](docs/LYNN_ENGINE_P51_ACTIVE_MOE_BUDGET_LADDER_20260516.md).

P52 note: the route now splits into two serious tracks. Track A keeps the
current Triton active-MoE math and attacks orchestration / graph overhead for
exact serving gains. Track B builds the real grouped native-FP4 active expert
FFN with CUTLASS/CuTe or custom CUDA, expressing block-diagonal selected experts
directly instead of `_scaled_mm` wrappers or top-k approximations. P52-A/B
further prove that replacing only selected gate/up with `torch._scaled_mm` is
slower and looser: active cosine bottoms at **0.976**, and the dominant loss
appears when Lynn's FP32 per-16 weight scales are compressed into the FP8
`scale_b` contract(min cosine **0.972**). The issue is the scale contract, not
mysterious tensor-core accumulation. MTP/spec decode is not the immediate main
line; it is a later serving multiplier, not the base kernel fix. See
[`docs/LYNN_ENGINE_P52_GROUPED_NATIVE_FP4_CONTRACT_20260516.md`](docs/LYNN_ENGINE_P52_GROUPED_NATIVE_FP4_CONTRACT_20260516.md).

P53 note: an external review proposed two "free" Triton improvements: hoisting
per-16 scale loads and simplifying the E2M1 decode expression. P53 tests both
as opt-in probes. The naive scale-hoist rewrite is too heavy to JIT/execute as
a practical probe. The lightweight E2M1 expression is numerically exact and
wins **6-8%** on three representative layers, but loses badly on layer36
(**0.536x**) and averages **0.936x**. Keep it as an allowlist/variant research
line, not a default replacement. See
[`docs/LYNN_ENGINE_P53_TRITON_RETUNE_REVIEW_20260516.md`](docs/LYNN_ENGINE_P53_TRITON_RETUNE_REVIEW_20260516.md).

P54 note: to answer whether Lynn can simply produce an NVIDIA/ModelOpt-friendly
e8m0/group32 NVFP4 side artifact and use vendor-style kernels, P54 adds an
offline scale-search upper-bound probe. `search_recon_mse` represents a
deployable-style static reconstruction search; `search_dot_upper_bound` is an
activation-aware optimistic dot-preservation upper bound. Across four
representative layers, the best upper-bound inter cosine is still only
**0.9869-0.9918**, below the 0.995 safety gate. The conclusion is that the
public NVIDIA NVFP4/Blackwell path validates the destination, but Lynn's
current per-16 FP32-scale artifact cannot be converted into that vendor scale
contract by simple offline exponent search. The mainline stays
**Lynn-native per-16 grouped active expert kernels**. See
[`docs/LYNN_ENGINE_P54_VENDOR_LAYOUT_FEASIBILITY_20260516.md`](docs/LYNN_ENGINE_P54_VENDOR_LAYOUT_FEASIBILITY_20260516.md).

Quantization v2 note: imatrix, layer strategy, and activation-aware scale
search are valuable for Lynn, but they are a **quantization-quality track**,
not a replacement for the current 100→155 TPS runtime blocker. The near-term
target is public GGUF/Q4_K_M quality: lower PPL/KLD and stronger
V8/V9/tool/long-context retention. The later target is a BF16-derived
vendor-compatible NVFP4 v2 artifact, not a direct conversion of the current
Lynn-native per-16 artifact. See
[`docs/LYNN_ENGINE_QUANTIZATION_V2_ROADMAP_20260516.md`](docs/LYNN_ENGINE_QUANTIZATION_V2_ROADMAP_20260516.md).

P55 note: a new gate/up-side `tile_inter` CUDA scalar probe lets one block
compute multiple intermediate rows while reusing the hidden-vector load. Across
four representative layers, `tile_inter=2` is exact against the Triton gate/up
reference (`max_abs=0`) and is **1.09-1.14x** faster locally; `tile_inter=4/8`
are slower. This is a positive tile-shape signal for the P56 grouped per-16
kernel, but it is still a local scalar probe and is not promoted to the default
runtime by itself. See
[`docs/LYNN_ENGINE_P55_GATEUP_TILE_INTER_20260516.md`](docs/LYNN_ENGINE_P55_GATEUP_TILE_INTER_20260516.md).

P56 note: wiring P55 `tile_inter=2` into an opt-in runtime gate raises median
decode TPS from about **99.15** to **114.43**(+15.4%), but full-generate greedy
IDs do not match and the candidate falls into a `!` loop. The decision is to
reject runtime promotion while preserving the positive tile-shape signal:
`tile_inter=2` should inform the P57/P58 real grouped per-16 native-FP4 kernel,
not become a scalar production shortcut. See
[`docs/LYNN_ENGINE_P56_GATEUP_TILE_RUNTIME_REJECTED_20260516.md`](docs/LYNN_ENGINE_P56_GATEUP_TILE_RUNTIME_REJECTED_20260516.md).

P57/vendor-route note: the official ModelOpt route is valid, but it needs a
separate **vendor-friendly NVFP4 v2** artifact quantized from BF16 final rather
than a post-hoc conversion of the current Lynn-native per-16 artifact. The
current R6000 env lacks `modelopt/llmcompressor`; `compressed_tensors` only has
a ModelOpt-NVFP4 converter for already-quantized artifacts. The 27B checkpoint
is also physical variable-expert and `hf_vanilla_compatible=false`, so the
vendor route needs padding/masking back to fixed 256 experts or vendor-side
variable-expert support. See
[`docs/LYNN_ENGINE_P57_VENDOR_ROUTE_INVENTORY_20260516.md`](docs/LYNN_ENGINE_P57_VENDOR_ROUTE_INVENTORY_20260516.md).

Dual-artifact policy: Lynn may keep both **Lynn-native NVFP4 final** and
**vendor-friendly NVFP4 v2**. The former remains the Lynn engine / per-16
grouped-kernel mainline; the latter is a BF16-derived compatibility artifact
for official ecosystems. They must use different directories, manifests, and
validation gates, and must never overwrite each other. See
[`docs/LYNN_ENGINE_DUAL_NVFP4_ARTIFACT_POLICY_20260516.md`](docs/LYNN_ENGINE_DUAL_NVFP4_ARTIFACT_POLICY_20260516.md)
and
[`docs/LYNN_ENGINE_VENDOR_VS_LYNN_NATIVE_TRADEOFF_20260516.md`](docs/LYNN_ENGINE_VENDOR_VS_LYNN_NATIVE_TRADEOFF_20260516.md).

P58 note: to rule out the possibility that P56 only failed because the CUDA
extension interacted badly with linear-block graph capture, P58 reruns
`cuda_tile_inter` with `LYNN_LINEAR_BLOCK_GRAPH*` disabled. The candidate drops
to **28.43 TPS** median and greedy mismatch remains. Conclusion: this is not a
graph-only bug; scalar tile-inter accumulation/scheduling is itself unsafe for
full greedy decode. Keep the shape hint, close the runtime bridge. See
[`docs/LYNN_ENGINE_P58_GATEUP_TILE_GRAPH_OFF_REJECTED_20260516.md`](docs/LYNN_ENGINE_P58_GATEUP_TILE_GRAPH_OFF_REJECTED_20260516.md).

P59 note: adds a metadata-only NVFP4 layout classifier before tensor loading.
It distinguishes `lynn_native_per16_variable`, `compressed_tensors_nvfp4`,
`modelopt_nvfp4`, `bf16_or_unquantized`, and unknown packed FP4. The goal is one
Lynn engine with two explicit NVFP4 artifact families: current Lynn-native
per-16 variable experts and a future vendor-friendly NVFP4 v2. Cross-loading
must fail loudly. See
[`docs/LYNN_ENGINE_P59_DUAL_NVFP4_LAYOUT_DISPATCH_20260516.md`](docs/LYNN_ENGINE_P59_DUAL_NVFP4_LAYOUT_DISPATCH_20260516.md).

P85-P89 note: the official/ModelOpt route remains open, but the runtime
mainline no longer waits for re-quantization. P85-P87 prove the SM120a
blockscaled FP4 MMA, E2M1 `<<2` shift, and CuTe fragment layout. P88 feeds a
real Lynn 27B gate/up packed-code tile into that contract with `max_abs=0`. P89
then proves that the current Lynn-native per-16 scale artifact can be consumed
directly through **split16 neutral-scale MMA plus explicit per-group scale
accumulation**, with only `rel_l2=1.0e-7` FP32-order tolerance. Folding two K16
scales into one K32 scale drifts too much(best rel_l2 is still 0.0227), so the
official/vendor-friendly NVFP4 v2 artifact should move with the MTP/retrain +
re-quant cycle instead of blocking P90. See
[`docs/LYNN_ENGINE_P85_BLOCKSCALED_FP4_MMA_CONTRACT_20260516.md`](docs/LYNN_ENGINE_P85_BLOCKSCALED_FP4_MMA_CONTRACT_20260516.md),
[`docs/LYNN_ENGINE_P86_FP4_SHIFT_CONTRACT_20260516.md`](docs/LYNN_ENGINE_P86_FP4_SHIFT_CONTRACT_20260516.md),
[`docs/LYNN_ENGINE_P87_FP4_LAYOUT_TILE_CONTRACT_20260516.md`](docs/LYNN_ENGINE_P87_FP4_LAYOUT_TILE_CONTRACT_20260516.md),
[`docs/LYNN_ENGINE_P88_REAL_GATEUP_TILE_CONTRACT_20260516.md`](docs/LYNN_ENGINE_P88_REAL_GATEUP_TILE_CONTRACT_20260516.md), and
[`docs/LYNN_ENGINE_P89_PER16_SCALE_TILE_CONTRACT_20260516.md`](docs/LYNN_ENGINE_P89_PER16_SCALE_TILE_CONTRACT_20260516.md).

P90 note: the first real split16 gate/up kernel moves P89 from tile proof to a
full-K row tile. It uses real Lynn 27B layer28 expert116, real activation, and
real packed gate/up weights, and computes 8 gate rows plus 8 up rows over
K=2048. Result: `max_abs=2.38e-7`, `rel_l2=1.53e-7`, tolerance PASS. Decision:
the current Lynn-native artifact can feed the native FP4 gate/up kernel
directly; do not wait for official/vendor re-quantization before P91. See
[`docs/LYNN_ENGINE_P90_SPLIT16_GATEUP_KERNEL_20260516.md`](docs/LYNN_ENGINE_P90_SPLIT16_GATEUP_KERNEL_20260516.md).

P91 note: the row-tile sweep gives the next shape. 8/16/32/64 rows all pass the
`1e-5` tolerance gate. The 64-row variant has median **0.0443 ms**, and rows/ms
rises from **350** at 8 rows to **2892**. Widening the tile preserves the P90
math and amortizes launch/atomic overhead. P92 should attempt a full 512-row
gate/up expert shape. See
[`docs/LYNN_ENGINE_P91_SPLIT16_ROWTILE_SWEEP_20260516.md`](docs/LYNN_ENGINE_P91_SPLIT16_ROWTILE_SWEEP_20260516.md).

P92 note: the full gate/up expert sub-operator passes. The probe uses real Lynn
27B layer28 expert116, 512 gate rows plus 512 up rows, and K=2048. Median is
**0.0502 ms**, with `max_abs=4.77e-7` and `rel_l2=1.53e-7`. This answers the
route question: the current Lynn-native artifact can directly drive a complete
native-FP4 gate/up expert sub-operator. The remaining work is production kernel
engineering around launch overhead, atomics, and row ownership. See
[`docs/LYNN_ENGINE_P92_FULL_GATEUP_EXPERT_20260516.md`](docs/LYNN_ENGINE_P92_FULL_GATEUP_EXPERT_20260516.md).

Packed-resident memory note: the default server still keeps BF16 shadows so it
can run multi-request prefill. P11 proved that in a session-scoped lifecycle,
after prefill, Lynn engine can release 56.47 GiB of BF16 shadows, dropping
allocated memory from 81.06 GiB to 24.59 GiB while preserving exact greedy
decode ids. P12 wires this into an opt-in one-shot OpenAI server mode: the
first request releases 56.47 GiB, and a second prefill request fails loudly
with HTTP 409. P12 also verifies that current-position graph slots still match
eager decode exactly after release. See
[`docs/LYNN_ENGINE_P11_PACKED_RESIDENT_MEMORY_20260515.md`](docs/LYNN_ENGINE_P11_PACKED_RESIDENT_MEMORY_20260515.md)
and
[`docs/LYNN_ENGINE_P12_ONESHOT_SERVER_20260515.md`](docs/LYNN_ENGINE_P12_ONESHOT_SERVER_20260515.md).

## 27B quality and base-model status

27B comes from the Qwen 3.6 35B-A3B BASE variable-expert pruning route:

```text
BASE 35B-A3B
  → activation profile
  → variable-target expert pruning (1010 experts cut, front layers protected)
  → router fine-tune
  → Recovery LoRA
  → step5000 selected as final
  → BF16 merge
  → Lynn-native NVFP4 quantization
```

Known quality status:

| Variant | Status | Result |
|---|---|---|
| 27B BF16 step5000 | ✅ full eval | V8 strict 33/34 = 97.06%, V9 adjusted 37/59 = 62.71% |
| 27B NVFP4 step5000 | ✅ runtime smoke | 6-prompt resident smoke PASS,2-token greedy sanity PASS |
| 27B Q4_K_M | ⏳ not primary | variable-expert GGUF needs padding/format work; not the current Lynn-native path |

Recovery v1.1 targeted longctx/chem/sql was tested but did not replace step5000: it did not improve longctx and reduced aggregate scores. The selected base is therefore **step5000 final**.

## Roadmap: Lynn engine native-kernel restart + pragmatic llama.cpp fallback

Current restart framing is in [`RELEASE_NOTES_20260603.md`](RELEASE_NOTES_20260603.md). The client may still use llama.cpp/GGUF as a pragmatic short-term fallback, but Lynn engine is back on the parallel mainline: catching llama.cpp through native kernels.

### Short term (2-4 weeks): Lynn client + llama.cpp integration ship path

| Layer | Role | Implementation |
|---|---|---|
| **Inference backend** | llama.cpp ecosystem | Mac Metal / Windows / Linux CUDA + Q4_K_M GGUF |
| **Default model** | Qwen3.5-9B Q4_K_M-imatrix (5.3 GB) | Good enough for 80% of users |
| **Pro model** | Qwen3.6-35B-A3B Q4_K_M-imatrix (20 GB) | NVIDIA 24 GB+ users, opt-in, same llama.cpp stack |
| **Lynn client** | Auto hardware detect + install llama.cpp + download model + start server + register provider + tool-call gate + local-first routing | Electron + brain backend (sister repo `MerkyorLynn/Lynn`) |
| **Lynn agent** | tool routing / 6-layer memory / MCP / skills / cross-model fallback | Lynn's real moat |

**llama.cpp does "run, run fast, install small". Lynn does "use models well, call tools, remember, configure automatically".**

### Long term R&D: Lynn engine NVFP4 / W4A8 FP8 exploration (this repo)

**Bar to return to mainline (both must pass)**:
1. **Speed**: at the same hardware × same model, Lynn engine approaches or exceeds llama.cpp.
2. **Quality**: irreplaceable moat (GPQA / long-ctx / stability / structured output).

**R&D directions** (all commits preserved on `main`, never reverted):

- **W4A8 FP8 vectorised expert dispatch + CUTLASS grouped GEMM** to bypass the Python decode overhead (the architectural signal Wave 2 retry #4 exposed)
- **C++ service loop** when host gap clearly exceeds GPU compute gap (memory `LYNN_ENGINE_CPP_RUST_REWRITE_ROI_20260517` enumerates a 3-tier rewrite ROI plan)
- **Consumer Blackwell 32GB cards with FP4 MMA** widely available → Lynn engine native FP4 path (5/15 R6000 107-108 TPS strict default) becomes competitive again
- **MTP K=1 sequential 6/6 @ 26.4 TPS** correctness-clean baseline preserved; overlay on top of W4A8 base when that stabilises
- **Long-context 6.77× SGLang at 16K** retained as "advanced mode" selling point for NVIDIA Pro

### Historical decisions invalidated post 5/20

The following 5/15-5/16 lock-ins are no longer in force after 5/20:

- ~~Primary inference format: Lynn-native NVFP4 only~~ → **Production default = GGUF Q4_K_M-imatrix (llama.cpp ecosystem). NVFP4 / W4A8 FP8 stay in R&D.**
- ~~Model lock: Lynn-27B-A3B variable-pruned family~~ → **5/17 strategic pivot back to upstream Qwen3.6-35B-A3B + Qwen3.5-9B (Lynn 27B self-distillation dropped — quality down ~10%).**
- ~~Vertical companion to Lynn LoRA + pruning pipeline~~ → **Lynn engine converges to providing specialised acceleration for upstream model families. LoRA training side has its own track.**
- **Inference scope: single prompt + batch=1** retained (no PagedAttention) — Lynn client user scenario is single-user, no batched serving.
- **Inference hardware: Blackwell sm_12x** retained (DGX Spark / 5090 / R6000 PRO), but priority dropped to R&D.

## Tutorials — read these even if you're not writing your own engine

In writing Lynn engine we collected the parts of Qwen 3.6 35B-A3B that **diverge from Llama / Qwen 2 in ways the docs don't tell you**. Seven deep tutorials in [`tutorials/`](tutorials/):

| # | Topic | TL;DR |
|---|---|---|
| [01](tutorials/01_rmsnorm_one_plus_weight.md) | RMSNorm `(1.0 + w)` not `w` | Qwen 3 family RMSNorm has a +1 offset. Llama-style hits ~10x error. |
| [02](tutorials/02_rope_three_gotchas.md) | RoPE three gotchas | theta in `rope_parameters`, partial=0.25, GPT-NeoX half-split (not Qwen 2 even/odd) |
| [03](tutorials/03_attn_output_gate.md) | q_proj 2× per-head split | Must reshape per-head before chunk, or head_i_gate leaks into head_i_q |
| [04](tutorials/04_gated_delta_net.md) | linear_attention = GatedDeltaNet | Mamba-style chunk recurrence + delta rule + l2norm Q/K |
| [05](tutorials/05_three_invisible_bugs.md) | self-consistent bug postmortem | reference + lynn same-source-same-bug = passes wrong tests |
| [06](tutorials/06_moe_router_softmax_topk_order.md) | MoE router order + shared expert | softmax-all → topK → renormalize, with sigmoid-gated shared expert |

[`tutorials/posts/zhihu_qwen36_engine_postmortem.md`](tutorials/posts/zhihu_qwen36_engine_postmortem.md) is a single Zhihu-blog-style writeup of the highlights.

## Quick start (R6000 / Blackwell)

```bash
# 1. Prepare the Lynn-native NVFP4 artifact
MODEL=/root/autodl-tmp/models/lynn-27b-variable-recovery-step5000-nvfp4-final

# 2. Enable the current R6000 best env
export PYTHONPATH=/root/autodl-tmp/lynn-engine
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LYNN_PREFILL_WARMUP=1
export LYNN_LINEAR_ATTN_RECURRENT_BACKEND=triton_fused_prepare
export LYNN_LINEAR_ATTN_RECURRENT_INPLACE=1
export LYNN_MOE_IMPL=packed_nvfp4
export LYNN_QK_NORM_ROPE_BACKEND=triton_pair
export LYNN_RMSNORM_GATED_BACKEND=triton
export LYNN_LINEAR_ATTN_INPROJ_FUSED_NATIVE_FP4=1
export LYNN_NATIVE_FP4_LM_HEAD=1
export LYNN_LINEAR_STATE_UPDATE=inplace
export LYNN_LINEAR_BLOCK_GRAPH=1
export LYNN_LINEAR_BLOCK_GRAPH_REUSE=1
export LYNN_LINEAR_BLOCK_GRAPH_PREWARM=1
export LYNN_PACKED_DECODE=0
export LYNN_PACKED_DECODE_PREPARE_NATIVE=0
export LYNN_PACKED_SHARED_EXPERT=0

# 3. Run resident smoke
python benchmarks/resident_cli.py \
  --model "$MODEL" \
  --prompts-jsonl /root/autodl-tmp/reports/lynn-engine-p5/p7i_6prompt.jsonl \
  --max-new 32 \
  --chat-template \
  --out /tmp/lynn_27b_nvfp4_smoke.json

# 4. Optional: OpenAI-compatible server
python -m server.openai_http \
  --model "$MODEL" \
  --host 0.0.0.0 \
  --port 18099
```

## Repository layout

```
engine/
  loader.py                       FP8 e4m3 dequant + per-layer safetensors loader
  qwen36_block.py                 Original full transformer block (P1.1; bug-compatible reference)
  qwen36_linear_attn_block.py     GatedDeltaNet port — BIT-EXACT vs HF
  full_forward.py                 40-layer end-to-end forward (correct)
  inference_state.py              KV cache + recurrent state per request
  incremental_decode.py           prefill / decode primitives (Phase 3.1)
  moe_optimized.py                Three MoE optimizations (active / bmm / indexed bmm)
  convert_fp8_to_bf16.py          Offline FP8 → BF16 converter (CPU)
  test_*.py                       Per-layer alignment tests + multi-prompt validation
triton_kernels/
  attention.py / rope.py / rmsnorm.py / moe.py    P1 spike kernels
  moe_expert_ffn.py               Phase 3.2.3 fused MoE kernel (skeleton)
server/
  openai_http.py                  FastAPI OpenAI-compatible server
  README.md                       Brain integration guide
docs/
  DESIGN.md                       Architecture + roadmap (long)
  PHASE3_KV_CACHE_DESIGN.md       Phase 3.1 design doc
tutorials/
  README.md + 7 markdown tutorials + zhihu post
```

## Why this exists

Lynn brain serves Qwen 3.6 35B-A3B as the primary route for thousands of agent requests/day. vLLM (production today) realizes ~80% of Spark's memory-bandwidth ceiling. **Single-model lock-in + Blackwell sm_12x specialization should give the remaining 20% + better tail latency**.

But the more important deliverable is the **understanding** that comes from writing your own engine — the tutorials above are the artifact.

## Honest trade-offs

- ❌ Locked to Qwen 3.6 35B-A3B + Blackwell sm_12x.
- ❌ If model lineage replaces incompatibly, 4-6 weeks rewrite.
- ❌ No batching / no concurrent requests today.
- ❌ Greedy decode only (no sampling, no beam, no speculative).
- ✅ Fits Lynn brain's deployment exactly.
- ✅ Vertical integration with LoRA + pruning training pipeline.
- ✅ All architectural insights documented for the next person.

## License

TBD (likely MIT, decided before Phase 6 production cutover).
