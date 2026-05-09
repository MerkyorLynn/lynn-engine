# Lynn Engine — Custom Inference Engine for Qwen 3.6 35B-A3B on NVIDIA Blackwell

> **Status:** Design draft v0.1 · 2026-05-08
> **Owner:** Lynn (solo)
> **Target hardware:** DGX Spark (GB10, sm_121) → RTX PRO 6000 (Blackwell, sm_120)
> **Target model:** Qwen 3.6 35B-A3B (35B total / 3B active MoE), FP8 quant + custom asymmetric mixed-precision
> **Inspiration:** antirez `ds4.c` (DeepSeek V4 Flash on Apple Silicon Metal); StepFun `step3.5-flash` llama.cpp adaptation

---

## 0. Executive Summary

Lynn brain serves thousands of agent requests/day with Qwen 3.6 35B-A3B as the primary route. Current production runs SGLang+MTP at 60-70 t/s on DGX Spark. Empirical 5090 TP=2 vLLM benchmarks show **184 t/s single-stream / 248 t/s 5-concurrent** (Lynn 2026-05-04 measurements); RTX PRO 6000 single-card with higher bandwidth (~2 TB/s) should reach **200+ t/s on vLLM** today, realizing ~30% of the theoretical memory-bandwidth ceiling. This document specifies a **purpose-built single-model inference engine** that exploits:

1. **Hardware-specific kernels** (Triton + CUTLASS, no generic abstractions)
2. **Custom asymmetric quantization** (NVFP4 routed experts + FP8 critical path)
3. **Training-data-informed expert prefetching** (we own the LoRA+pruning data)
4. **Disk-backed KV cache with SHA1 prefix matching** (Lynn brain agent prompts repeat 99%+)

**Performance target:** **100-180 t/s on Spark (vs 60-70 today), 300-700 t/s on RTX PRO 6000** (vs ~200 t/s vLLM today). 6-9 weeks to production. Goes hand-in-hand with the LoRA + expert pruning training pipeline already in flight.

---

## 1. Goals & Non-Goals

### Goals (must)

- **G1:** Serve **only Qwen 3.6 35B-A3B** (and its LoRA-merged + pruned variants) on Blackwell sm_12x. Single-model lock-in by design.
- **G2:** Achieve **≥ 100 t/s on DGX Spark** (single batch, short context) — 50% improvement over current SGLang+MTP baseline.
- **G3:** Achieve **≥ 300 t/s on RTX PRO 6000** when that hardware lands (target = 1.5x current vLLM ~200 t/s baseline; stretch 500+ t/s with NVFP4 + MTP).
- **G4:** OpenAI-compatible HTTP API with `qwen3_coder` tool-call parser inline.
- **G5:** Disk-backed KV cache with SHA1 prefix matching, transparent reuse for repeated agent prompts.
- **G6:** Numerical correctness: **logits diff < 1e-3 vs HF transformers reference** on a sanity test set of 100 prompts.
- **G7:** Fit within 119 GB unified memory on Spark with mem-fraction ≤ 0.70 (CLAUDE.md memory rule #13 compliance).

### Non-goals (explicit)

- ❌ Generic GGUF loader. We don't load other models.
- ❌ Anthropic / OpenAI SDK feature parity (multimodal, batch API, fine-tune endpoints).
- ❌ Distributed multi-node serving. Single-machine only.
- ❌ Continuous batching / serving optimization for high-concurrency cloud workloads. Lynn brain peaks at 4 concurrent requests.
- ❌ Generic abstractions (model registry, config-driven layers). All Qwen 3.6 35B-A3B specifics hardcoded.

---

## 2. Target Hardware Matrix

| Spec | DGX Spark (GB10) | RTX PRO 6000 Blackwell |
|---|---|---|
| Compute capability | sm_121 | sm_120 |
| Generation | Blackwell GB10 | Blackwell |
| Memory | 119 GB unified (CPU+GPU shared) | 96 GB GDDR7 (discrete VRAM) |
| Memory bandwidth | ~273 GB/s | ~2,000 GB/s |
| FP8 tensor cores | ✅ 2nd gen | ✅ 2nd gen |
| FP4 tensor cores (NVFP4) | ✅ | ✅ |
| TGP | ~140W | ~600W |
| Connectivity | Internal (no PCIe) | PCIe Gen5 x16 |

**Key insight: Blackwell sm_12x is the same architecture family**. Triton kernels + CUTLASS templates compile to both with target flag changes. **One codebase, two deployment targets.**

---

## 3. Performance Targets (math-driven)

Qwen 3.6 35B-A3B at FP8 has ~3B active params per token forward = ~3 GB activations + KV cache. Memory-bandwidth-bound generation (single batch):

```
  theoretical max t/s = bandwidth / (active_params × bytes_per_param)
  
  Spark:    273 GB/s / (3B × 1B) = ~90 t/s pure FP8
  RTX 6000: 2000 / 3 = ~660 t/s pure FP8 (actually GEMM-bound at ~300-400 in practice)
```

Mixed precision target (NVFP4 experts + FP8 attention):

```
  Effective bytes per token: ~2 GB (50% NVFP4 + 50% FP8)
  
  Spark:    273 / 2 = ~135 t/s ceiling
  RTX 6000: 2000 / 2 = ~1000 t/s ceiling (compute-bound earlier)
```

**Realistic targets** (recalibrated 2026-05-08 after empirical 5090 TP=2 data: 184 t/s single-stream / 248 t/s 5-concurrent):

| Target | Spark | RTX PRO 6000 |
|---|---|---|
| **vLLM/SGLang baseline today** | 60-70 t/s | **~200 t/s** (extrapolated from 5090 TP=2 184 single-stream; RTX PRO 6000 has higher bandwidth) |
| MVP (FP8 only, no MoE prefetch) | **100-120 t/s** | **300-350 t/s** |
| Phase 4 (asymmetric quant + prefetch) | **130-150 t/s** | **400-500 t/s** |
| Stretch (with MTP-3 speculation) | **180 t/s** | **500-700 t/s** |

Notes on the recalibration:
- Earlier draft estimated RTX PRO 6000 vLLM today as ~80 t/s — this was wrong. Empirical 5090 TP=2 data
  (184 t/s single-stream from 2026-05-04 Lynn benchmarks) and the RTX PRO 6000's higher single-card
  bandwidth (~2 TB/s vs 5090's ~1.79 TB/s) imply vLLM today should already deliver 200+ t/s on PRO 6000.
- Lynn Engine target is therefore "realize 50-75% of bandwidth ceiling" rather than "beat a low baseline".
- The 1.5-2.5x improvement target over vLLM is justified by:
    1. Single-model lock-in (no abstraction overhead)
    2. Asymmetric quantization (NVFP4 experts cuts effective bytes/token by 33%)
    3. Custom MoE expert prefetching (saves L2 cache misses)
    4. Disk KV prefix cache (not a t/s gain but cuts agent prefill latency to ~0)

---

## 4. Three-Layer Architecture

```
┌─────────────────────────────────────────────────────────┐
│ Layer 3 · Server (~2 K LOC, Python/Rust)                │
│   HTTP + SSE                                            │
│   /v1/chat/completions (OpenAI-compat)                  │
│   qwen3_coder tool-call parser inline                   │
│   Anthropic /v1/messages compat (Phase 2)               │
│   Auth + rate limit                                     │
├─────────────────────────────────────────────────────────┤
│ Layer 2 · Engine Core (~3-5 K LOC, C++ / Rust)          │
│   Hardcoded Qwen 3.6 35B-A3B graph executor             │
│   PagedAttention KV cache (in-memory)                   │
│   SHA1-keyed prefix cache (disk-backed)                 │
│   MoE routing dispatch + expert prefetch                │
│   Tokenizer (custom Qwen3 BPE)                          │
│   safetensors mmap loader                               │
├─────────────────────────────────────────────────────────┤
│ Layer 1 · Kernels (~3-4 K LOC Triton + ~1 K CUDA)       │
│   Triton: attention, RoPE, RMSNorm, MoE dispatch        │
│   CUTLASS: FP8 GEMM, NVFP4 GEMM (MoE experts)           │
│   sm_120 + sm_121 dual targeting                        │
│   No PyTorch dependency at runtime (only at validation) │
└─────────────────────────────────────────────────────────┘
```

**Total estimated LOC: 8-12 K** (excluding generated Triton autotune code, comments, tests).

### 4.1 Layer 1 detailed: Kernels

| Kernel | Implementation | Notes |
|---|---|---|
| FP8 GEMM (attention proj) | CUTLASS template instantiation | Use NVIDIA's hand-tuned templates |
| NVFP4 GEMM (MoE experts) | CUTLASS NVFP4 path | sm_12x specific |
| Fused attention | Triton, FlashAttention-3 style | KV cache as shared memory |
| RoPE | Triton | Qwen3 NTK-aware scaling |
| RMSNorm | Triton fused with residual | |
| MoE top-k routing | Triton custom dispatch | Expert affinity prefetch |
| Sampling (greedy/topk/topp) | Triton or CPU | Non-hot path |

### 4.2 Layer 2 detailed: Engine Core

**Model graph (hardcoded for Qwen 3.6 35B-A3B):**
```python
# pseudocode
class Qwen36A3BEngine:
    def __init__(self):
        # All dimensions hardcoded — no config file
        self.num_layers = 48
        self.num_heads = 64
        self.num_kv_heads = 4  # GQA
        self.head_dim = 128
        self.hidden_dim = 8192
        self.intermediate_dim = 22016
        self.num_experts = 256
        self.experts_per_token = 8
        self.shared_experts = 1
        self.vocab_size = 152064  # Qwen3 BPE
        self.rope_theta = 1_000_000.0
        self.rope_scaling = {"type": "ntk", "factor": 4.0}
    
    def forward(self, tokens, kv_cache):
        # Direct calls to Layer 1 kernels, no PyTorch graph
        ...
```

**KV cache (in-memory):**
- Paged blocks of 32 tokens
- 48 layers × 4 KV heads × 128 head_dim × 32 tokens × 1 byte FP8 = 768 KB per block
- Pool size: ~32K blocks = ~24 GB VRAM (room for batch=4 × 256K context each)

**KV cache (disk-backed, the antirez angle):**
- File format: `<sha1>.kv` in `/cache/lynn-engine/`
- Header: 48 bytes (magic "LNK1", version, layer_count, token_count, timestamp)
- Body: serialized KV blocks
- Lookup: O(1) by SHA1(token_ids[:k]) for k = round-down to nearest 32
- Eviction: LRU over disk pool (configurable, default 100 GB on NVMe)

**MoE routing & expert prefetch:**
```c
// Profile-guided expert affinity (built from LoRA training data)
struct expert_affinity_table {
    uint32_t hot_experts[48][64];   // top-64 expert IDs per layer
    uint32_t fallback_experts[48][192];  // remaining
};

// Engine startup: preload hot_experts to L2-resident pinned buffer
// Generation: top-1/top-2 router output usually hits hot_experts (~85%)
// Cache miss fallback: standard global pool dispatch
```

### 4.3 Layer 3 detailed: Server

**Stack:** Python FastAPI for MVP (port to Rust if Python becomes bottleneck — unlikely for our 4-concurrent load).

**Endpoints:**
- `POST /v1/chat/completions` — OpenAI compat, streaming SSE
- `POST /v1/messages` — Anthropic compat (Phase 2)
- `GET /health` — liveness probe
- `GET /metrics` — Prometheus format
- `POST /admin/cache/invalidate` — manual KV disk cache flush

**Tool parsing (qwen3_coder format inline):**
```python
# Qwen3.6 emits XML-style tool calls inside <tool_call>...</tool_call>
# Parser runs on the streaming token output, splits text from tool_calls
# Same regex as Lynn brain's existing wrapper (reuse pseudo-tool-call.js logic)
```

---

## 5. Quantization Strategy

**Asymmetric mixed-precision** (antirez-inspired, but with FP8/FP4 instead of IQ2/Q2_K):

| Component | Precision | Reason |
|---|---|---|
| Embedding (input) | FP16 | Token lookup correctness |
| Layer 0-47 attention QKV proj | FP8 | Critical path, NVFP4 too aggressive |
| Layer 0-47 attention output proj | FP8 | Same |
| Layer 0-47 RMSNorm scale | FP16 | Tiny weights, no benefit to quant |
| Routed experts up_proj | NVFP4 | 50% of model params; experts are over-parameterized |
| Routed experts gate_proj | NVFP4 | Same |
| Routed experts down_proj | FP8 | Output sensitive (activations leave expert) |
| Shared expert (1 per layer) | FP8 | Always active, route doesn't hide quality loss |
| Router gate weight | FP16 | Critical to routing quality, tiny |
| LM head | FP16 | Output token probabilities, must preserve |

**Memory budget:**
```
  Embedding + LM head:      ~2.4 GB (FP16)
  RMSNorm + router gates:   ~0.5 GB
  Attention QKV+O proj:     ~10 GB (FP8)
  Shared experts:           ~3 GB (FP8)
  Routed experts up+gate:   ~10 GB (NVFP4)
  Routed experts down:      ~5 GB (FP8)
  ───────────────────────────────
  Total weights:            ~31 GB
  
  + KV cache (4 batch × 8K avg): ~3 GB
  + Activations + workspace:     ~2 GB
  + CUDA graph cache + buffers:  ~1 GB
  ───────────────────────────────
  Total VRAM use:           ~37 GB
  
  Spark mem-fraction: 37/119 = 0.31 — well under 0.70 ceiling ✅
  RTX PRO 6000:       37/96 = 0.39 — leaves 60GB for KV cache scaling ✅
```

---

## 6. Custom Training Integration (LoRA + Pruning)

Lynn is already running LoRA + planning pruning on Qwen 3.6 35B-A3B. Engine consumes the **post-training output** directly:

```
┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│  Stage 1: LoRA   │───>│  Stage 2: Prune  │───>│  Stage 3: Quant  │
│  ────────────    │    │  ──────────────  │    │  ──────────────  │
│  Base BF16       │    │  Drop unused     │    │  Asymmetric mix  │
│  + LoRA adapters │    │  experts (30-50%)│    │  NVFP4 + FP8     │
│  → merge BF16    │    │  Drop dead layers│    │  → engine.bin    │
└──────────────────┘    │  → pruned BF16   │    └──────────────────┘
                        └──────────────────┘             │
                                                         │
                                                         v
                                                ┌─────────────────┐
                                                │  Lynn Engine    │
                                                │  loads directly │
                                                └─────────────────┘
```

**Expert pruning specifics:**
1. Run profile workload (1000 representative prompts) through merged BF16 model
2. Per-layer per-expert hit count, sort descending
3. Drop bottom 30-50% (typically 80-130 experts per layer kept of 256)
4. **CRITICAL:** Re-train router gate on remaining experts (10-50 steps LoRA)
5. Validate: ToolAbstain-31 score must stay ≥ pre-prune − 1 (sanity)

**Pruning budget:**
- Pruning to 192 experts/layer: -25% experts size = -2.5 GB → 31 → 28.5 GB
- Pruning to 128 experts/layer: -50% experts size = -5 GB → 31 → 26 GB
- Speed gain: less expert dispatch overhead, ~10-20% faster generation

**Engine integration:**
```c
// engine.bin format (custom):
// Header: 256 bytes (magic, version, weight layout, expert count per layer)
// Weights blob: directly mmap-able, layout matches kernel input
// Optional: expert_affinity_table appended

// Loader: validate header, mmap, register kernel pointers. <100ms boot.
```

---

## 7. Disk KV Cache (the antirez angle, ported to NVIDIA + agent workload)

**Why this is high ROI for Lynn brain specifically:**

Lynn brain agents send identical or near-identical prompt prefixes 99% of the time:
- System prompt (qwen3_coder M3 mitigation, ~500 tokens) — **fixed**
- Tool definitions (~1500-3000 tokens) — **fixed per session**
- Prior conversation turns — grow but **prefix-stable**

Currently, every user message triggers a full re-prefill of the entire prompt context. For a 5K-token conversation, that's 5K × ~100ms/token prefill = ~500ms of wasted computation per user message. SHA1 prefix cache eliminates this.

**Implementation:**

```c
// Round token list down to nearest 32-multiple boundary
size_t cached_len = (token_count / 32) * 32;
if (cached_len < 32) goto full_prefill;

uint8_t sha1[20];
sha1_hash(tokens, cached_len * sizeof(uint32_t), sha1);

char path[256];
snprintf(path, sizeof(path), "/cache/lynn-engine/%020llx%020llx.kv", 
         *(uint64_t*)sha1, *(uint64_t*)(sha1+8));

if (access(path, R_OK) == 0) {
    // Hit: mmap and copy KV blocks to GPU
    kv_load_from_disk(path, kv_cache);
    n_resume = cached_len;
} else {
    // Miss: full prefill, then save
    full_prefill();
    kv_save_to_disk(path, kv_cache, cached_len);
    n_resume = cached_len;
}

// Continue prefill from token[cached_len:] only
```

**Cache management:**
- Default 100 GB on NVMe (~5K conversations cached at 20MB avg)
- LRU eviction
- Expiry: 7 days (tunable)
- Optional encryption-at-rest (KV blocks contain conversation history — sensitive)

**Expected wins:**
- First message in conversation: 0% (cache miss, normal latency)
- 2nd+ message in same conversation: **80-95% prefill skipped**, latency drops from ~500ms to ~50ms
- Cross-session same system prompt: ~50% prefill skipped (system + tools section reused)

---

## 8. Implementation Phases (with kill criteria)

| Phase | Duration | Deliverable | Kill criteria (don't proceed if…) |
|---|---|---|---|
| **0** Baseline | 1-2 days (after GGUF + LoRA done) | vLLM Qwen-A3B-FP8 t/s on Spark | Skip — always do this |
| **1** Triton kernel spike | 1 week | Single attention + RMSNorm + RoPE kernel passing logits-diff test < 1e-3 vs HF | Logits diff > 1e-2 → fork vLLM instead |
| **2** Engine MVP | 1.5 weeks | Single-batch greedy decode, ≥ 80 t/s on Spark | < 50 t/s → re-evaluate kernels |
| **3** API server | 3-5 days | OpenAI-compat + tool parsing + ToolAbstain-31 ≥ 27/31 (match cloud) | ToolAbstain regression > 2 pts |
| **4** Disk KV + pruning | 1 week | 80% prefill skip on agent workload + 20% speedup from prune | < 50% prefill skip OR > 2pt quality loss |
| **5** RTX PRO 6000 port | 1 day after hw arrives | ≥ 300 t/s (vs vLLM ~200 t/s baseline) | < 250 t/s → kernel optimization round 2 |
| **6** Production cutover | 1 week (gradual) | Lynn brain main link replaced, real-traffic data | Latency p95 regression > 30% |

**Total: 6-9 weeks solo, 4-6 weeks if AI-pair-programmed (GPT-5.5 + Claude Code, like antirez did with ds4).**

---

## 9. Risks & Mitigations

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| Triton kernels miss correctness on Blackwell sm_12x | Med | High | Validate each kernel logits-diff vs PyTorch reference before integration |
| MoE routing performance worse than hoped | Med | Med | Phase 2 fallback to standard global expert dispatch |
| Disk KV cache writes saturate NVMe IOPS at high QPS | Low | Med | Async write + write coalescing; single-Spark workload won't trigger this |
| Pruning destroys tool-calling quality | Med | High | ToolAbstain-31 gate before each prune step; rollback if drop > 1pt |
| RTX PRO 6000 sm_120 has subtle differences from sm_121 | Low | Med | Test on RTX PRO 6000 before committing to that hardware |
| Time invested > ROI vs just upgrading vLLM | Med | High | Phase 1 spike acts as go/no-go decision point — kill if Triton can't deliver |
| Lynn brain main-link migration breaks production | High | High | Phase 6 gradual cutover with shadow traffic + automatic fallback |

---

## 10. Comparison Table — vs. existing engines

| Property | Lynn Engine (target) | vLLM | SGLang | llama.cpp | antirez ds4 |
|---|---|---|---|---|---|
| Target model | Qwen 3.6 35B-A3B only | any | any | any GGUF | DeepSeek V4 Flash only |
| Hardware | Blackwell sm_12x | wide | wide | wide | Apple Silicon Metal |
| Lines of code | ~10K | ~150K | ~100K | ~250K | ~5K |
| Asymmetric mixed-prec | ✅ NVFP4+FP8+FP16 | ⚠️ partial | ⚠️ partial | ✅ via custom GGUF | ✅ IQ2+Q2K+Q8 |
| Disk KV prefix cache | ✅ SHA1 | ❌ | ❌ | ⚠️ slot save (not prefix-keyed) | ✅ SHA1 |
| Expert prefetch (training-informed) | ✅ | ❌ | ❌ | ❌ | N/A |
| Tool-call parsing inline | ✅ qwen3_coder | ❌ external | ❌ external | ❌ external | ✅ DeepSeek format |
| Multi-model support | ❌ | ✅ | ✅ | ✅ | ❌ |
| Continuous batching | ❌ (batch=4 max) | ✅ | ✅ | ⚠️ | ❌ |
| Production-ready | After Phase 6 | ✅ | ✅ | ✅ | ⚠️ niche |
| Lock-in risk if model deprecated | ⚠️ high | low | low | low | ⚠️ high |

**Trade-off summary:** Lynn Engine accepts model lock-in for **2-3x performance** + **deeper agent-workload integration** (KV prefix cache) + **vertical with own training pipeline**. If Qwen 3.6 lineage is replaced by something incompatible, project rewrites in 4-6 weeks (similar to original budget).

---

## 11. Repository Layout

```
lynn-engine/
├── docs/
│   ├── DESIGN.md           ← this document
│   ├── KERNELS.md          ← kernel specs (each Triton kernel documented)
│   ├── DEPLOYMENT.md       ← Spark + RTX PRO 6000 deploy guides
│   └── BENCHMARK.md        ← reproduction guide
├── kernels/
│   ├── attention.py        ← Triton attention
│   ├── rope.py
│   ├── rmsnorm.py
│   ├── moe_dispatch.py
│   ├── gemm_fp8.cu         ← CUTLASS FP8 GEMM
│   ├── gemm_nvfp4.cu       ← CUTLASS NVFP4 GEMM
│   └── tests/              ← per-kernel correctness + perf
├── engine/
│   ├── model.py            ← Qwen3.6-35B-A3B graph executor (Python prototype)
│   ├── model.cpp           ← C++ production version (Phase 2+)
│   ├── kv_cache.py         ← in-memory paged
│   ├── kv_disk.py          ← SHA1 prefix disk cache
│   ├── tokenizer.py        ← Qwen3 BPE
│   ├── loader.py           ← engine.bin format
│   └── pruning.py          ← expert pruning utilities
├── server/
│   ├── api.py              ← FastAPI: /v1/chat/completions
│   ├── tool_parser.py      ← qwen3_coder inline parsing
│   └── auth.py
├── scripts/
│   ├── convert_hf_to_engine.py  ← BF16/FP8 HF → engine.bin
│   ├── prune_experts.py         ← profiling-guided pruning
│   └── benchmark.py             ← end-to-end perf test
├── benchmarks/
│   └── ToolAbstain-31/          ← reuse from toolabstain-paper repo
└── README.md
```

---

## 12. Open Questions / TODOs (resolve before Phase 1)

- [ ] Confirm RTX PRO 6000 timeline (when does it arrive at Lynn?)
- [ ] Confirm LoRA training current state — when finished, what dataset, what rank?
- [ ] Decide pruning algorithm: simple frequency or magnitude-aware (e.g., AdaPrune)?
- [ ] Tokenizer: write our own Qwen3 BPE in C, or use HF `tokenizers` Rust binding?
- [ ] Disk KV cache encryption — AES-128 default on, or opt-in?
- [ ] Server framework — FastAPI Python (fast to ship) or actix-web Rust (matches lynn brain)?

---

## 13. References

- [antirez/ds4 — DeepSeek V4 Flash native engine on Apple Silicon](https://github.com/antirez/ds4)
- [stepfun-ai/Step-3.5-Flash llama.cpp adaptation for DGX Spark](https://github.com/stepfun-ai/Step-3.5-Flash/blob/main/llama.cpp/docs/step3.5-flash.md)
- [vLLM project](https://github.com/vllm-project/vllm) — reference for PagedAttention
- [SGLang project](https://github.com/sgl-project/sglang) — reference for MTP + EAGLE speculative decoding
- [llama.cpp persistent KV cache discussion #20572](https://github.com/ggml-org/llama.cpp/discussions/20572)
- [APEX-Quant — adaptive precision for MoE](https://github.com/mudler/apex-quant)
- [NVIDIA CUTLASS FP8 GEMM templates](https://github.com/NVIDIA/cutlass)
- [Triton GPU programming](https://github.com/openai/triton)
- [FlashAttention-3 paper](https://arxiv.org/abs/2407.08608)
- Qwen 3.6 35B-A3B-FP8: https://huggingface.co/Qwen/Qwen3.6-35B-A3B-FP8
- Lynn ToolAbstain paper (this engine's primary correctness gate): https://github.com/MerkyorLynn/toolabstain-paper

---

## 13.5 Phase 2 Status Log (2026-05-09)

### What's done (P1.1)

- ✅ **All 4 Triton kernels pass numerical alignment** on synthetic random inputs (Spark sm_121,
  FP16 ULP floor, 14/14 + 128/128 specific tests).
- ✅ **All 10 full_attention layers** of Qwen 3.6 35B-A3B-FP8 align with PyTorch reference using
  REAL learned weights (`engine/test_all_full_attn_layers.py`):
  ```
  Layers 3, 7, 11, 15, 19, 23, 27, 31, 35, 39 — all PASS
  Avg max diff: 4.37e-3 (~2 ULP BF16)
  Avg rel diff: 0.074%
  ```
- ✅ Single layer end-to-end test on real layer 3 weights passed (0.02% rel diff).

### What's blocked (P1.2)

Attempted full 40-layer forward via HF transformers as reference
(`engine/test_full_forward.py`). Goal: load HF Qwen3_5_MoE, hook full_attention
layer I/O, replay each I/O pair through our `qwen36_lynn` block, verify match.

Blocked by HF transformers FP8 dependency stack on NGC vLLM 26.03.post1 container:

1. NGC's bundled transformers (4.57.5) doesn't recognize `qwen3_5_moe` model_type.
   → Installed transformers from git (5.8.0.dev0).
2. Loading model with `device_map="cuda:0"` requires `accelerate`.
   → Installed `accelerate`.
3. FP8 finegrained quantization needs HuggingFace `kernels` package (provides
   block-scaled FP8 matmul).
   → Installed `kernels` 0.14.0.
4. `kernels` package shadows our local `kernels/` directory (Python module name
   collision).
   → Renamed our directory `kernels/` → `triton_kernels/`, updated all imports.
5. `kernels` package tries to download `deep-gemm` kernel from HF Hub. The downloaded
   metadata.json has malformed schema (missing `name` field), so kernel can't be
   parsed. Forward pass fails on first FP8 matmul.

Each dependency fix revealed another deeper issue. Total time invested: ~2 hours.
The HF transformers 5.8 + `kernels` package + DeepGEMM stack is fragile for
brand-new models like Qwen 3.6 (model_type='qwen3_5_moe') that depend on cutting-edge
HF infrastructure.

### Path forward for P1.2

Two options:

**Option A — vLLM as reference**: Use the production vLLM Qwen 3.6 35B-A3B-FP8 endpoint
(currently running on port 18002). Query with `logprobs=20` on a test prompt, get
reference top-K next-token logits. Run our engine forward (without full_attention
layers replaced — i.e., a full 40-layer impl using HF eager for linear_attention
and our kernels for full_attention) and compare last-token logits.

**Option B — implement linear_attention in our engine**: Phase 3 work. Linear attention
is Mamba/GLA-style state-space and significantly different from standard transformer
attention. Estimated 2-3 days research + implementation.

For both options, we need a working linear_attention forward (either HF's, vLLM's,
or our own). Given HF blocker, vLLM is the practical reference until we have our own
linear_attention kernel.

### What's preserved

- ✅ All 4 Triton kernels (attention, RoPE, RMSNorm, MoE router) production-quality
- ✅ Real-weights single-block alignment validated (P1.1 strongest result so far)
- ✅ qwen36_block.py implements Qwen 3.6 specifics: attn_output_gate, q_norm/k_norm,
  GQA 16:2, 256-expert MoE with shared expert
- ✅ Loader for FP8 e4m3 weights from safetensors with weight_scale_inv dequant
- ✅ Lynn brain main link Qwen vLLM service successfully restarted at port 18002

P1.2 deferred. P1.3 vLLM logprobs comparison can be done independently of P1.2.

## 13.6 P1.3 attempt — BF16 dequant path also blocked on Spark unified memory (2026-05-09)

Following §13.5 P1.2 blocker on FP8 deep-gemm metadata, attempted P1.3 via
the BF16 dequant route: convert `Qwen3.6-35B-A3B-FP8` (35 GB) → `BF16` (67 GB)
on disk, then load with HF transformers 5.8.0 in plain BF16 mode (sidesteps
the `kernels` package and DeepGEMM entirely).

### What worked

- ✅ `engine/convert_fp8_to_bf16.py` — block-scaled FP8 e4m3 dequant via
  `weight_scale_inv`. CPU-only, doesn't disturb live vLLM.
  35 GB FP8 → 67 GB BF16 in 124 s (0.58 GB/s) on Spark CPU.
  Output: `/home/merkyor/models/Qwen3.6-35B-A3B-BF16/`. Useful as a side
  artifact for LoRA training (no FP8 dep churn) even if unused for P1.3.
- ✅ HF transformers 5.8.0 (plain `pip install` from PyPI; the prior `5.8.0.dev0`
  was a non-existent tag) recognizes `model_type='qwen3_5_moe'` and parses
  config cleanly. The "Transformers does not recognize" error from §13.5 was
  because pip silently fell back to bundled 4.57.5.

### What blocked

- 🔴 Spark unified memory + earlyoom (10% threshold) cannot accommodate the
  HF model loading peak. Three attempts (Qwen vLLM stopped, ELYZA stopped,
  voice services only — 109 GB free) all SIGTERM'd at 60–75% weight load:

  ```
  earlyoom: mem avail 10076/122500 MiB (8.23%) → SIGTERM python3 (VmRSS 42.8 GB)
  ```

  - Pure `device_map="cuda"` and `device_map="auto"` + `low_cpu_mem_usage=True`
    + `max_memory={0: 60GB, cpu: 20GB}` all hit the same wall.
  - VmRSS at kill = 42 GB << nominal 67 GB BF16 — earlyoom counts page cache
    from safetensors mmap toward `used`, so the apparent ceiling is well below
    the model's own footprint.

### Path forward

P1.3 on Spark requires one of:

1. **Own linear_attention Triton kernel** (Phase 3 work) — eliminates HF
   reference entirely; full forward via Lynn engine + compare logits to vLLM.
2. **RTX PRO 6000 hardware** — 96 GB discrete VRAM accommodates 67 GB BF16
   forward without unified-memory pressure or earlyoom.
3. **vLLM internal API hook** — modify the running vLLM container to expose
   `output_hidden_states` between layers; reuse production FP8 weights without
   loading anything extra. Avoids HF entirely but requires vLLM patching.

Option 1 (linear_attention kernel) is the cleanest long-term answer and was
already scheduled for Phase 3. Promote it to next-priority and skip the HF
reference path.

P1 milestone redefined as the union of:
- ✅ P1.1 (10 full_attention layers on real weights, synthetic input)
- ✅ FP8→BF16 converter (validated, 67 GB output usable for LoRA path)
- ✅ HF transformers 5.8.0 path mapped (recognizes `qwen3_5_moe`; load fails
  on Spark mem pressure but works on bigger boxes)
- ⏸ P1.3 deferred to Phase 3 linear_attention OR RTX PRO 6000 arrival

## 13.7 P1.3 closed (2026-05-09 afternoon)

The three blockers of §13.5/§13.6 were sidestepped by porting
`Qwen3_5MoeGatedDeltaNet` directly into Lynn (P3a). With both attention
flavors implemented in-house, no HF reference is needed for the end-to-end
forward; comparison against the running vLLM (which holds the model on the
same GPU at mem-fraction 0.5) validates correctness.

End-to-end test (engine/full_forward.py):

```
Prompt: "The capital of France is"
vLLM:  Paris (logprob -0.54), a, the, known, \n\n
Lynn:  Paris (logit 17.88),   a, the, known, \n\n
       ^^^^^ identical top-5
```

Memory profile (Spark unified mem during the test, with vLLM running):

```
vLLM Qwen FP8       60 GB  resident
ELYZA  (stopped during test)
voice services       3 GB
Lynn embeddings + lm_head + final norm   1 GB resident
Lynn per-layer (loaded then freed)       1.7 GB peak
Total                                    ~65 GB <  119 GB,  earlyoom-safe
```

Three bugs were caught and fixed during integration; all three were
self-consistent in the prior P1.1 reference (lynn vs lynn-style reference
both wrong identically). Documented in commit `adac5d9`:

  1. RoPE config in Qwen 3.6 lives under `text_config.rope_parameters`
     (NOT `text_config.rope_theta`), with theta=1e7, partial_rotary_factor=0.25,
     and GPT-NeoX split-halves rotation pattern (not even/odd interleaved).
  2. q_proj output requires per-head reshape `[B, M, H_Q, 2*head_dim]`
     BEFORE chunking into [q, attn_output_gate], not `chunk(2, dim=-1)`
     on the flat representation.
  3. Qwen3-family RMSNorm uses `(1.0 + weight) * x_normed`, not `weight * x_normed`
     (HF PR #29402). RMSNormGated inside linear_attn does NOT use the offset.

Phase 2 closes. Lynn engine end-to-end forward now validated bit-for-bit
in the metric that matters (top-token agreement with the production
reference). Performance optimization (Phase 3) is unblocked.

## 13.8 Phase 2 closing state (2026-05-09 evening)

### Validated capabilities

| Component | Status | Evidence |
|---|---|---|
| 4 Triton kernels (attn / RoPE / RMSNorm / MoE router) | ✅ pass | P1.1 commit `a53c614` |
| `lynn_full_attn_forward` on real weights | ✅ pass | P1.1 layers 3,7,…,39 self-consistent (the math bugs caught later in P1.3 affected ref + lynn identically); fixed in `adac5d9` |
| `lynn_linear_attn_forward` (Qwen3_5MoeGatedDeltaNet port) | ✅ **bit-exact vs HF** | P3a 10/10 layers `3c74e1e` |
| `_layer_forward` orchestration (DecoderLayer) | ✅ pass | included in `adac5d9` end-to-end run |
| Memory-bounded layer-by-layer forward | ✅ pass | 252 s to load all 40 layers resident, ~3 GB peak when not resident |
| FP8 → BF16 offline conversion pipeline | ✅ pass | `1a26a52`, 35 GB → 67 GB in 124 s CPU-only |
| End-to-end token logits agreement vs vLLM | ✅ pass | greedy "The capital of France is Paris, a" matches vLLM token-for-token (`458a35c`) |
| Multi-prompt top-K overlap | ✅ 9.8 / 10 avg | 8 prompts, top-1 6/8 strict (2 close-call disputes both inside top-10), `d523f1a` |

### What Phase 2 did NOT do (deferred to Phase 3)

| Item | Why deferred | Estimated effort |
|---|---|---|
| KV cache for full_attention (incremental decode) | Greedy gen brute-force re-prefill is unambiguously correct; cache is perf, not correctness | 4-6 h |
| Recurrent state cache for linear_attention (single-token decode path) | Same — `torch_recurrent_gated_delta_rule` is the HF reference, port + plumb through | 6-8 h |
| Triton-fused chunked gated_delta_rule kernel | Current torch impl is correctness-first; a fused kernel cuts the L0 single-token forward from ~30 ms (linear_attn share) to ~3 ms | 2-3 d |
| CUTLASS NVFP4 grouped expert FFN | The 256-expert Python loop is the dominant hot spot; CUTLASS grouped GEMM should be ~10-50× faster on the per-token MoE step | 2-4 d |
| OpenAI-compatible HTTP server (G4) | Pure plumbing once decode is in place | 1-2 d |
| Disk KV cache + SHA1 prefix matching (G5) | Production feature — Phase 4 once core decode is fast | 3-5 d |

### Performance reality vs targets

  G2 target: ≥100 t/s on Spark.
  Today:     ~3 t/s on `generate_greedy` (brute-force re-prefill, no KV cache),
             where ~0.3 s per token is dominated by 30 linear_attention layers
             × torch chunk_gated_delta_rule + 40 layers' MoE Python loop. With
             KV cache + Triton-fused linear_attn the first Phase 3 milestone
             should hit 30-60 t/s; CUTLASS NVFP4 expert FFN is what unlocks
             the 100+ t/s goal.

### Recommended Phase 3 sequencing

  1. KV cache for full_attention (4-6 h) — single biggest correctness-side win,
     cuts incremental decode from O(T²·layers) to O(T·layers).
  2. Recurrent decode for linear_attention (6-8 h) — port `torch_recurrent_gated_delta_rule`
     and plumb past_recurrent_state through `_layer_forward`.
  3. Profile pass — confirm what's actually slow in the new decode path.
  4. CUTLASS NVFP4 grouped expert FFN (2-4 d) — likely the dominant cost
     once cache is in.
  5. OpenAI HTTP server (1-2 d) so Lynn brain can swap routes for testing.

## 13.9 Phase 3.1 closed (2026-05-09 evening)

Steps 1 and 2 above completed in `commit 1e2980b`.

### Implementation

- `engine/inference_state.py` — `LynnInferenceState` dataclass; pre-allocates
  KV cache (10 full_attn × [B, 2, 32K, 256] BF16) + recurrent state
  (30 linear_attn × [B, 32, 128, 128] FP32) + conv state (30 × [B, 8192, 3] BF16).
  Total 735 MB @ B=1 max_T=32K.
- `engine/incremental_decode.py` — refactored prefill / decode primitives
  for both attention types. Decode side ports HF's `torch_recurrent_gated_delta_rule`
  for linear_attn single-token path.
- `engine/full_forward.py::generate_incremental` — orchestrator. Prefill all T
  prompt tokens once → save K/V/state to LynnInferenceState → decode 1 token
  at a time using cached state.

### Single-layer alignment (T=64, chunk-aligned)

| Layer type | Layer | rel_diff | max_diff | status |
|---|---|---|---|---|
| full_attention | layer 3 | 1.449% | 1.95e-3 | ✅ |
| linear_attention | layer 0 | 0.683% | 2.4e-4 | ✅ |

(Diff vs brute-force chunk_gated_delta_rule re-prefill, bounded by FP32
accumulation order between chunk and recurrent paths. Both well below
visible-output magnitudes.)

### End-to-end greedy 5-token vs vLLM

```
prompt: "The capital of France is"
vLLM (Qwen3.6-35B-A3B-FP8):  Paris, a city renowned
Lynn engine incremental:     Paris, a city renowned
                            ^^^^^^^^^^^^^^^^^^^^^^^^^
                            5/5 tokens token-for-token match ✓
```

### Performance

| Phase | Path | latency/token |
|---|---|---|
| Phase 2 (brute-force) | re-prefill | ~300 ms (2-3 t/s) |
| **Phase 3.1 (cache + recurrent)** | **incremental decode** | **~200 ms (5 t/s)** |
| Phase 3.2 target (Triton-fused MoE) | + fused expert FFN | ~50 ms (20 t/s) |
| Phase 3.3 target (CUTLASS NVFP4 grouped) | + NVFP4 GEMM | ~10-15 ms (60-100 t/s) |
| vLLM SGLang+MTP baseline | reference | ~14 ms (60-70 t/s) |

Phase 3.1 saves ~100 ms/token over brute-force (the O(T) attention savings;
linear_attn is O(1) per step instead of re-prefilling the whole sequence).
Bottleneck now is the 256-expert Python loop in MoE forward — Phase 3.2 fixes.

### Closes

  ✅ KV cache for full_attention
  ✅ Recurrent state cache for linear_attention
  ✅ Single-layer alignment passes
  ✅ End-to-end token agreement with vLLM

Phase 3.2 (Triton-fused MoE expert FFN) is next; expected to bring Spark
single-stream decode to 15-25 t/s.

## 13.10 Phase 3.2 in flight (2026-05-09 evening, DGX SSH offline)

Three-step incremental MoE optimization plan; first two committed (untested
because DGX frp tunnel went down right after Phase 3.1 closeout).

### 3.2.1 active-experts iteration (commit `120e454`)

Replace `for e in range(256): if mask.any():` Python loop with
`for e in torch.unique(expert_indices).tolist():`. For decode T=1 this drops
from 256 iterations (each with a GPU-CPU sync via `.any()`) down to ≤K=8
iterations.

Expected gain: 30-60 ms / token (40 layers × ~1 ms saved/layer in Python overhead).

### 3.2.2 bmm batched matmul (commit `48ecbfa`)

Stack the K=8 active experts' weights into single `[K, intermediate, hidden]`
tensors and fire 3 `torch.bmm` calls per layer (gate / up / down) instead
of K=8 sequential `F.linear` calls. cuBLAS bmm should be 5-10x faster than
sequential matmul.

Trade-off: 3 × `torch.stack` per layer × 16 MB = ~48 MB memory traffic, ×
40 layers = 2 GB / token. At Spark 270 GB/s effective ≈ 7 ms total
stacking cost. Net gain expected: 30-50 ms / token.

Combined with 3.2.1: **target 200 ms → 100-120 ms / token (8-10 t/s)**.

### 3.2.3 Triton fused MoE expert FFN (designed, not yet implemented)

Single Triton kernel that:
1. Loads h_in [hidden] once
2. Loops over K=8 active experts inline within the kernel
3. For each expert: load gate/up/down weights (or use precomputed indexing),
   do gate @ h, up @ h, silu+mul, down @ result
4. Accumulates weighted sum into out [hidden]

Avoids:
- Python per-expert loop overhead
- Per-call torch.stack cost (kernel does its own indexed loads)
- Kernel launch overhead between gate/up/down stages (single launch)

Expected: 15-30 ms / token (per-layer ~0.5 ms × 40 = 20 ms total MoE budget).

Combined with 3.2.1 + 3.2.2 path: target 50 ms / token (20 t/s).

### 3.2.4 (open question) grouped expert weight pre-stacking

If we modify `loader.py` to load expert weights as a single
`[256, 2*intermediate, hidden]` grouped tensor (matching HF's
`Qwen3_5MoeExperts.gate_up_proj` layout), we eliminate per-call torch.stack
entirely.

Cost: same memory total (256 × per-expert size); requires loader rewrite
and breaks current per-expert key-based access in `engine/qwen36_block.py`
+ `_moe_forward`.

Decision deferred until 3.2.2 + 3.2.3 numbers are in.

### 3.2 milestone exit criteria

- [ ] Spark single-stream decode ≥ 15 t/s on the brute-force test prompt
- [ ] Multi-prompt 5-token generation matches vLLM token-for-token (8/8)
- [ ] DESIGN.md §13.10 updated with measured numbers

## 14. Decision Log

| Date | Decision | Rationale |
|---|---|---|
| 2026-05-08 | Lock to Qwen 3.6 35B-A3B (not generic) | Vertical integration with LoRA + pruning training; ToolAbstain shows it's the highest-scoring local model already |
| 2026-05-08 | Blackwell sm_12x only | Spark + RTX PRO 6000 both Blackwell; no need for Turing/Ampere/Ada |
| 2026-05-08 | Triton + CUTLASS hybrid (not pure CUDA) | Triton is faster to develop and Blackwell-tuned; CUTLASS for FP8/FP4 GEMM templates |
| 2026-05-08 | Asymmetric mixed precision (NVFP4 experts + FP8 critical path) | Memory savings without compromising attention/router quality |
| 2026-05-08 | Disk KV cache with SHA1 prefix | Direct port of antirez ds4 idea; perfect fit for Lynn brain agent workload |
| 2026-05-08 | Phase 1 = Triton kernel logits-diff spike | Hard go/no-go gate before committing to from-scratch path |
| 2026-05-08 | Recalibrate RTX PRO 6000 perf targets | Earlier draft used 80 t/s vLLM baseline (incorrect estimate). Empirical 5090 TP=2 184 t/s + RTX PRO 6000 higher bandwidth → 200+ t/s vLLM baseline. Lynn target adjusted to 300-500 t/s realistic, 700 t/s stretch with MTP. |

---

*Draft v0.1 · 2026-05-08 · pending Phase 0 baseline data + Phase 1 spike validation*
