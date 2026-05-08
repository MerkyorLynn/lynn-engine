# Lynn Engine — Custom Inference Engine for Qwen 3.6 35B-A3B on NVIDIA Blackwell

> **Status:** Design draft v0.1 · 2026-05-08
> **Owner:** Lynn (solo)
> **Target hardware:** DGX Spark (GB10, sm_121) → RTX PRO 6000 (Blackwell, sm_120)
> **Target model:** Qwen 3.6 35B-A3B (35B total / 3B active MoE), FP8 quant + custom asymmetric mixed-precision
> **Inspiration:** antirez `ds4.c` (DeepSeek V4 Flash on Apple Silicon Metal); StepFun `step3.5-flash` llama.cpp adaptation

---

## 0. Executive Summary

Lynn brain serves thousands of agent requests/day with Qwen 3.6 35B-A3B as the primary route. Current production runs SGLang+MTP at 60-70 t/s. Theoretical memory-bandwidth ceiling on the same hardware is **180 t/s on RTX PRO 6000** and **24 t/s on DGX Spark single-batch**, **vLLM's generic implementation only realizes ~40% of that**. This document specifies a **purpose-built single-model inference engine** that exploits:

1. **Hardware-specific kernels** (Triton + CUTLASS, no generic abstractions)
2. **Custom asymmetric quantization** (NVFP4 routed experts + FP8 critical path)
3. **Training-data-informed expert prefetching** (we own the LoRA+pruning data)
4. **Disk-backed KV cache with SHA1 prefix matching** (Lynn brain agent prompts repeat 99%+)

**Performance target:** **80-150 t/s on Spark (vs 60-70 today), 250-400 t/s on RTX PRO 6000** (vs ~80 t/s vLLM today). 6-9 weeks to production. Goes hand-in-hand with the LoRA + expert pruning training pipeline already in flight.

---

## 1. Goals & Non-Goals

### Goals (must)

- **G1:** Serve **only Qwen 3.6 35B-A3B** (and its LoRA-merged + pruned variants) on Blackwell sm_12x. Single-model lock-in by design.
- **G2:** Achieve **≥ 100 t/s on DGX Spark** (single batch, short context) — 50% improvement over current SGLang+MTP baseline.
- **G3:** Achieve **≥ 250 t/s on RTX PRO 6000** when that hardware lands.
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

**Realistic targets (with kernel + cache overhead):**

| Target | Spark | RTX PRO 6000 |
|---|---|---|
| MVP (FP8 only, no MoE prefetch) | **80 t/s** | **180 t/s** |
| Phase 4 (asymmetric quant + prefetch) | **120 t/s** | **300 t/s** |
| Stretch (with MTP-style speculation) | **150 t/s** | **400 t/s** |

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
| **5** RTX PRO 6000 port | 1 day after hw arrives | ≥ 250 t/s | < 200 t/s → kernel optimization round 2 |
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

## 14. Decision Log

| Date | Decision | Rationale |
|---|---|---|
| 2026-05-08 | Lock to Qwen 3.6 35B-A3B (not generic) | Vertical integration with LoRA + pruning training; ToolAbstain shows it's the highest-scoring local model already |
| 2026-05-08 | Blackwell sm_12x only | Spark + RTX PRO 6000 both Blackwell; no need for Turing/Ampere/Ada |
| 2026-05-08 | Triton + CUTLASS hybrid (not pure CUDA) | Triton is faster to develop and Blackwell-tuned; CUTLASS for FP8/FP4 GEMM templates |
| 2026-05-08 | Asymmetric mixed precision (NVFP4 experts + FP8 critical path) | Memory savings without compromising attention/router quality |
| 2026-05-08 | Disk KV cache with SHA1 prefix | Direct port of antirez ds4 idea; perfect fit for Lynn brain agent workload |
| 2026-05-08 | Phase 1 = Triton kernel logits-diff spike | Hard go/no-go gate before committing to from-scratch path |

---

*Draft v0.1 · 2026-05-08 · pending Phase 0 baseline data + Phase 1 spike validation*
