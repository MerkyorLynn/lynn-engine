# Lynn Engine

> **Custom inference engine for Lynn 27B-A3B NVFP4 on NVIDIA Blackwell.**
> This is a narrow, vertical engine for Lynn's own variable-pruned MoE + NVFP4 artifact. The goal is not to become another general serving framework; the goal is to make one model family fast, understandable, and production-ownable on R6000 / Spark-class Blackwell hardware.

[阅读中文版](README.md) · [Strategy](docs/STRATEGY.md) · [Architecture](docs/DESIGN.md)

[![commits](https://img.shields.io/github/commit-activity/m/MerkyorLynn/lynn-engine)](https://github.com/MerkyorLynn/lynn-engine/commits/main)
[![license](https://img.shields.io/badge/license-TBD-orange)](.)

## Current status (2026-05-15)

Lynn engine has moved from "Qwen 35B architecture bring-up" to an independent runtime for the **final Lynn 27B NVFP4 base**:

| Item | Status |
|---|---|
| **27B final BF16** | ✅ Recovery step5000 final merged; structural validation PASS; greedy sanity PASS |
| **27B Lynn-native NVFP4** | ✅ 20G artifact produced and transferred to R6000; manifest integrity PASS |
| **Independent loader** | ✅ No vLLM / SGLang / TRT-LLM / llama.cpp dependency; reads safetensors + Lynn quant manifest directly |
| **6-prompt coherent smoke** | ✅ Chinese explanation / Python / RoPE-ALiBi / English arithmetic / tool JSON / long-context prompts all pass |
| **Current R6000 strict full path** | ✅ **103.44 tok/s** with packed NVFP4 MoE + opt-in native FP4 lm_head |
| **Serving replay ceiling** | ✅ **107.23 tok/s** 40-layer body graph, reproducible |
| **Next target** | production-stable 100+ TPS:remove BF16 resident shadows,finish native FP4 kernels and usability |

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
| Long target | <5 ms | >200 | native FP4 / larger fused blocks |

Current best R6000 environment:

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LYNN_PREFILL_WARMUP=1
export LYNN_LINEAR_ATTN_RECURRENT_BACKEND=triton_fused_prepare
export LYNN_LINEAR_ATTN_RECURRENT_INPLACE=1
export LYNN_MOE_IMPL=packed_nvfp4
export LYNN_QK_NORM_ROPE_BACKEND=triton
export LYNN_RMSNORM_GATED_BACKEND=triton
export LYNN_LINEAR_ATTN_INPROJ_FUSED_NATIVE_FP4=1
export LYNN_LINEAR_STATE_UPDATE=inplace
```

Measured final step5000 NVFP4:

```text
strict full path:      103.44 tok/s  (native FP4 lm_head opt-in)
serving replay/body:   107.23 tok/s  (40-layer graph ceiling)
BF16 lm_head path:     99.86 tok/s
quality smoke:         6/6 coherent + 2-token greedy sanity PASS
```

The native FP4 lm_head path is currently a deterministic/greedy opt-in optimization:6/6 prompt top-1 match,minimum top-20 overlap 15/20,and minimum logits cosine 0.9924. Sampling-heavy production traffic keeps the BF16 lm_head fallback until a larger parity gate is complete.

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

## Roadmap: R6000 first, Spark second

| Stage | Status | Goal |
|---|---|---|
| **P6** | done | 50 TPS cleared: resident graph smoke 63-66 TPS |
| **P7** | done | graph reuse / prewarm / RMSNormGated / serving env,66-68 TPS |
| **P8** | done | 78.8 TPS CUDA graph ceiling + 81 TPS compile spike |
| **P9** | done | packed NVFP4 active expert path, near-100 TPS |
| **P10** | current | native FP4 lm_head + 103 TPS full path, production-stable 100+ |
| **P11** | next | shared expert / grouped expert / larger fused kernels, target 200 TPS |

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
export LYNN_QK_NORM_ROPE_BACKEND=triton
export LYNN_RMSNORM_GATED_BACKEND=triton
export LYNN_LINEAR_ATTN_INPROJ_FUSED_NATIVE_FP4=1
export LYNN_LINEAR_STATE_UPDATE=inplace

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
