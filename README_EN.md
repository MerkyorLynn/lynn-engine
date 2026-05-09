# Lynn Engine

> **Custom inference engine for Qwen 3.6 35B-A3B on NVIDIA Blackwell.**
> From-scratch single-model engine — written to (a) understand every layer of the model, (b) eventually beat generic frameworks like vLLM at single-prompt latency for our specific workload.

[阅读中文版](README.md) · [Strategy](docs/STRATEGY.md) · [Architecture](docs/DESIGN.md)

[![commits](https://img.shields.io/github/commit-activity/m/MerkyorLynn/lynn-engine)](https://github.com/MerkyorLynn/lynn-engine/commits/main)
[![license](https://img.shields.io/badge/license-TBD-orange)](.)

## What's working today

✅ **End-to-end correctness validated** — Lynn engine generates **token-for-token identical output** to production vLLM on greedy decode.

```
prompt: "The capital of France is"
vLLM:   ' Paris, a city renowned'
Lynn:   ' Paris, a city renowned'   ← 5/5 token exact match
```

✅ **All 40 layers numerically verified**:
- 30 linear_attention (GatedDeltaNet) layers — bit-exact vs HF reference
- 10 full_attention layers — verified via end-to-end logits agreement
- Multi-prompt validation: 8 diverse prompts, 9.8/10 average top-K (10) overlap with vLLM

⚙️ **Phase 3.1 incremental decode shipped** — KV cache for full_attention + recurrent state cache for linear_attention.

⚙️ **OpenAI-compat HTTP server** — Brain can swap one URL to A/B test.

## Performance status

| Path | Latency / token | t/s | Status |
|---|---|---|---|
| Phase 2 brute-force | ~300 ms | 2-3 | shipped |
| **Phase 3.1 incremental decode** | **~200 ms** | **5** | **shipped (commit `1e2980b`)** |
| Phase 3.2 (active-experts + bmm + indexed) | target ~100-130 ms | 8-10 | code committed, untested |
| Phase 3.3 (Triton-fused MoE FFN) | target ~50 ms | 20 | scaffold + design done |
| Phase 3.4 (CUTLASS NVFP4 grouped) | target ~10-15 ms | 60-100 | future |
| vLLM SGLang+MTP baseline | ~14 ms | 60-70 | reference |

## Tutorials — read these even if you're not writing your own engine

In writing Lynn engine we collected the parts of Qwen 3.6 35B-A3B that **diverge from Llama / Qwen 2 in ways the docs don't tell you**. Six deep tutorials in [`tutorials/`](tutorials/):

| # | Topic | TL;DR |
|---|---|---|
| [01](tutorials/01_rmsnorm_one_plus_weight.md) | RMSNorm `(1.0 + w)` not `w` | Qwen 3 family RMSNorm has a +1 offset. Llama-style hits ~10x error. |
| [02](tutorials/02_rope_three_gotchas.md) | RoPE three gotchas | theta in `rope_parameters`, partial=0.25, GPT-NeoX half-split (not Qwen 2 even/odd) |
| [03](tutorials/03_attn_output_gate.md) | q_proj 2× per-head split | Must reshape per-head before chunk, or head_i_gate leaks into head_i_q |
| [04](tutorials/04_gated_delta_net.md) | linear_attention = GatedDeltaNet | Mamba-style chunk recurrence + delta rule + l2norm Q/K |
| [05](tutorials/05_three_invisible_bugs.md) | self-consistent bug postmortem | reference + lynn same-source-same-bug = passes wrong tests |
| [06](tutorials/06_moe_router_softmax_topk_order.md) | MoE router order + shared expert | softmax-all → topK → renormalize, with sigmoid-gated shared expert |

[`tutorials/posts/zhihu_qwen36_engine_postmortem.md`](tutorials/posts/zhihu_qwen36_engine_postmortem.md) is a single Zhihu-blog-style writeup of the highlights.

## Quick start (DGX Spark)

```bash
# 1. Convert FP8 → BF16 once (avoids HF FP8 deep-gemm metadata blocker)
docker run --rm --user 1000:1000 \
  -v /home/merkyor/models:/models \
  -v /tmp/lynn-engine:/work -w /work \
  nvcr.io/nvidia/vllm:26.03.post1-py3 \
  python3 engine/convert_fp8_to_bf16.py \
    --src /models/Qwen3.6-35B-A3B-FP8 \
    --dst /models/Qwen3.6-35B-A3B-BF16

# 2. Stop vLLM (Lynn needs ~67 GB resident BF16; fp8 path can co-resident)
docker stop vllm-qwen35a3b

# 3. Run incremental decode demo
docker run --rm --gpus all --ipc=host --user 1000:1000 \
  -v /home/merkyor/models:/models \
  -v /tmp/lynn-engine:/work -w /work \
  -e PYTHONPATH=/work \
  nvcr.io/nvidia/vllm:26.03.post1-py3 \
  bash -c "pip install -q --user transformers==5.8.0 && \
           python3 engine/full_forward.py \
             --prompt 'The capital of France is' \
             --max-new 5 --mode incremental"

# 4. (Optional) Start the OpenAI-compat HTTP server
docker run -d --rm --gpus all --ipc=host --user 1000:1000 \
  -v /home/merkyor/models:/models \
  -v /tmp/lynn-engine:/work -w /work \
  -p 127.0.0.1:18099:18099 \
  -e PYTHONPATH=/work \
  nvcr.io/nvidia/vllm:26.03.post1-py3 \
  bash -c "pip install -q --user transformers==5.8.0 fastapi uvicorn && \
           python3 -m server.openai_http \
             --model /models/Qwen3.6-35B-A3B-FP8 \
             --host 0.0.0.0 --port 18099"
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
  README.md + 6 markdown tutorials + zhihu post
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
