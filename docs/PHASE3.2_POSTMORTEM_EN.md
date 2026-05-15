# Phase 3.2 Postmortem — Why 78.6% Match Rate Is a Bug

> **TL;DR**: bmm/indexed_bmm MoE fast paths showed `1.85×` decode speedup and
> 100% single-prompt token match against the `optimized` ground truth. A
> multi-prompt N=14 gate then revealed `11/14 (78.6%)` exact-match — three
> prompts diverge from the first divergent token onward. Root cause is a
> physical limitation in the cuBLAS algorithm selection between
> `gemm` (used by `F.linear`) and `gemmStridedBatched` (used by `torch.bmm`).
> Default impl remains `optimized`. bmm and indexed_bmm are now **opt-in only**
> via `LYNN_MOE_IMPL` env var.

This document is the engineering record for the new
[Universal review rule in `docs/STRATEGY.md`](STRATEGY.md):

> Single-prompt PASS does NOT endorse any default-impl change.
> All default switches require N≥14 multi-prompt exact-match gate to pass.

---

## 1. Background — Phase 3.2 three-tier MoE optimization

Lynn engine MoE expert FFN has three implementations in
[`engine/moe_optimized.py`](../engine/moe_optimized.py):

| Tier | Name | Algorithm | Status |
|---|---|---|---|
| 3.2.1 | `optimized` | active-experts loop with `F.linear` per expert | DEFAULT |
| 3.2.2 | `bmm` | stack 8 active expert weights → 3× `torch.bmm` | opt-in |
| 3.2.2.5 | `indexed_bmm` | pre-stack all 256 experts once → indexed bmm | opt-in |

The bmm design is intuitive: at decode time `T=1` the router selects K=8
experts; instead of 8 sequential `F.linear` calls, stack their weights into
`[K, intermediate, hidden]` and do 3 batched matmul calls (gate / up / down).

```python
# moe_optimized.py::moe_forward_decode_bmm  (excerpt)
gate_stack = torch.stack([w[f"mlp.experts.{e}.gate_proj.weight"] for e in expert_ids])
up_stack   = torch.stack([w[f"mlp.experts.{e}.up_proj.weight"]   for e in expert_ids])
down_stack = torch.stack([w[f"mlp.experts.{e}.down_proj.weight"] for e in expert_ids])

h_broadcast = h_flat.unsqueeze(0).expand(K, -1, -1)
gate_out = torch.bmm(h_broadcast, gate_stack.transpose(-1, -2))
up_out   = torch.bmm(h_broadcast, up_stack.transpose(-1, -2))
inter    = F.silu(gate_out) * up_out
ffn_out  = torch.bmm(inter, down_stack.transpose(-1, -2))
```

### 1.1 Initial smoke results — looked production-ready

```
Prompt: "The capital of France is" (max_new=10)

  optimized:    [11751, 11, 264, 3177, 34756, 364, 1141, 25438, 57902, 1680]
  bmm:          [11751, 11, 264, 3177, 34756, 364, 1141, 25438, 57902, 1680]
  indexed_bmm:  [11751, 11, 264, 3177, 34756, 364, 1141, 25438, 57902, 1680]
  ✓ 10/10 EXACT MATCH

Decode t/s:
  optimized:    12.84 t/s
  bmm:          23.56 t/s   (1.83×)
  indexed_bmm:  24.79 t/s   (1.93×)
```

At this point a (now-reverted) decision to set `LYNN_MOE_IMPL=bmm` as default
was made and STRATEGY.md was edited to mark Phase 3.2.2 as
"production-ready baseline."

---

## 2. Multi-prompt N=14 gate revealed the false positive

A routine multi-prompt verification — 14 diverse prompts (English / Chinese /
code / math / short / long), 8 tokens each, 3 impls compared:

| # | Prompt | optimized | bmm | indexed_bmm | Match? |
|---|---|---|---|---|---|
| 1 | `The capital of France is` | `[11751, 11, 264, 3177, ...]` | same | same | ✓ |
| 2 | `What is the speed of light` | ✓ | ✓ | ✓ | ✓ |
| 3 | `用一句话解释什么是 transformer` | ✓ | ✓ | ✓ | ✓ |
| 4 | `def fibonacci(n):` | ✓ | ✓ | ✓ | ✓ |
| **5** | **`2+2=`** | **`[20, 369, 264, 220, 16, 24, 23, 18]`** | **`[20, 271, 248068, 198, 8160, 579, 264, 7047]`** | **same as bmm** | **✗** |
| **6** | **`Python is`** | **`[264, 5243, 15019, 3992, 421, 369, 13177, 1429]`** | **`[264, 2172, 3992, 364, 3604, 795, 6157, 11]`** | **same as bmm** | **✗** |
| 7 | `Write a haiku about spring` | ✓ | ✓ | ✓ | ✓ |
| 8 | `今天天气` | ✓ | ✓ | ✓ | ✓ |
| 9 | `import torch` | ✓ | ✓ | ✓ | ✓ |
| 10 | `Hello world` | ✓ | ✓ | ✓ | ✓ |
| 11 | `The largest planet is` | ✓ | ✓ | ✓ | ✓ |
| **12** | **`I love eating`** | **`[3487, 13, 198, 40, 2854, 11834, 3487, 13]`** | **`[3487, 11, 694, 353, 1459, 914, 1040, 16725]`** | **same as bmm** | **✗** |
| 13 | `The quick brown fox` | ✓ | ✓ | ✓ | ✓ |
| 14 | `import numpy as` | ✓ | ✓ | ✓ | ✓ |

Final stats:

```
Multi-prompt gate (N=14, 8 tokens each):
  optimized   ground truth        12.53 t/s
  bmm         11/14 (78.6%) ✗   23.24 t/s
  indexed_bmm 11/14 (78.6%) ✗   24.74 t/s

Mismatch prompts (bmm and indexed_bmm fail the same 3):
  - 2+2=
  - Python is
  - I love eating
```

Two important properties of the failure:

1. **Token 1 matches**, but token 2+ completely diverge. Not a "single-logit
   off-by-one" — the router selected a different expert, and from then on
   the entire sequence cascades.
2. **bmm and indexed_bmm fail the exact same 3 prompts** — the bug lives in
   the shared batched-matmul path, not in indexed_bmm's pre-stack logic.

---

## 3. Pattern analysis — multi-meaning prompts are the canary

Looking at which prompts pass vs which fail:

| Passing prompt | Continuation pattern |
|---|---|
| `The capital of France is` | → "Paris" near-deterministic |
| `def fibonacci(n):` | → code mode, top-K logits well separated |
| `import torch` / `import numpy as` | → standard code completion |
| `今天天气` | → standard Chinese continuation |
| `Hello world` | → exclamation / period |

| Failing prompt | Continuation pattern |
|---|---|
| `2+2=` | → "4" / "Let me think" / "解" — multiple top-K candidates close in logit |
| `Python is` | → broadly descriptive, top-K crowded |
| `I love eating` | → ambiguous emotion / food, top-K crowded |

**Pattern**: passing prompts have a **deterministic continuation with
well-separated top-K logits**. Failing prompts have **multi-meaning
continuations with crowded top-K logits**.

This strongly suggests the numerical ε difference does not affect router
choice when top-K logits are spread out, but **flips selection when top-K
candidates are within ε of each other**. Multi-meaning prompts function as
canaries for ε-level numerical drift through the network.

---

## 4. The first hypothesis was wrong — FP32 accumulator promote

The natural hypothesis: bmm uses BF16 accumulator and the precision is
insufficient. Patch:

```python
# Proposed fix:
h_broadcast_fp32 = h_broadcast.float()
gate_w_t_fp32    = gate_stack.transpose(-1, -2).float()
gate_out = torch.bmm(h_broadcast_fp32, gate_w_t_fp32)              # FP32 accumulator
inter    = F.silu(gate_out) * up_out_fp32
ffn_out  = torch.bmm(inter, down_w_t_fp32).to(h.dtype)             # cast back to BF16
```

Smoke `2+2=` with patch applied:

```
optimized:        [20, 369, 264, 220, 16, 24, 23, 18]
bmm (FP32 fix):   [20, 271, 248068, 198, 8160, 579, 264, 7047]
                  ^ identical to BF16 baseline — fix had zero effect
```

**The patch was a no-op** because:

> cuBLAS already uses an FP32 accumulator internally when handed BF16 inputs.
> This is the standard behavior for BF16 Tensor Core GEMM. Promoting inputs
> to FP32 before `torch.bmm` is redundant — the accumulator was already FP32.

In other words `torch.bmm(BF16, BF16) → BF16` already runs:

- Input dtype: BF16
- Internal accumulator: FP32 (Tensor Core auto-promote)
- Output dtype: BF16

`.float()` cast adds nothing.

---

## 5. Real root cause — cuBLAS algorithm selection

Cross-checking which cuBLAS kernels each impl actually invokes:

| Impl | PyTorch op | cuBLAS kernel |
|---|---|---|
| `optimized` | `F.linear(x, w)` (8 sequential calls) | `gemm` |
| `bmm` | `torch.bmm(x_batch, w_batch)` (1 batched call) | `gemmStridedBatched` |

Two **different** cuBLAS kernels. cuBLAS does not guarantee that
`gemm` run 8 times sequentially produces LSB-identical output to
`gemmStridedBatched` run once over the same 8 problems, even with FP32
accumulators on both sides:

1. **Different tile sizes** — `gemmStridedBatched` chooses tile sizes that
   account for the batch dimension; standalone `gemm` picks differently.
2. **Different reduction order** — sum-reductions inside a GEMM are
   sequenced by the kernel block schedule. Different schedules → different
   reduction orders.
3. **FP addition is not associative** — `(a + b) + c ≠ a + (b + c)` even in
   FP32. ε is small but non-zero; in BF16 with 7-bit mantissa, ε ≈ 8e-5.

So even though both kernels use FP32 accumulators, the **reduction order
differs** → ε-level differences in output.

### 5.1 Why a single-layer ε difference becomes a token divergence

A single-layer logit comparison shows `<10%` relative difference, well within
Phase 3.1 alignment tolerance. But Lynn engine's Qwen3.6-35B-A3B has 40
layers, and each layer's MoE output is added back into the residual stream.

ε accumulates through the residual stream:

```
Layer 1 hidden state diff:    ~1e-4
Layer 5:                       ~5e-4
Layer 10:                      ~2e-3
Layer 20:                      ~1e-2
Layer 40:                      ~5e-2
```

By layer 40, the diff is in the same magnitude as the gap between top-K
router logits on borderline prompts. The router picks a different expert.
That expert produces very different output. Token cascade diverges.

This matches the observed timing perfectly:

```
Prompt: 2+2=
optimized first token:   20 ('5')   ← matches (router gap large enough)
bmm first token:         20 ('5')   ← matches
optimized second token:  369 (' is')   ← divergence starts here
bmm second token:        271 ('\n\n')  ← different expert chosen
```

---

## 6. Physical conclusion — bmm cannot byte-match F.linear in BF16

> **bmm and F.linear are not byte-exact under BF16. This is a property of
> cuBLAS kernel selection, not a bug in either implementation.**

Evidence supporting this:

- Not an accumulator precision issue (both use FP32 accumulators).
- Not a broadcast vs explicit-repeat issue
  (verified separately: SDPA `enable_gqa=True` produces byte-exact output
  to the prior `repeat_interleave` path — see commit message for
  `engine/incremental_decode.py`).
- It's the `gemm` vs `gemmStridedBatched` reduction-schedule difference.

The fix direction must change. The original goal "make bmm numerically equal
to optimized" is **physically unreachable** under BF16 + cuBLAS.

The fix actually applied:

1. Default `LYNN_MOE_IMPL` reverts to `optimized` (the byte-exact path).
2. bmm and indexed_bmm remain available as **opt-in fast paths** for users
   who can tolerate the prompt-variance trade-off.
3. Phase 3.2.2 is **not** marked production-ready in STRATEGY.md.

```bash
# Default: byte-exact, ground truth, multi-prompt 14/14 PASS
python engine/full_forward.py --prompt "..." --mode incremental

# Opt-in fast path: ~1.85× speedup, but ~22% prompt fail rate
LYNN_MOE_IMPL=bmm python engine/full_forward.py --prompt "..." --mode incremental
```

---

## 7. Universal review rule (new in STRATEGY.md)

> **Single-prompt PASS does not endorse any implementation.** All
> default-impl switches, production-ready labels, and merge-to-main decisions
> must be gated by **multi-prompt exact-match** results.
>
> **Gate minimum requirements**:
> - N ≥ 14 diverse prompts (mix of English / Chinese / code / math /
>   multi-meaning continuations)
> - Exact-match rate = 100% (i.e. `mismatches_count = 0`)
> - Any borderline-prompt mismatch is an immediate overall FAIL — partial
>   credit not allowed
>
> **Watchdog rules**:
> - Check `exact_match_rate < 1.0` or `mismatches_count > 0` directly → FAIL
> - **Do not trust** any `"verdict": "PASS"` field — it can be mislabeled or
>   missing
> - **Do not** use single-prompt benchmark data to drive default-impl changes

This rule originates from the bmm 78.6% incident. It is now codified in
[`docs/STRATEGY.md`](STRATEGY.md) under section "⚠️ Universal review rule".

The class of failure is the same as the **self-consistent bug** that Codex
review #2 warned about: when test code and implementation share an
assumption that is wrong, the test reaches a fixed-point that confirms the
implementation, but the fixed-point is not the truth. Here the bmm output
self-consistently confirmed itself on a single prompt; only N=14 with
multi-meaning prompts broke the alignment.

---

## 8. Test design lesson — multi-meaning prompts are required

If the verification set contained only deterministic-continuation prompts
(`The capital of France is` / `def fibonacci(n):` / `import torch`), the
gate would have stayed at 100% PASS regardless of N. The bug would have
shipped, and `2+2=` in production would have broken.

A correctness gate test set must include:

```
Required prompt categories:
  Math           "2+2=" / "5+7=" / "sqrt(16)="
  Description    "Python is" / "AI 是" / "The meaning of life is"
  Emotion        "I love eating" / "I don't know what to"
  Short          "你好" / "Hi"

Insufficient on their own:
  Deterministic-only set — 100% PASS does not catch ε-level drift
```

The N=14 set used here is a minimum baseline. Future correctness work
(NVFP4, quantization-error harness) should expand to N≥20+ with explicit
borderline-prompt coverage.

---

## 9. Profiler analysis — GEMM is not the bottleneck

A separate profile run (10 decode steps, `torch.profiler`) shows:

```
Self CUDA time total:   126.86 ms  →  12.7 ms/step GPU compute
Wall clock (no profile):              43 ms/step  =  ~23 t/s (bmm)
Implied Python orchestration:         30 ms/step  ≈ 70% of wall
```

GPU-side breakdown:

| Category | Share | Time/step |
|---|---|---|
| `cuBLAS gemvx` (attention path) | 40.7% | 5.1 ms |
| `aten::cat` + `aten::copy_` | 28% | 3.6 ms |
| Elementwise `mul` + `add` | 15% | 1.9 ms |
| `aten::mm` + `aten::bmm` | 9% | 1.1 ms |
| Norm / router / silu / topk | 7% | 0.9 ms |

**MoE GEMM (`mm` + `bmm`) is only 9% of GPU time.** Even a perfect-fused
Triton MoE FFN kernel could only reclaim a fraction of those 1.1 ms/step.
Phase 3.3 (Triton MoE FFN) was previously planned as the next major
optimization; that priority is now retired in favor of work that targets
the actual bottleneck — Python orchestration.

A theoretical-upper-bound calculation (eliminate all `cat` + `copy_`):

```
GPU: 12.7 - 2.0 (cat) - 1.6 (copy_) = ~9 ms/step
+ Python orchestration: 30 ms/step (cannot be reduced by GPU-side patches)
= ~39 ms/step wall = ~25 t/s
```

So the GPU-side simple-patch ceiling is ~25 t/s. Breaking that requires
restructuring the Python orchestration layer (`_decode_layer` signature,
LynnInferenceState mutation pattern, CUDA Graph compatibility) — a separate
work track filed in STRATEGY.md.

---

## 10. Engineering takeaway

What looked like:

> **A 1.85× speedup with full single-prompt token match — production-ready.**

was actually:

> **A 78.6% multi-prompt match rate with a physically unfixable BF16 ε
> drift through 40 residual layers, surfacing only on multi-meaning prompts
> where the router top-K is crowded.**

The technical lessons are short. The engineering lessons are:

- **Single-prompt PASS does not endorse any implementation.**
- **Multi-meaning prompts are the canary** for ε-level numerical drift.
- **Test sets must include borderline cases by design**, not by accident.
- **A self-consistent test (test ↔ impl share assumption) is a worse failure
  mode than an obvious bug**, because it ships.

Default `LYNN_MOE_IMPL=optimized`. bmm and indexed_bmm are opt-in. The
Universal review rule in STRATEGY.md applies to all future default-impl
changes.

---

## 11. Cross-references

| Topic | Location |
|---|---|
| Universal review rule | [`docs/STRATEGY.md`](STRATEGY.md) §"⚠️ Universal review rule" |
| Phase 3.2.x repositioning | [`docs/STRATEGY.md`](STRATEGY.md) §"Phase 3.2.x 价值重新定位" |
| Avoidance guide (BF16 / cuBLAS / profiler / multi-prompt gate traps) | [`docs/AVOIDANCE_GUIDE_2026-05-10.md`](AVOIDANCE_GUIDE_2026-05-10.md) |
| `LYNN_MOE_IMPL` switch | [`engine/full_forward.py`](../engine/full_forward.py) `_decode_layer` |
| Pre-stack hook (indexed_bmm) | [`engine/full_forward.py`](../engine/full_forward.py) `generate_incremental` |
| SDPA `enable_gqa` patch | [`engine/incremental_decode.py`](../engine/incremental_decode.py) `decode_full_attn` |
| Single-file safetensors fallback | [`engine/loader.py`](../engine/loader.py) |
| Multi-prompt correctness harness | [`benchmarks/nvfp4_multi_prompt_correctness.py`](../benchmarks/nvfp4_multi_prompt_correctness.py) |

---

*Co-authored: Lynn, Codex, Claude. Tested on RTX PRO 6000 Blackwell sm_120,
CUDA 12.8, PyTorch 2.8.0+cu128, Qwen3.6-35B-A3B-FP8.*
