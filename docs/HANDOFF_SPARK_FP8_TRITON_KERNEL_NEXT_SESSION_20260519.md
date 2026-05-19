# Handoff · Spark FP8 fused Triton kernel deep work

**Date**: 2026-05-19 evening
**Branch**: `claude/phase-a-foundation-20260517` (lynn-engine)
**Successor session goal**: design + land fused MoE / dense FFN FP8 Triton kernels on Spark sm_121, lift single-stream decode TPS above current W4A16 baseline.

---

## 0. TL;DR for the next session

| What | Where it stands today | Where you take it |
|---|---|---|
| FP8 MMA HW peak on Spark sm_121 | **162 TFLOPS** (1.64× BF16 99), `torch._scaled_mm` real native | Treat as hard ceiling — design kernels to capture most of it |
| Naive `_scaled_mm` swap inside `_dense_ffn_forward` | **9B Dense 21 → 14.3 TPS (-32%)**, 35B MoE **38.96 → 9.17 TPS (-76%)** | Already proved this approach loses — DO NOT extend |
| W4A8 quality vs W4A16 | Spark fake-quant FFN-only **MMLU 75.80 / GPQA 43.94** ≈ W4A16 (76.00 / 42.93) | Use as baseline; do not re-eval after kernel ships |
| 35B-A3B Spark W4A16 NVFP4 single-stream | **38.96 TPS** (lynn-engine), llama.cpp Q4_K_M 207 TPS | Target ≥ **60 TPS** with fused kernel |
| 9B Dense Spark W4A16 NVFP4 single-stream | **21 TPS** (lynn-engine) | Target ≥ **30 TPS** |
| R6000 sm_120a | Native FP4 MMA → **108 TPS strict** W4A16 NVFP4 | Do NOT change R6000 path; keep FP4 native main course |

**Root cause of PoC loss**: per-call activation BF16→FP8 cast + tiny M=1 matmul + per-expert iteration × multiple `_scaled_mm` launches → Python/launch/cast overhead eats the 1.64× HW MMA advantage.

**Path to win** is unambiguous from the strategy memo (`reference_spark_fp8_w4a8_design_strategy_20260519.md`):
1. Offline NVFP4 → FP8 col-major repack (skip inference-time decode + layout)
2. Fused gate/up Triton kernel (1 launch / layer for 8 experts × 2 matmul)
3. Fused down + combine Triton kernel (1 launch / layer)
4. Resident-runner sm dispatch (sm_121 → FP8 fused / sm_120a → FP4 native)
5. (Stretch) CUDA graph capture for the whole decode step

---

## 1. Required reading before writing code

Open these in order. Don't skip — each one closes a class of mistakes the prior session already made.

### 1.1 Strategy + ISA reality (the "why we can't naive swap")
- ⭐⭐⭐ `~/.claude/projects/-Users-lynn-Downloads-Lynn/memory/reference_spark_fp8_w4a8_design_strategy_20260519.md` — 4-step path, what was already proved slow, scope of task #12
- ⭐⭐⭐ `~/.claude/projects/-Users-lynn-Downloads-Lynn/memory/reference_sm121_isa_capability_map.md` — Spark FP8 E4M3/E5M2 m16n8k32 ✅, **FP4/FP6/`kind::f8f6f4` ALL ptxas-reject** (you must not write FP4 MMA on Spark)

### 1.2 Quality baselines + canonical release numbers
- ⭐⭐⭐ `~/.claude/projects/-Users-lynn-Downloads-Lynn/memory/reference_qwen36_35b_release_numbers_20260519.md` — **AMBER 113-114 is BLACKLISTED** (P37 drift), strict default is **108 TPS** on R6000. Don't quote AMBER.
- ⭐⭐ `lynn-engine/reports/qwen35_9b/SPARK_QWEN35_9B_W4A8_VS_W4A16_QUALITY_REPORT_20260519.md` — confirms fake-quant W4A8 ≈ W4A16, so kernel work won't regress quality

### 1.3 Mainline framing
- ⭐⭐⭐ `~/.claude/projects/-Users-lynn-Downloads-Lynn/memory/project_mainline_pivot_to_9b_dense_20260519.md` — 9B Dense is now the **primary** target (no MoE complexity, Q4_K_M already proves model is fast on R6000, NVFP4 quality wins GPQA +5.56pp vs Q4_K_M). 35B is down-prioritized.
- ⭐⭐⭐ `~/.claude/projects/-Users-lynn-Downloads-Lynn/memory/project_lynn_4quadrant_release_matrix_20260518.md` — 4-cell ship requires both 9B and 35B in both Q4_K_M and NVFP4 quadrants

### 1.4 Existing PoC code (read before writing — don't recreate)
- `lynn-engine/engine/full_forward.py:186` — `_dense_ffn_forward` with `LYNN_W4A8_NATIVE_FP8=1` env (the naive PoC that lost)
- `lynn-engine/engine/moe_optimized.py` — `_expert_ffn` + `moe_forward_decode_optimized` (35B MoE site; naive native FP8 PoC also wired here as baseline)
- `/tmp/test_fp8_v3.py` (on Spark) — 162 TFLOPS HW peak measurement script

---

## 2. Subtasks A–E (execute in order; don't merge)

### A. Offline NVFP4 → FP8 repack tool

**Path**: `lynn-engine/scripts/spark_pack_nvfp4_to_fp8.py`

**Input**: Lynn NVFP4 packed weights (E2M1 mantissa + BF16 per-16 scale), as currently shipped at `/home/merkyor/models/<model>/`.

**Output**: a sibling file (e.g., `<weight>.fp8e4m3.safetensors`) containing
- FP8 E4M3 packed weight tensor, **col-major / TN-friendly** layout for `_scaled_mm`
- per-row OR per-tensor FP32 scales (decide empirically — start per-tensor, escalate to per-row if cos drops)

**Math**:
- Dequant NVFP4 → BF16 (existing path) → quantize to FP8 E4M3 with absmax-derived scale → re-pack
- LUT for E2M1 → FP8 may be lossless; verify

**Acceptance**:
- `dequant(fp8_repacked) cos > 0.999` vs `dequant(original NVFP4)` per-tensor
- Total size on disk ≤ 1.05× NVFP4 (don't blow up)
- Smoke: load packed file, run a known prompt through MoE/dense path with `LYNN_W4A8_REPACKED=1`, output coherent

**Why offline matters**: removes per-inference NVFP4→FP8 cast + layout transform, which is the biggest cycle-killer on M=1 decode.

---

### B. Fused gate/up Triton kernel (MoE + Dense)

**Path**: `lynn-engine/triton_kernels/spark_moe_fp8_fused_gate_up.py`

**Variants**:
- `fused_gate_up_moe`: one launch per layer for 8 experts × 2 matmul (gate, up) + SwiGLU
- `fused_gate_up_dense`: same kernel pattern but single expert (9B Dense)

**Kernel design points**:
- Input: BF16 hidden state (M=1 or batched), FP8 packed weight, per-row/tensor scales
- Inside: per-call BF16 → FP8 activation cast (try absmax dynamic + cached static scale variants)
- Use `tl.dot` with `out_dtype=tl.float32`, FP8 inputs via Triton FP8 dtype path
- Concatenated `[W_gate; W_up]` GEMM → split → SwiGLU → return BF16 inter
- For MoE: select active experts at launch boundary, dispatch tokens to slot via gather (NOT per-expert separate launches)

**Acceptance**:
- Numerical: cos > 0.998 vs BF16 reference SwiGLU on 1k random hidden states
- Perf: single-launch latency ≤ **0.8 ms / layer / token** on Spark sm_121 (current naive path ~5+ ms)

---

### C. Fused down + combine Triton kernel

**Path**: `lynn-engine/triton_kernels/spark_moe_fp8_fused_down_combine.py`

**Goal**: collapse the down_proj + expert output combine (for MoE: weighted sum across selected experts) into **one launch per layer**.

**Why combine matters**: in MoE decode, the per-expert down_proj + the routing-weighted accumulation are separate today → multiple launches. Combine kills that.

**Acceptance**:
- Numerical: cos > 0.998 vs current path
- Perf: ≤ **0.5 ms / layer / token**

---

### D. Resident-runner integration (sm dispatch)

**Path**: `lynn-engine/engine/resident_runner.py`

**Decision tree at startup** (per device):
```
if sm == (12, 1):                 # Spark GB10
    if FP8 repack present:
        use fused FP8 path (kernels from B + C)
    else:
        fallback BF16 dequant
elif sm == (12, 0):               # R6000 Blackwell with FP4
    use native FP4 path (existing, 108 TPS strict)
else:
    fallback BF16
```

**Env vars to add** (keep consistent with existing `LYNN_*` naming):
- `LYNN_SPARK_FP8_FUSED=1` — opt-in for Spark fused path
- `LYNN_SPARK_FP8_STATIC_SCALE=1` — use cached static activation scale instead of per-call absmax

**Don't**:
- Don't split artifacts (one weight file, runtime dispatches)
- Don't change R6000 FP4 path — it's the main course there
- Don't introduce a new Python class hierarchy; reuse current `_expert_ffn` / `_dense_ffn_forward` call sites

**Acceptance**:
- All current R6000 strict-default tests still pass unchanged (108 TPS, P37 strict, AMBER blacklisted)
- Spark single-stream 35B-A3B ≥ **60 TPS**
- Spark single-stream 9B Dense ≥ **30 TPS**

---

### E. (Stretch) CUDA graph capture

**Goal**: capture the whole decode step under CUDA graph after B+C+D land.

**Likely site**: extend the slot pool pattern (`_capture_full_attn_layer_graph_slot` already exists for R6000 work — re-use ABI).

**Don't start E until A–D are GREEN.** It's a multiplier; without fused kernels it just hides launch cost behind another launch.

**Acceptance** (if attempted):
- Same numerical cos as eager
- Additional +20-30% TPS on top of B+C+D
- Graceful fallback if capture fails (no hangs)

---

## 3. Spark environment state (as of handoff)

- Container `lynn-engine-35b-w4a8-native` running the PoC; can `docker kill` and restart fresh
- Weights:
  - `/home/merkyor/models/qwen35-9b-*` (9B Dense, W4A16 NVFP4 + Q4_K_M + BF16)
  - `/home/merkyor/models/qwen36-35b-a3b-*` (35B MoE, W4A16 NVFP4 + Q4_K_M)
- Evaluator: `/home/merkyor/lynn-engine/scripts/` (thinking-on bug is **patched** — don't reintroduce)
- GPU mem free: ~109 GiB on Spark (unified mem total ~119 GiB) — plenty of headroom
- CUDA 13.0 / Torch 2.9.1 / Triton 3.5.1 (per `reference_lynn_hardware_specs_canonical.md`)
- `torch._scaled_mm` E4M3 m16n8k32 confirmed real HW MMA (not emulation)

---

## 4. Success criteria for handoff completion

A successful session ends with all of:
- [ ] 35B-A3B Spark single-stream decode ≥ **60 TPS** (currently 38.96 W4A16)
- [ ] 9B Dense Spark single-stream decode ≥ **30 TPS** (currently 21 W4A16)
- [ ] P37 70-prompt strict exact-match: PASS at parity with W4A16 (math is equivalent — any drift is a kernel bug)
- [ ] R6000 strict-default tests unchanged
- [ ] One artifact per model (no Spark-vs-R6000 weight split)
- [ ] Branch pushed; one short report under `lynn-engine/reports/spark/` describing kernel + numbers

---

## 5. Hard prohibitions (these will cost the session)

- ❌ **Do not** quote AMBER 113-114 TPS for 35B — they have P37 drift and 70-set failures. Use **108 strict default**.
- ❌ **Do not** re-introduce thinking-on bug — evaluator already patched.
- ❌ **Do not** extend the naive `_scaled_mm` swap pattern — it lost (-32% / -76%) and the path forward is fused, not naive.
- ❌ **Do not** add FP8 kernels on R6000 — R6000's main course is native FP4 (108 TPS strict). FP8 is Spark-only.
- ❌ **Do not** split artifacts (one weight file per model, runtime dispatches by sm).
- ❌ **Do not** push AGPL code (Atlas etc.) into lynn-engine — clean-room only.
- ❌ **Do not** start subtask E (CUDA graph) before A–D are GREEN.

---

## 6. Cross-references

- 4-quadrant release matrix (must fill all 4 cells before ship): `project_lynn_4quadrant_release_matrix_20260518.md`
- Spark TPS baseline triple: `project_spark_single_stream_tps_baseline_20260518.md` (BF16 30.14 / Q4_K_M 69.77 / W4A16 NVFP4 38.96)
- 35B AMBER blacklist reasoning: `reports/qwen36_35b/QWEN36_35B_A3B_NVFP4_PROGRESS_20260519.md`
- Stream B full-attn layer graph reuse (reusable ABI for stretch goal E): `docs/STREAM_B_FULL_ATTN_LAYER_GRAPH_REUSE_SPEC_20260518.md`
- 9B / 35B quality cross-section (no need to re-eval after kernel ship): `project_qwen35_9b_q4km_thinking_baseline_20260519.md`
