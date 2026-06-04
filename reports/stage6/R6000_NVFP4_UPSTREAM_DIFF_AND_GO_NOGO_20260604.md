# R6000 NVFP4 FP4-MMA — Upstream-vs-Lynn diff + GO/NO-GO gate (2026-06-04)

> **Audience: codex (R6000/Stage-6 line).** Lead/owner evidence-based verdict. Read before the next R5 gate.
> Bottom line: **NVFP4 is commoditizing INTO upstream llama.cpp. Stop building it as a private Lynn moat.
> The ONLY defensible R6000 investment is the grouped-MoE FP4-MMA gap — under a hard "bank real speed or stop" gate.**

---

## TL;DR
- Upstream llama.cpp has **merged the NVFP4 format/CPU/convert** and has an **open dense Blackwell FP4-MMA PoC (#23572, +40% PP)**. NVFP4 is becoming a first-class llama.cpp quant type, community-staffed (74 nvfp4 PRs).
- Lynn R6000 has banked **ZERO speed** so far: `banked_kernel_speed=false`, grouped-MoE POC `UNIMPLEMENTED`, and the first speed candidate **R5-C4 was just rejected (trace-derived)**.
- **VERDICT: NO-GO on "own the NVFP4 kernel/engine."** The single remaining unique angle = **grouped-MoE FP4-MMA** (upstream #23572 is dense-only). Pursue ONLY under the hard gate in §5; design it to be **upstreamable**, not private.

---

## 1. Upstream llama.cpp NVFP4 state (verified 2026-06-04, ggml-org/llama.cpp)
- **Format MERGED**: `GGML_TYPE_NVFP4 = 40`, `GGML_FTYPE_MOSTLY_NVFP4 = 26`, `block_nvfp4 { uint8_t d[4] (UE4M3 per-16 sub-scales); uint8_t qs[32] (E2M1) }` (block=64, per-16 sub-scale — this **IS** Lynn's NVFP4 layout). CPU quantize/dequantize/vec_dot implemented (12 hits in `ggml-quants.c`). gguf-py + `llama-quantize` mapping landing (#22897).
- **Gemma4 MERGED**: #23682 (`convert: Gemma4ForCausalLM`), #22804 (`Gemma4 NvFp4 convert→gguf fixes`), #21971 (`model: NVFP4 tensors for Gemma4`). → text-Gemma4 + its NVFP4 already convertible/runnable upstream. (Correction to an earlier Lynn note that "llama.cpp has no gemma4": the **unified-multimodal** `Gemma4Unified…` variant is the remaining gap, not text Gemma4.)
- **CUDA Blackwell FP4-MMA = NOT settled, DENSE-only**: #23572 `CUDA: native 4-bit float quant (Blackwell PP +40%)` — **OPEN PoC** (small, "starting point… encourage iteration"), **+40% PP (prefill) dense** on **RTX 5090 (sm_120)**, decode (TG) ≈ flat, quant quality mediocre (`N4_0` KLD ~between Q3_K and Q4_0). Earlier #21896 (`ggml-cuda: Blackwell native NVFP4`) is **CLOSED**. **No MoE FP4-MMA CUDA exists upstream.**
- Other active: AVX dot #23961, Metal #20456, RISC-V #23402, MSE-quality #23692.

## 2. Lynn R6000 FP4-MMA state (from `stage6_evidence_ledger`, 2026-06-04 ~21:42)
| gate | status |
|---|---|
| `r6000_fp4_mma_census` | BANKED (bring-up + public-kernel census) |
| `r6000_grouped_moe_fp4_mma_poc_contract` | **CONTRACT_READY_UNIMPLEMENTED** |
| `r5a_layout_bridge` (per-16 NVFP4 → block-scaled FP4) | DIAGNOSTIC_BANKED |
| `r5b_e8m0_repack` | **CLOSED_NEGATIVE** |
| `r5c_cutlass_ue4m3_census` / `r5c1` numeric / `r5c2*` shape+slot | BANKED (census / numeric-smoke / contract) |
| `R5-C4` full-active-MoE prefill speed A/B | **first speed candidate REJECTED (trace-derived, 21:32)** |

Lynn's own verdict (R5-C4 report): *"trace-only speed evidence; no grouped-MoE FP4-MMA POC, decode TPS, server behavior, RC quality, or default promotion is banked. `banked_kernel_speed=false`."*

## 3. Head-to-head
| Axis | Upstream llama.cpp | Lynn R6000 |
|---|---|---|
| NVFP4 format / CPU / convert | ✅ MERGED (commoditized) | private format — now **redundant** w/ upstream |
| Dense FP4-MMA CUDA | 🔵 #23572 PoC, **+40% PP** measured (RTX 5090) | not the focus |
| **Grouped-MoE FP4-MMA** | ❌ **gap** (upstream dense-only) | 🔴 UNIMPLEMENTED; first speed candidate rejected |
| **Banked speed win** | +40% PP (real, even if PoC) | **ZERO** (`banked_kernel_speed=false`) |
| Quant quality | N4_0 mediocre (~Q3_K-Q4_0 KLD) | NVFP4 E4M3-per-16 higher (MMLU 84.40 / GPQA 49.49) |
| Hardware | sm_120 (5090) | sm_120 (RTX PRO 6000) — same family |

## 4. Strategic finding
1. **NVFP4 format + dense kernel = upstream's now.** Re-deriving them on R6000 = waste.
2. **Lynn's only unique angle = grouped-MoE FP4-MMA + higher-quality E4M3-per-16** — but it's **undelivered** (zero banked speed after a dozen R5 gates; R5-C4 rejected).
3. Even if delivered, the leverage move is to **contribute it upstream** (#23572's author explicitly invites iteration) — private = no model zoo, maintenance burden, and likely subsumed by the community.

## 5. GO / NO-GO GATE (hard — apply at the next R5 checkpoint)
**GO (keep investing R6000) ⟺ ALL of the following are banked in the next focused push:**
1. A **grouped-MoE** FP4-MMA prefill kernel that is **RC-exact** (cos≈1 / token-coherent vs the dequant→bf16 reference), AND
2. a **real measured e2e prefill A/B** (NOT trace-derived) showing it **beats the current path on R6000 by a meaningful margin**, AND
3. the win is in the **MoE regime that upstream #23572 does NOT cover** (dense is upstream's; don't compete there).

**NO-GO (stop R6000) if any of:**
- the next **2–3 speed candidates** get rejected like R5-C4 (the "many gates, zero speed" pattern persists), OR
- the kernel can't reach RC-exact, OR
- upstream lands a MoE FP4-MMA path first.

**On NO-GO:**
- (a) Package the R5-A layout-bridge + R5-C CUTLASS-UE4M3-ABI learnings as a **contribution proposal to upstream #23572** (clean-room, MIT).
- (b) **Stop the private R6000 grind**; use upstream llama.cpp NVFP4 once the CUDA path matures.
- (c) Redirect R6000 hours + attention to the durable moat: the **product (desktop agent)** and short-term Qwen-specific optimization.

## 6. Hard rules
- **Census / contract / ABI / numeric-smoke ≠ a speed result.** Only a real e2e A/B prefill speedup counts as GO evidence. (You already enforce this — keep it; it's why R5-C4 was correctly rejected.)
- **Baseline = the right thing**: compare vs the current dense/best path AND vs upstream #23572's +40% PP — not vs an unoptimized starting point.
- **Build to upstream, not to own**: any MoE FP4-MMA kernel must be clean-room + contributable to llama.cpp's NVFP4 effort.

## 7. Sources
Upstream: PRs #23572 (open, dense Blackwell FP4-MMA +40% PP), #21896 (closed, earlier Blackwell NVFP4 CUDA), #23682/#22804/#21971 (merged, Gemma4 + NVFP4 convert/load), #22897/#23961/#20456/#23402/#23692; `ggml/include/ggml.h` (GGML_TYPE_NVFP4=40), `ggml/src/ggml-common.h` (block_nvfp4). Lynn: `reports/stage6/stage6_evidence_ledger_20260604.json`, `R5C4_FULL_ACTIVE_MOE_PREFILL_SPEED_AB_CONTRACT_20260604.md`, `R5C_NVF4_UE4M3_CUTLASS_CONTRACT_20260604.md`.
