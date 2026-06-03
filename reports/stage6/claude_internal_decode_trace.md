I have all the evidence needed. Here is my complete independent analysis.

---

# Stage 6 Phase-0 — claude-internal independent trace

**Scope:** RC decode config (`LYNN_MOE_IMPL=packed_nvfp4`, `LYNN_MOE_FAST_FIXED=1`, `LYNN_NATIVE_GATEUP_BACKEND=triton_fast_decode`, `LYNN_PACKED_DECODE=0`, `LYNN_PACKED_SHARED_EXPERT=0`, `LYNN_LINEAR_ATTN_INPROJ_FUSED_NATIVE_FP4=1`, `LYNN_NATIVE_FP4_LM_HEAD=1`) — i.e. the exact env in `scripts/spark_35b_nvfp4_decode_profile.py:18-33` and `scripts/spark_stage6_shadow_byte_audit.py:26-47`.

**Model:** Qwen3.6-35B-A3B, hidden=2048, 40 layers (30 linear-attn / 10 full-attn, `engine/inference_state.py:39`), 256 experts top-8, `moe_intermediate_size=512`, `vocab=248320`, `head_dim=256`, `num_kv_heads=2` (`reports/a100/a100_qwen36_35b_a3b_mtp_index_audit.json:5-8`, `engine/inference_state.py:27-36`). `num_attention_heads` and `shared_expert_intermediate_size` (`SI`) not checked into the repo — flagged below; conclusions don't depend on them.

---

## TL;DR — the headline correction

**The Stage-6 step1 "evidence-lock" conclusion is wrong, and it changes the whole campaign.** The commit `c13a924` reads the post-release `KeyError('mlp.experts.1.gate_proj.weight')` as *"decode DEPENDS on BF16 MoE expert weights"* and locks the target to *"write a read-4bit + register-dequant + bf16-GEMV MoE active-expert decode kernel (LYNN_MOE_EXPERT_FP4_GEMV=1)"*.

But:
1. **That KeyError is from PREFILL, not decode.** `generate()` re-runs a full prefill before decoding (`engine/resident_runner.py:1652-1656`), and the per-expert key format `mlp.experts.{e}.gate_proj.weight` is read **only** in the prefill MoE (`engine/full_forward.py:337-339`). The decode MoE reads `mlp.experts._gate_up_packed` and would raise a *different* message (`"packed NVFP4 MoE aliases missing"`, `engine/moe_packed_nvfp4.py:982-984`) — never that key.
2. **The decode MoE expert path is ALREADY a read-4bit / register-dequant / bf16-accumulate GEMV.** The kernel the campaign proposes to write already exists and runs in the RC (`triton_kernels/nvfp4_moe.py:251-349, 352-427`). Building `LYNN_MOE_EXPERT_FP4_GEMV=1` re-implements it → **~0 TPS gain**.
3. The "4× weight cut 64.7→15 GiB → wall 40→140 TPS" math is invalid: the 60 GiB BF16 it counts is the prefill MoE shadow that **decode never reads**. Dropping it is a *memory* win, not a per-token *bandwidth* win.

Details and the real lever below.

---

## (1) DECODE per-token byte-path trace — FP4 vs BF16

Decode driver: `resident_runner._decode_layer_fast` (`:1256`) → `full_forward._decode_layer` (moe_fn resolved to `moe_forward_decode_packed_nvfp4`, `:669-670`).

### Per-tensor tag (file:line + dtype actually read)

| Block (×layers) | Tensor | Path / dispatch | Bytes |
|---|---|---|---|
| **Full-attn q/k/v** (×10) | `self_attn.{q,k,v}_proj.weight` | `incremental_decode.py:420-422` → `_decode_weight` returns `w[key]` because `LYNN_PACKED_DECODE=0` and no `LYNN_PACKED_DECODE_FULL_ATTN` (`:145-159`) | **BF16** |
| **Full-attn o** (×10) | `self_attn.o_proj.weight` | `_full_attn_o_proj` → `_linear` → `F.linear` (`:93-112`, `:90`) | **BF16** |
| **Linear-attn in-proj** (×30) | `linear_attn._in_proj_qkv_z_b_a.weight` | `incremental_decode.py:1173-1174`; alias is a `PackedNVFP4FusedLinear` built by `_prepare_linear_attn_inproj_fused_native_fp4` (`resident_runner.py:337-338, 824`); `_linear`→`forward_native_fast_2d`→`_scaled_mm` (`nvfp4_runtime.py:470-489`) | **FP4** |
| **Linear-attn out-proj** (×30) | `linear_attn.out_proj.weight` | `incremental_decode.py:1285` (`# 8. out_proj`) → `_linear`→`F.linear` | **BF16** |
| **Linear-attn conv/A_log/dt** (×30) | `conv1d.weight`, `A_log`, `dt_bias` | `:1192,1197-1200` | BF16 (tiny) |
| **MoE routed experts** (×40, top-8/256) | `mlp.experts._gate_up_packed`, `_down_packed` (+ scales) | `moe_packed_nvfp4.py:793-804` (gate_up) & `:860-870` (down) → kernels `nvfp4_moe.py:251,352` | **FP4** |
| **MoE shared expert** (×40, dense) | `mlp.shared_expert._gate_up_proj.weight`, `down_proj.weight` | `moe_packed_nvfp4.py:889-900`; fused path **requires** `dtype==bfloat16` (`_try_fused_shared_expert_output` `:285-286`); else `F.linear` (`:894-899`) | **BF16** |
| **MoE router** (×40) | `mlp.gate.weight` | `_router_linear` `:163-186` → `F.linear`/`torch.mm` | **BF16** |
| **LM head** (×1) | native-FP4 lm_head | `_prepare_native_fp4_lm_head` (`resident_runner.py:349-350`) | **FP4** |
| Norms, embed-row | various | per-layer | BF16 (negligible) |

### Bytes/token estimate (FP4 packed = 0.5 B/param + per-16 fp8 scale ≈ 0.0625 B/param; BF16 = 2 B/param)
Using H_Q=16 (Qwen3-Next attn config; **confirm on Spark**) and SI=512 (lower-bound; **confirm**):

| Group | dtype | MB/token |
|---|---|---|
| LM head (508.5M params) | FP4 | ~286 |
| Routed experts (8×3.15M ×40) | FP4 | ~566 |
| Linear-attn in-proj (25.3M ×30) | FP4 | ~426 |
| **FP4 subtotal** | | **~1.28 GB** |
| Full-attn q/k/v/o (27.3M ×10) | **BF16** | ~545 |
| Linear-attn out-proj (8.39M ×30) | **BF16** | ~505 |
| Shared expert (3.15M ×40, SI=512) | **BF16** | ~252 |
| Router (0.52M ×40) | **BF16** | ~42 |
| **BF16 subtotal** | | **~1.35 GB** |
| **Total weights/token** | | **~2.6 GB** |

### Answer to the gating question
**YES — significant BF16 weight IS on the steady-state decode read path: ~1.35 GB/token** (full-attn q/k/v/o + linear-attn out_proj + shared expert + router).

**But — critically — this BF16 is NOT the 60 GiB shadow.** These tensors have **no `.packed` alias** (the alias-attach `_prepare_packed_decode_aliases` is gated off in the RC, `resident_runner.py:329-334`), so `release_decode_bf16_shadows()` does **not** touch them (`:1384-1390` only drops `*.weight` that has `*.weight.packed`). They are genuinely-resident, genuinely-read weights. The 60 GiB shadow is a *separate* set (the routed-expert prefill BF16, §3) that decode never reads.

**Reconciliation with the LEAD's "implied 5.37 GB/token":** `1/44.71 TPS × 240 GB/s = 5.37 GB`. Actual weight reads ≈ **2.6 GB** (half FP4). The 2× gap (5.37 vs 2.6) is **not** unaccounted weight bandwidth — it's KV/recurrent-state traffic (~0.1–0.2 GB/tok) plus, dominantly, **launch/latency overhead converted into a bandwidth-equivalent number**. The implied-bytes figure overstates true weight bandwidth → decode is **not cleanly bandwidth-bound**. This directly contradicts reading 5.37 GB/tok as "confirms the ~6 GB BF16-shadow premise."

---

## (2) FP4 kernel efficiency — do they materialize a BF16 temp in HBM?

**MoE gate_up** — `_grouped_gate_up_silu_fast_decode_kernel` (`triton_kernels/nvfp4_moe.py:251-349`):
- Loads packed 4-bit directly (`tl.load(gate_up_packed_ptr…)` `:316-325`), decodes nibble→e2m1 **in registers** (`_e2m1_from_nibble_fast` `:328-329`), multiplies by scale·x, **fp32 register accumulate** (`:341-345`), stores only the `[slot, intermediate]` activation as bf16 (`:349`).
- **No full BF16 weight is ever materialized in HBM.** ✅ True read-4bit/register-dequant/bf16-GEMV.

**MoE down** — `_grouped_down_weighted_sum_kernel` (`:352-427`): identical pattern — packed load (`:409-413`), register dequant (`_e2m1_from_nibble` `:415`), fp32 accumulate weighted by route, bf16 store (`:427`). **No HBM temp.** ✅

**Attn/dense/lm_head** — `forward_native_fast_2d` (`nvfp4_runtime.py:348/470`) → `_native_decode_scaled_mm` (`:64-92`) → `torch._scaled_mm` on `self._native_weight_t()`, which is just `weight_packed.view(float4_e2m1fn_x2).t()` (`:447-450`) — a **view**, no Python-side BF16 materialization. With `LYNN_NVFP4_BF16_OUT=1` the output is bf16 directly (`:73-82`), eliminating the fp32→bf16 copy.

> **One GPU-side unknown for LEAD:** whether CUTLASS's NVFP4 `_scaled_mm` on **sm_121** (spec: no FP4 MMA) internally dequantizes the weight to a *wide HBM temp* + separate GEMM, or dequants in mainloop/smem. From Python this is undecidable. **Concrete check:** `ncu`/`nsys` one full-attn `forward_native_fast_2d` and look for (a) an extra kernel writing an `[out,in]` bf16 tensor and (b) HBM **write** traffic ≈ weight size before the matmul. Absent → CUTLASS already register-dequants (no win there). Present → a custom register-dequant bf16-GEMV for these projections is justified. Given the model decodes at 44 TPS with the 508M-param lm_head on this path, a full per-projection BF16 temp is unlikely (it would be ruinous), but verify.

**Verdict for (2):** the FP4 kernels are clean — option (4b) "kernels waste HBM on a BF16 temp" is **not** supported for the MoE path, and only an unverified maybe for the CUTLASS internals.

---

## (3) PREFILL BF16 + is the 60 GiB shadow droppable for decode-only?

**Where prefill reads per-expert BF16:** `engine/full_forward.py::_moe_forward`. Three branches:
- FP8 (`:251-294`) — reads `mlp.experts.gate_up_proj.weight_fp8`.
- **Fused stacked BF16** (`:323-335`): `F.linear(x_e, w["mlp.experts.gate_up_proj"][e])` and `…["mlp.experts.down_proj"][e]` — this is the path the RC stack uses (stacked BF16 resident = the 60 GiB).
- **Per-expert BF16 fallback** (`:337-339`): `w[f"mlp.experts.{e}.gate_proj.weight"]` — only reached when the stacked key is absent.

**There is NO packed-NVFP4 prefill MoE.** Prefill must read FP8 or BF16 expert weights.

**The KeyError mechanism (disambiguated):** `release_decode_bf16_shadows()` drops `mlp.experts.gate_up_proj` + `mlp.experts.down_proj` (`resident_runner.py:1379-1382`). The byte-audit then calls `tps()`→`generate()`, whose **prefill** (`:1652-1656`) calls `_moe_forward`; with the stacked key now gone, `fused_experts=False` (`full_forward.py:323-324`) → falls to `:337` → `mlp.experts.1.gate_proj.weight` doesn't exist → **`KeyError('mlp.experts.1.gate_proj.weight')`**. **This is prefill, exactly the observed error.** Decode (`moe_forward_decode_packed_nvfp4`) never references that key.

**Is the 60 GiB shadow droppable for decode-only?** **Yes.** Resident 87.22 GiB = BF16 64.72 + uint8 15.00 + fp32 7.50. The 60 GiB freed = stacked BF16 routed-expert weights, a pure dequant duplicate of the 15 GiB `_gate_up_packed`/`_down_packed`. Decode reads only the packed copy → **decode survives shadow-free**. The correct in-process proof already exists: `generate(release_decode_shadows_after_prefill=True)` (`resident_runner.py:1627, 1764`) — release *after* prefill, then decode in the *same* call (no re-prefill). The byte-audit's between-`generate()` release is what triggers the false KeyError.

**Caveat for serving:** a prefill+decode server **cannot** drop the shadow per request unless it (a) adds a packed prefill MoE, or (b) reloads shadows before each prefill. The docstring says exactly this (`:1355-1357`). So the 60 GiB is a **decode-only / single-prompt memory win** (frees 60 GiB on the shared 128 GB Spark → bigger KV/batch headroom), **not** a TPS lever.

---

## (4) The real Stage-6 lever — ranked, with evidence

**#0 (do first, ~free): Kill the mis-locked target & prove the shadow thesis is dead.** Run `generate(release_decode_shadows_after_prefill=True)` (or the byte-audit with that flag) → decode runs shadow-free at **~same TPS, cos≈1**. That single result invalidates "delete shadow → 40→140 TPS" and the `LYNN_MOE_EXPERT_FP4_GEMV` kernel (which duplicates the existing FP4 MoE GEMV, `nvfp4_moe.py:251/352`). **Do not write that kernel.**

**#1 (the real lever): attack dispatch / launch overhead — reusable decode CUDA graph + launch-folding.**
Evidence: implied 5.37 GB/tok vs actual ~2.6 GB/tok of weights = ~2× bandwidth left unused → **latency/launch-bound, not bandwidth-bound**; matches the banked campaign finding *"real blocker = eager runtime overhead, not offset"* (commit `14e1476`). 40 layers × {qkv, conv, recurrent, gate, gate_up, down, shared, out_proj, norms} = hundreds of tiny M=1 launches/token. The infra already exists (linear-block graph capture, `_get_reusable_linear_block_graphs` `:1496`; fixed-shape full-attn `LYNN_FULL_ATTN_FIXED_SHAPE`). **Highest ROI for 45→70.**

**#2 (cheap A/B, zero new code): flip the EXISTING packed-decode flags for the BF16 decode weights.**
~1.05 GB/tok of BF16 (full-attn q/k/v/o + linear-attn out_proj) can go FP4 today via `LYNN_PACKED_DECODE_FULL_ATTN=1` + `LYNN_PACKED_DECODE_LINEAR_ATTN=1` → `_prepare_packed_decode_aliases` attaches `.packed` and `_decode_weight` routes to `forward_native_fast_2d` (`incremental_decode.py:145-159`). **Telling fact: these flags exist and are OFF in the RC** — strongly implying a prior A/B found no TPS gain (consistent with launch-bound). Re-run the A/B to *confirm*: if TPS doesn't move, that's positive proof for #1 and closes (4c). **Gate:** `forward_native_fast_2d` (fp4) is **not** bit-exact vs BF16 `F.linear` → needs a cos≈1 quality check, not token-exact. (Shared expert + router would also need a packed path; lower priority, smaller bytes.)

**#3: only if #1 saturates and ncu shows a CUTLASS wide temp (per §2)** — write a custom register-dequant bf16-GEMV for the dense projections behind a new flag, default OFF, byte-checked. Not justified until the profiler shows the temp.

**Net:** the Stage-6 win is **dispatch/graph (launch overhead)**, not "delete the BF16 shadow." The shadow drop is a real but separate **memory** win (60 GiB, decode-only). The MoE expert decode is already FP4 — re-confirm #0, then put the effort into #1.

---

*Constraints honored: no files modified, no git commit, analysis to stdout only. All claims carry file:line and are checkable on Spark by LEAD; the GPU-side items (CUTLASS temp in §2, the #2 A/B, the #0 shadow-free proof) are explicitly marked for verification.*
