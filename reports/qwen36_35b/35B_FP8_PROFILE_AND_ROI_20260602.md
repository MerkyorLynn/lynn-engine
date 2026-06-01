# 35B-A3B FP8 decode profile + engine ROI decision (2026-06-02)

## Profile (LYNN_MTP_PROFILE, 64-token decode, graph-safe FP8 MoE eager)
| section | ms/tok | share |
|---|---:|---:|
| `full_attn_t1.rope` | 2.40 | |
| `full_attn_t1.qkv` | 2.23 | |
| `full_attn_t1.o_proj_total` + `o_proj.default` | 2.37 | |
| `full_attn.attention.sdpa` | 0.37 | |
| gate + cache_write | 0.37 | |
| **all attention** | **7.7** | **2.6%** |
| **MoE FFN + lm_head + glue** (uninstrumented) | **~286** | **97.4%** |
| **measured total** | **294.2** (3.4 TPS) | |

`fp8_decode_profile.json` has the raw snapshot.

## Diagnosis
35B FP8 decode (3.4 TPS) is **not** attention-bound (2.6%) and **not**
dispatch-bound (the reusable graph gave only +10%). **97% is the FP8 MoE FFN** —
the per-expert decode loop: for each of K=8 active experts, `index_select` to
gather the expert's weights from the [256, ...] tensor + a Triton fp8 gate/up
kernel + a `torch._scaled_mm` down-proj, repeated across ~30 MoE layers/token.

The **same model's NVFP4 path runs the identical MoE in ~26 ms/tok (38.96 TPS)
— ~11× faster** — because its `native_fast_2d` packed kernels avoid the
per-expert Python iteration + gathers + per-expert `_scaled_mm`.

## ROI decision (Spark + priorities delegated)
- **Low ROI — fix FP8 MoE (P2 grouped GEMM):** even a perfect grouped FP8 GEMM
  must beat NVFP4's already-fast 26 ms; at M=1 decode both are memory-bound and
  FP8 reads ~2× the bytes (8-bit vs 4-bit). Unlikely to clearly win. Deprioritize.
- **High ROI — reusable graph on the NVFP4 path:** the NVFP4 35B is already
  38.96 TPS and the capture-once/replay-many decode graph (M3) is built + proven
  token-exact. Make the NVFP4 MoE decode graph-safe (P1-style fixed-K dispatch
  for the packed path) and capture it → target > 38.96 toward llama.cpp 69.77.
- **FP8's real role:** M>1 (MTP batched verify), where the per-expert cost
  amortizes across K rows and FP8-MMA's 1.64× actually applies. 35B already has
  a trained MTP sidecar (`qwen36-35b-a3b-mtp`) → K≥2 usable.

## Next (high-ROI, next focused session)
1. Graph-safe NVFP4 MoE decode dispatch (P1 for the `native_fast_2d`/packed path).
2. Capture the 35B **NVFP4** decode in the reusable graph; measure vs 38.96.
3. If graph helps NVFP4 → stack the 35B MTP sidecar (K≥2) on top.
