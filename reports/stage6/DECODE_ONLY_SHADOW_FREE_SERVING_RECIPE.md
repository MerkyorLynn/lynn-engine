# Decode-only shadow-free serving — verified recipe (2026-06-03)

**The bankable Stage-6 win: free 60 GiB on Spark by dropping the BF16 dequant-shadow during decode.**
Verified end-to-end on the 35B-A3B NVFP4 RC stack (`scripts/spark_stage6_decode_only_serving_verify.py`).

## Why it's safe
The ~60 GiB BF16 weights are a dequant duplicate of the 15 GiB resident packed NVFP4, read **only by
PREFILL** (`engine/full_forward.py::_moe_forward` stacked-BF16 experts + prefill attn `F.linear`).
**DECODE reads packed NVFP4 only** — so the shadow can be dropped for the whole decode phase with zero
output change.

## Primitives (already implemented in `engine/resident_runner.py`)
- `release_decode_bf16_shadows()` — drop the shadow (resident 88 → 28 GiB). Decode runs unchanged.
- `reload_decode_bf16_shadows()` — rebuild it from the resident packed NVFP4 via GPU re-dequant
  (`_dequant_nvfp4_slot`), **no disk I/O**. Needed before the next PREFILL.
- `generate(..., release_decode_shadows_after_prefill=True)` — does `prefill → release → decode` in one call.

## Measured (Spark GB10, RC stack, 3 requests)
| metric | value |
|---|---|
| resident with shadows (baseline) | **88.2 GiB** |
| resident during decode (released) | **28.2 GiB**  → **~60 GiB freed** |
| output | **token-exact ×3** vs baseline |
| decode TPS | 44.16 vs 44.27 = **0.998×** (no regression) |
| reload cost (per prefill) | **~23 s** (GPU re-dequant of 60 GiB; no disk) |

## Two serving modes
1. **Long single session (SHIP NOW).** One prefill, then a long decode/chat turn: call
   `generate(release_decode_shadows_after_prefill=True)`. Reload cost = **0** (no further prefill).
   The runner sits at **28 GiB** for the whole turn → 60 GiB free for co-resident services / concurrent
   decode streams / a larger KV (if allocated after release). This is the drop-in for the desktop
   single-session chat (the common case) and Spark's long-ctx value.
2. **Multi-request server (needs the follow-up).** Interleaved prefills would pay **~23 s reload each** —
   too heavy. The right fix is a **packed-NVFP4 prefill path** (option (a) in the spawned task) so prefill
   never needs the shadow → steady **27 GiB always**, zero reload. Until then, multi-request servers should
   NOT use the release/reload cycle per request.

## Caveat (follow-up)
KV is pre-allocated at `LynnInferenceState` creation (`max_seq_len`, **before** prefill/release), so fitting
a *longer single context* needs KV-allocation-after-release reordering. The immediate, verified win is the
**steady-state decode footprint (88→28 GiB)** → multi-service co-residency + concurrent-decode headroom, not
a longer single-request context yet.

## Wiring pointer (Brain / desktop serving)
For single-session chat: pass `release_decode_shadows_after_prefill=True` to the resident runner's
`generate`. For a server that prefills per request: implement the packed prefill (spawned task
"Productize decode-only shadow-free") rather than per-request reload.
