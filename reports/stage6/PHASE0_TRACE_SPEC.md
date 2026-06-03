# Stage 6 Phase-0 Contract — packed-prefill / zero-reload serving

**Owner:** Codex autonomous restart track  
**Date:** 2026-06-03  
**Branch:** `claude/fp8-9b-revival-graph-mtp-20260601`

## Verdict

The old "decode zero-shadow will push 45 -> 70 TPS" hypothesis is closed. The
Spark RC decode path has already been measured around 44-45 TPS, and the
bankable Stage-6 win is memory/capability: release the BF16 dequant shadow after
prefill and decode at ~28 GiB resident.

The next real lever is **packed-NVFP4 prefill**:

- Current multi-request server cycle is `reload -> prefill -> release -> decode`.
- It is correct, but reload costs ~23-24 s because it rebuilds ~60 GiB of BF16
  shadows from packed NVFP4.
- A packed-prefill path makes the cycle `prefill-from-packed -> decode` and keeps
  the runner around 27-28 GiB without per-request reload.

This is a service portability goal first, not a decode TPS claim.

## Evidence Anchors

| claim | source |
|---|---|
| Prefill enters attention/FFN through `_prefill_layer` | `engine/full_forward.py::_prefill_layer` |
| Full-attn prefill now calls `_linear_prefill(... _prefill_weight(...))` for q/k/v/o | `engine/incremental_decode.py::prefill_full_attn` |
| Linear-attn prefill now calls `_linear_prefill` for qkv/z/b/a/out projections | `engine/incremental_decode.py::prefill_linear_attn` |
| MoE packed-prefill proof path reuses exact T=1 packed decode MoE | `engine/full_forward.py::_moe_forward_packed_prefill_slow` |
| Current packed linear kernels remain T=1-only | `engine/nvfp4_runtime.py::PackedNVFP4Linear.forward` |
| Server release/reload cycle is already wired | `server/openai_http.py::LynnEngineHandle.generate` |

Run the static census:

```bash
python3 scripts/stage6_phase0_static_census.py --markdown
```

## New Prototype Flag

`LYNN_PACKED_PREFILL_SLOW=1`

Default is off. When enabled:

- Prefill linear projections use BF16 weights if present.
- If a BF16 `.weight` has been released but `.weight.packed` exists,
  `_linear_prefill` loops over rows with the existing packed-NVFP4 T=1 kernels.
- MoE prefill defaults to `LYNN_PACKED_PREFILL_SLOW_MODE=stream_bf16`: it reads
  packed NVFP4, dequants only the current layer into temporary BF16 tensors, runs
  the same BF16 MoE math, and releases those temporaries before the next layer.
- `LYNN_PACKED_PREFILL_SLOW_MODE=decode_kernel` keeps the older T=1 decode-kernel
  replay as a diagnostic only; Spark P0.1 attempt #1 proved it is memory-clean
  but not token-exact across decode state.

This is intentionally slow. Its only purpose is to prove no-reload correctness
and memory residency before writing batched/grouped prefill kernels.

## Spark Gate P0.1

Run after the 60 GiB release path is already verified:

1. Load RC stack with packed MoE aliases attached.
2. Run one baseline request with BF16 shadows present.
3. Call `release_decode_bf16_shadows(include_projection_aliases=False)` to drop
   the banked MoE BF16 shadow only.
4. Without calling `reload_decode_bf16_shadows()`, run a second prefill with
   `LYNN_PACKED_PREFILL_SLOW=1`.
5. Assert:
   - no `reload_decode_bf16_shadows()` call;
   - no `KeyError('mlp.experts.*')` from released BF16 expert shadows;
   - token-exact output against the BF16 prefill baseline for a short prompt;
   - resident memory stays near 28 GiB before and after prefill;
   - record packed-prefill latency honestly.

P0.1 deliberately does **not** delete every projection/shared-expert BF16 weight.
Those aliases interact with decode env flags and shared-expert routing, so they are
the next gate rather than part of the first no-reload proof.

**Result (2026-06-03): PASSED with `LYNN_PACKED_PREFILL_SLOW_MODE=stream_bf16`.**
See `reports/stage6/P01_NO_RELOAD_SMOKE_20260603.md`.

The older `decode_kernel` replay mode failed token-exactness despite clean memory
and no reload, so it remains diagnostic only.

P0.1 promoted the resident inventory gate; P0.2 is now recorded below.

## Spark Gate P0.2

**Result (2026-06-03): PASSED as a resident-byte inventory gate.**
See `reports/stage6/P02_RESIDENT_INVENTORY_20260603.md`.

After the 60 GiB grouped-MoE shadow release, only **4.72 GiB** BF16 resident
weights remain:

- `linear_attn.projection`: 1.884 GiB
- `outside.embed`: 0.947 GiB
- `outside.lm_head`: 0.947 GiB
- `full_attn.projection`: 0.508 GiB
- `moe.shared_expert`: 0.391 GiB
- `moe.router`: 0.039 GiB

The script found **0.0 GiB** packed-alias candidates in the normal inventory
mode, so further resident reductions require explicit packed-prefill / packed
lookup paths. Router and norms are too small to lead the next phase.

## Next Engineering Phases

| phase | target | gate |
|---|---|---|
| P1 single projection | **PASSED 2026-06-04**: real `linear_attn.in_proj_qkv` M=1 packed Triton matvec | numeric/no-shadow/microbench all pass; see `reports/stage6/P1_DENSE_PROJECTION_POC_20260604.md` |
| P1-A batched projections | Replace row-loop packed linear prefill with batched/M>1 packed-NVFP4 projection kernels for full-attn and linear-attn qkv/z/b/a/o | token-exact, lower prefill latency than reload+BF16, no projection BF16 shadow |
| P2 | Replace row-loop packed MoE prefill with grouped M>1 packed expert kernels | token-exact vs BF16 MoE prefill, no MoE BF16 shadow, latency measured by prompt length |
| P3 | Server integration: if `LYNN_PACKED_PREFILL=1`, skip per-request reload and keep 27-28 GiB steady-state | multi-request A/B: no reload, memory flat, decode TPS unchanged |
| P4 | RC quality and long-context headroom | `spark_rc_quality_regression.py`, long prefill smoke, `/health` metrics |

Do not promote any phase on speed-only evidence. Each gate requires correctness,
memory, latency, and failure logs.
