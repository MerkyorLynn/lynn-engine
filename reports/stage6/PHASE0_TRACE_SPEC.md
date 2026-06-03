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
| P1-A naive batched bridge | **REJECTED 2026-06-04**: numeric/no-shadow pass, M>1 perf fail | see `reports/stage6/P1A_BATCHED_PROJECTION_POC_20260604.md` |
| P1-A tiled scalar bridge | **REJECTED 2026-06-04**: numeric/no-shadow pass and up to 25.93x faster than naive, but still slower than BF16 GEMM for M>1 | see `reports/stage6/P1A_TILED_PROJECTION_SWEEP_20260604.md`; dense M>1 needs native FP4-MMA/CUTLASS-style bridge if pursued |
| P2 census | **PASSED 2026-06-04**: single-layer routed MoE census proves `stream_bf16` is exact but 0.49-0.51s/layer; `smallm` verifier is memory-clean but slow | see `reports/stage6/P2_GROUPED_MOE_PREFILL_CENSUS_20260604.md` |
| P2-A single-expert gate/up | **COMPONENT ONLY 2026-06-04**: packed no-shadow gate/up+silu passes numeric in main run, but scalar-dequant loses to BF16 (best sweep M=64 0.115x) | see `reports/stage6/P2A_GATEUP_PREFILL_POC_20260604.md`; do not wire into serving |
| P2-B routed gate/up grouping | **LOWER-BOUND PASS 2026-06-04**: route-grouped packed gate/up is numeric/no-shadow pass and M64 20.0ms/layer, but 0.423x vs BF16 gate/up | see `reports/stage6/P2B_ROUTED_GATEUP_GROUPING_POC_20260604.md`; continue to routed down |
| P2-C active routed MoE | **LOWER-BOUND PASS 2026-06-04**: packed gate/up + packed down active path is numeric/no-shadow pass; M64 23.83ms/layer, ~21x faster than `stream_bf16`, 0.560x vs BF16 active | see `reports/stage6/P2C_ACTIVE_MOE_LOWER_BOUND_POC_20260604.md`; continue to shared/router-inclusive one-layer |
| P2-D one-layer hybrid | **MIXED PASS 2026-06-04**: router/shared-inclusive hybrid is numeric/no-shadow pass; M64 29.21ms/layer, ~17x faster than `stream_bf16`, but 0.741x vs BF16 full MoE | see `reports/stage6/P2D_ONE_LAYER_MOE_HYBRID_POC_20260604.md`; do not wire into serving |
| P2-E scheduler / active retune | **PASSED 2026-06-04**: sort scheduler + `block_inter=8` is numeric/no-shadow pass; M64 hybrid 20.25ms/layer vs BF16 21.41ms (1.057x) | see `reports/stage6/P2E_SCHEDULER_ACTIVE_RETUNE_20260604.md`; continue to one-layer opt-in replacement |
| P2-F one-layer opt-in replacement | **PASSED 2026-06-04**: P2-E moved into engine dispatch as `LYNN_PACKED_PREFILL_SLOW_MODE=p2e_hybrid`; M64 20.23ms/layer vs BF16 21.10ms (1.043x), 24.96x vs `stream_bf16`, peak 0.643GiB | see `reports/stage6/P2F_ONE_LAYER_REPLACEMENT_VERIFY_20260604.md`; continue to multi-layer smoke |
| P2-G multi-layer no-reload smoke | **PASSED 2026-06-04**: 4-layer residual-scale MoE smoke numeric/no-shadow/speed pass; M64 80.97ms vs BF16 85.12ms (1.051x), 30.04x vs `stream_bf16`, peak 2.527GiB | see `reports/stage6/P2G_MULTILAYER_MOE_SMOKE_20260604.md`; continue to full prefill selected-layer smoke |
| P2-H full prefill selected-layer smoke | **PASSED 2026-06-04**: selected-layer `_prefill_layer` smoke passes with RMSNorm + linear/full attention cache + MoE; mixed L0-3 T16 45.97ms vs BF16 58.57ms (1.274x), 52.09x vs `stream_bf16`, peak 2.606GiB vs stream 14.585GiB | see `reports/stage6/P2H_SELECTED_LAYER_PREFILL_SMOKE_20260604.md`; continue to P2-I all-selected MoE expansion and P2-J linear-attn prefill trace |
| P2-I selected-MoE expansion | **PASSED 2026-06-04**: mixed L0-7 T16 selected prefill numeric/no-shadow/speed pass; 88.96ms vs BF16 113.82ms (1.279x), 46.70x vs `stream_bf16`, peak 5.123GiB vs stream 17.102GiB | see `reports/stage6/P2I_SELECTED_MOE_EXPANSION_SMOKE_20260604.md`; continue to P2-J linear-attn prefill trace before server promotion |
| P2-J linear-attn prefill trace | **PASSED 2026-06-04**: segment trace is exact vs `prefill_linear_attn`; `chunk_gated_delta_with_state` dominates T16..512 at 71-76% of traced wall time | see `reports/stage6/P2J_LINEAR_ATTN_PREFILL_TRACE_20260604.md`; next native target is P2-K gated-delta prefill kernel |
| P3 | Server integration: if `LYNN_PACKED_PREFILL=1`, skip per-request reload and keep 27-28 GiB steady-state | multi-request A/B: no reload, memory flat, decode TPS unchanged |
| P4 | RC quality and long-context headroom | `spark_rc_quality_regression.py`, long prefill smoke, `/health` metrics |

Do not promote any phase on speed-only evidence. Each gate requires correctness,
memory, latency, and failure logs.
