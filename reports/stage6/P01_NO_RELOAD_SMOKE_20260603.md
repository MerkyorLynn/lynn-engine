# Stage 6 P0.1 — packed-prefill no-reload smoke

**Date:** 2026-06-03  
**Host:** Spark GB10 (`dgx-via-n5`)  
**Model:** `/home/merkyor/models/Qwen3.6-35B-A3B-lynn-native-w4a16-nvfp4-from-bf16-20260526`  
**Prompt:** `2+2=`  
**Max new:** `4`  
**Runner:** `scripts/spark_stage6_packed_prefill_no_reload_smoke.py` in docker `lynn-eval-base:cu13`, `PYTHONNOUSERSITE=1`.

## Verdict

**P0.1 PASSED with `LYNN_PACKED_PREFILL_SLOW_MODE=stream_bf16`.**

The runner can release the 60 GiB grouped-MoE BF16 shadow, skip
`reload_decode_bf16_shadows()`, prefill from resident packed NVFP4, and remain
token-exact against the BF16 prefill baseline.

This is a correctness/memory proof, not the final performance path. Streaming
current-layer BF16 temporaries costs ~20.75 s for this tiny prompt. P2 must
replace it with grouped M>1 packed kernels.

## Attempt 1 — `decode_kernel` replay failed token-exactness

Remote log: `/home/merkyor/lynn-engine/reports/stage6/p01_no_reload_20260603_233003/run.log`

| check | result |
|---|---:|
| baseline resident | 88.18 GiB |
| release | 60.00 GiB / 80 tensors |
| resident after release | 28.18 GiB |
| reload calls | 0 |
| probe peak | 28.30 GiB |
| probe prefill | 0.115 s |
| baseline ids | `[20, 198, 1409, 27102]` |
| probe ids | `[20, 271, 248068, 198]` |
| ALL_PASS | false |

Interpretation: replaying the T=1 packed decode MoE is memory-clean and fast, but
it is **not state-coherent** with BF16 prefill across decode steps. It produced
the same first generated token (`5`) and then diverged. Keep this mode as a
diagnostic only: `LYNN_PACKED_PREFILL_SLOW_MODE=decode_kernel`.

## Attempt 2 — `stream_bf16` passed

Remote log: `/home/merkyor/lynn-engine/reports/stage6/p01_no_reload_20260603_233755/run.log`

| check | result |
|---|---:|
| baseline resident | 88.18 GiB |
| baseline prefill | 0.306 s |
| baseline decode | 42.97 tok/s |
| release | 60.00 GiB / 80 tensors |
| resident after release | 28.18 GiB |
| reload calls | 0 |
| probe peak | 40.28 GiB |
| probe resident after decode | 28.18 GiB |
| probe prefill | 20.751 s |
| probe decode | 42.47 tok/s |
| baseline ids | `[20, 198, 1409, 27102]` |
| probe ids | `[20, 198, 1409, 27102]` |
| token-exact | true |
| ALL_PASS | true |

Interpretation: streaming per-layer BF16 temporaries from packed NVFP4 proves the
service can keep the 60 GiB shadow non-resident and avoid the 23-24 s full
reload while staying token-exact. Peak memory is ~40 GiB, not 88 GiB, so this
does not secretly rebuild the full shadow.

## Operational Notes

- APEX was idle before both runs (`requests_processing=0`, `requests_deferred=0`).
- The wrapper stopped the live APEX llama-server, ran the 35B Python runner, and
  restored APEX on `:18098`.
- Post-run APEX health was `{"status":"ok"}` and llama-server returned to ~30 GiB.

## Next Gates

| phase | target | gate |
|---|---|---|
| P0.2 | Inventory projection/shared BF16 residents and decide which can be removed without changing decode semantics | token-exact, no hidden reload, explicit resident byte table |
| P1 | Batched packed-NVFP4 projection prefill kernels | token-exact vs BF16, lower latency than streaming BF16, bounded peak |
| P2 | Grouped M>1 packed MoE prefill kernel | token-exact vs `stream_bf16`, no 60 GiB shadow, latency measured by prompt length |
| P3 | Server mode `LYNN_PACKED_PREFILL=1` with zero reload | multi-request A/B, memory flat, decode TPS unchanged |
