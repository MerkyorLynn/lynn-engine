# Stage 6 R5-C2C Real D-Row Slot Scatter Smoke Summary

| Field | Value |
|---|---|
| Result | `/root/autodl-tmp/src/lynn-engine-r5c2c-codex/reports/stage6/r5c3a_gateup_prefill_timing_smoke_20260604_203052/result.json` |
| Decision | `PASS_R5C2C_REAL_D_ROW_SLOT_SCATTER_SMOKE` |
| CUTLASS dir | `/root/autodl-tmp/src/cutlass` |
| CUTLASS example | `/root/autodl-tmp/src/cutlass/examples/79_blackwell_geforce_gemm/79d_blackwell_geforce_nvfp4_grouped_gemm.cu` |
| Benchmark file | `/root/autodl-tmp/src/lynn-engine-r5c2c-codex/reports/stage6/r5c3a_gateup_prefill_timing_smoke_20260604_203052/result.benchmark.txt` |
| D-row digest file | `/root/autodl-tmp/src/lynn-engine-r5c2c-codex/reports/stage6/r5c3a_gateup_prefill_timing_smoke_20260604_203052/result.d_row_digest.jsonl` |
| Selected tokens/top_k/experts | `256 / 8 / 8` |
| Tokens per expert | `[256, 256, 256, 256, 256, 256, 256, 256]` |
| Gate/up output width N | `1024` |
| Temporary D-row digest patch applied/restored | `True` / `True` |
| Real D-row slot scatter banked | `True` |
| Selected-output epilogue kernel banked | `False` |
| SwiGLU/down projection banked | `False` |
| Grouped-MoE FP4-MMA POC banked | `False` |
| Kernel speed banked | `False` |
| Default promotion banked | `False` |
| Avg runtime ms (trace only) | `[0.0206432, 0.02376]` |

## Run Gates

| Gate | Value |
|---|---:|
| build_invoked | `True` |
| build_succeeded | `True` |
| d_row_digest_patch_applied | `True` |
| d_row_digest_patch_restored | `True` |
| run_succeeded | `True` |
| cooperative_passed | `True` |
| pingpong_passed | `True` |
| host_reference_seen | `True` |
| dispositions_passed_count_ge_2 | `True` |
| groups_seen_match_experts | `True` |
| tokens_per_expert_match | `True` |
| grouped_order_complete | `True` |
| digest_file_exists | `True` |
| schedules_captured | `True` |
| schedule_scatters_passed | `True` |

## Schedule Scatter Gates

| Schedule | Records | Row counts | D/ref row digest match | Scatter match | Fault injections |
|---|---:|---|---:|---:|---:|
| `cooperative` | `2048` | `[256, 256, 256, 256, 256, 256, 256, 256]` | `True` | `True` | `True` |
| `pingpong` | `2048` | `[256, 256, 256, 256, 256, 256, 256, 256]` | `True` | `True` | `True` |

## Boundary

- This R5-C2C artifact banks only `banked_real_d_row_slot_scatter=true`.
- It emits and scatters real CUTLASS D/ref row digests into `[T, top_k, N_gateup]` selected slots.
- It does not bank an in-epilogue selected-output CUDA kernel.
- It does not perform or bank SwiGLU activation, down projection, router validation, full grouped-MoE speed, server behavior, RC quality, or runtime default promotion.
- Runtime/TFLOPS are trace-only; speed is first eligible at a later grouped active-MoE POC gate.

