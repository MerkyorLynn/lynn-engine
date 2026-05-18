# Qwen3.6 35B W4A16 Native MoE P153 Stage Split

Date: 2026-05-19

## Purpose

P152 proved that the packed NVFP4 native slot-MoE probe is slightly faster than
the Triton active-MoE stage, but not exact against the P147 Triton-stage
contract.  P153 splits the native path into gate/up and down stages to identify
which side owns the drift.

## Result

Verdict: **GATEUP_DRIFT**

| Check | Exact | Max Abs | Mean Latency |
|---|---:|---:|---:|
| native inter vs Triton inter | 6/18 | 2.44140625e-4 | 0.04132 ms |
| native down(Triton inter) vs Triton out | 15/18 | 2.38418579e-7 | 0.04946 ms |
| native full split vs Triton out | 12/18 | 1.22070312e-4 | 0.09052 ms |

The down kernel is effectively aligned with the Triton contract when it consumes
the Triton BF16 intermediate.  The non-exact full-path rows track the native
intermediate drift, so the next kernel work should target gate/up reduction,
SiLU, and BF16 intermediate store semantics.

## Worst Rows

| Stage | Worst Fixture | Max Abs | Cosine |
|---|---|---:|---:|
| inter | L28/P00 | 2.44140625e-4 | 1.000000000 |
| inter | L08/P01 | 2.44140625e-4 | 1.000000000 |
| down from Triton inter | L28/P01 | 2.38418579e-7 | 1.000000000 |
| full | L08/P01 | 1.22070312e-4 | 0.999999940 |

## Artifacts

- `reports/qwen36_35b/p153_native_packed_stage_split_20260519_035358.json`
- R6000 source reference: `/root/autodl-tmp/reports/qwen36_35b/p147_triton_stage_reference_20260519_0318`
- R6000 packed fixtures: `/root/autodl-tmp/reports/qwen36_35b/p138_packed_slot_fixtures_kimi_20260518`

## Next Step

Do not escalate native packed MoE to resident P37 yet.  Implement a strict
gate/up candidate that mirrors Triton block accumulation and BF16 store order,
then rerun P153 before P37/P25/structured gates.
