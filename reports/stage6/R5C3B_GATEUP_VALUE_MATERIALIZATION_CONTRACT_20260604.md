# Stage 6 R5-C3B — Gate/Up Value Materialization Contract

Date: 2026-06-04

Verdict: **contract/tooling gate only; no full grouped-MoE FP4-MMA speed,
decode TPS, server behavior, RC quality, or default promotion is banked.**

## Why This Gate Exists

R5-C2C proved that real CUTLASS gate/up D-row **digests** can be scattered back
through Lynn's selected-output slot bridge. R5-C3A then recorded a positive
prefill-shaped gate/up timing trace: the native FP4-MMA gate/up kernel is about
2.11x faster than the same-shape BF16 `bmm` trace baseline.

Neither gate is enough for full active-MoE: SwiGLU and down projection need the
actual gate/up values, not only row hashes or timing. R5-C3B therefore exists
to prove that full D-row **values** can be materialized, slot-preserved, and
made ready for SwiGLU without changing promotion boundaries.

## Scope

R5-C3B uses a small batched selected-expert gate/up shape, not a full server or
RC run. The minimum valid smoke is:

```text
tokens >= 64
top_k >= 2
active experts >= 4
N_gateup is even
K_hidden is aligned to 32
```

The candidate path must:

1. run CUTLASS native NVF4+UE4M3 grouped gate/up with host-reference verification;
2. emit full D-row values, not only `d_hash` / `ref_hash`;
3. scatter D and reference values through the R5-C2B inverse order into
   `[T, top_k, N_gateup]`;
4. verify value-level D/ref parity after scatter;
5. split `N_gateup` into gate/up halves and record a host-side SwiGLU checksum.

## Required PASS Fields

```text
PASS_R5C3B_GATEUP_VALUE_MATERIALIZATION_SMOKE
banked_gateup_value_materialization=true
banked_host_swiglu_checksum_smoke=true
banked_down_projection_numeric_parity=false
banked_grouped_moe_fp4_mma_poc=false
banked_kernel_speed=false
banked_default_promotion=false
```

The result must also include:

| Field | Requirement |
|---|---|
| `full_d_row_values_captured` | true for D and reference rows, both schedules |
| `value_digest_matches_r5c2c_digest` | true, so values are tied back to the digest contract |
| `scatter_values_d_ref_match` | true after inverse-order scatter |
| `host_swiglu_checksum_recorded` | true, finite, deterministic |
| `fault_injections_detected` | swapped rows, missing rows, duplicate slots rejected |

## Explicit Non-Claims

- R5-C3B does not bank down projection.
- R5-C3B does not bank weighted top-k reduction.
- R5-C3B does not bank full grouped active-MoE speed.
- R5-C3B does not bank decode TPS.
- R5-C3B does not change runtime defaults.
- R5-C3B must not widen numeric thresholds if D/ref value parity fails.

## Next Gate

R5-C3C consumes the materialized `[T, top_k, I]` host-SwiGLU values and proves
down projection + weighted top-k reduction against a BF16/P3 reference. Only
after R5-C3C may a full grouped active-MoE prefill speed candidate be measured
as more than gate/up-only evidence.

## Local Static Gate

```bash
python3 scripts/test_stage6_r5c3b_gateup_value_materialization_contract_static.py
```

This static gate does not prove R5-C3B runtime success. It prevents digest-only
or timing-only artifacts from being promoted as full MoE evidence.
