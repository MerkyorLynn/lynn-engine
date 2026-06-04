# Stage 6 R5-C2B Slot-Preserving Selected-Output Contract

Date: 2026-06-04

Verdict: **contract/invariant gate only; no CUTLASS kernel, speed claim, server
claim, RC claim, or default promotion is banked by this document.**

## Why R5-C2B Exists

R5-C2 banked a selected-expert gate/up numeric smoke by mapping
`tokens_per_expert=[32,64,64,96]` to CUTLASS 79d per-group `M` shapes. That was
the right next bridge after R5-C2A, but it still does not prove Lynn can preserve
the selected-output order expected by MoE runtime code.

The next failure-prone boundary is not FP4 math; it is routing semantics:

```text
expert-grouped GEMM rows  ->  [T, top_k, inter] selected-output slots
```

R5-C2B therefore freezes the slot-preserving contract before the next CUDA/CUTLASS
harness is written.

## Required Route Metadata

Any R5-C2B implementation must explicitly carry:

| Field | Meaning |
|---|---|
| `token_idx` | Original token row before expert grouping |
| `top_k_slot` | Slot inside the router top-k result for that token |
| `expert_id` | Routed expert for `(token_idx, top_k_slot)` |
| `pair_order` | Stable list of `(token_idx, top_k_slot, expert_id)` pairs before grouping |
| `grouped_order` | Same pairs sorted/grouped by `expert_id` for grouped GEMM |
| `tokens_per_expert` | Count of routed top-k pairs per expert; must sum to `T * top_k` |
| `expert_offsets` | Prefix offsets into the expert-grouped output rows |
| `inverse_order` | Scatter map from grouped rows back to original `(token_idx, top_k_slot)` |

## Invariant Gate

A PASS requires:

```text
PASS_R5C2B_SLOT_PRESERVING_SELECTED_OUTPUT_CONTRACT
banked_slot_bridge_contract=true
banked_selected_output_kernel=false
banked_grouped_moe_fp4_mma_poc=false
banked_kernel_speed=false
banked_default_promotion=false
```

The GPU-free invariant fixture must encode and guard:

- `sum(tokens_per_expert) == T * top_k`.
- Each `(token_idx, top_k_slot)` appears exactly once in `pair_order`.
- `top_k` slots for one token preserve slot identity; they are not deduplicated by
  expert ID.
- `grouped_order` is grouped by `expert_id` and records per-expert prefix
  offsets.
- `inverse_order` scatters synthetic grouped rows exactly back to `[T, top_k]`.
- Fault-injection checks reject swapped grouped rows, duplicate scatter slots, and
  missing inverse entries.
- The contract includes the selected-output tensor shape `[T, top_k, inter]`.

## Explicit Non-Claims

- R5-C2B does not prove a CUTLASS selected-output kernel exists.
- R5-C2B does not prove down projection, full MoE composition, speed, server
  behavior, RC quality, or runtime defaults.
- R5-C2B is a route/order/scatter contract. The next GPU gate must use this
  contract when wiring CUTLASS output rows back to Lynn selected slots.
