# R6000 Active-MoE Profile for W4A8 v2 - 2026-05-17

Report:

```text
reports/a100/r6000_p23_active_moe_profile_v2_20260517_120647.json
```

## Result

The profile runs all 40 layers against the R6000 v2 package with the safe
Config D-style runtime:

```text
LYNN_MOE_FAST_FIXED=1
LYNN_NATIVE_DOWN_BACKEND=triton
LYNN_ROUTER_TOPK_SORTED=0
```

Total measured MoE time across 40 layers:

| Component | Total ms/token | Share of full MoE |
|---|---:|---:|
| Router/top-k/softmax | 1.908 | 23.5% |
| Gate/up | 1.403 | 17.3% |
| Down | 1.297 | 16.0% |
| Active routed experts end-to-end | 2.582 | 31.8% |
| Shared expert | 2.286 | 28.2% |
| Full MoE path | 8.120 | 100.0% |
| Residual composition/dispatch | 1.344 | 16.5% |

Layer timing is nearly flat: the slowest layers are all about
`0.203-0.205 ms` for the full MoE path. This is not a single bad layer problem.

## Interpretation

The R6000 155 TPS target needs roughly this step-time movement:

```text
current safe server decode: ~100.45 tok/s -> ~9.95 ms/token
target:                    155 tok/s    -> ~6.45 ms/token
gap:                                      ~3.50 ms/token
```

Active routed experts alone are not enough. Even a perfect active-expert
replacement only accounts for about `2.58 ms/token`; the remaining gap also
requires router/top-k, shared expert, and Python/composition overhead work.

This explains why the `cudaTileDown` service sweep reached `~108.9 tok/s` but
was not promotable: it improves one component, while the full decode stack still
needs broader correctness-preserving changes.

## Next Speed Work

The next runtime milestones should be:

1. Keep Config D as the serving fallback.
2. Treat MTP as a required multiplier, not a cosmetic add-on.
3. Start a native active-MoE boundary that owns router output, gate/up, down,
   and shared expert accounting together, instead of only swapping the down
   projection.
4. If doing a C++/CUDA refactor, target a single per-layer active-MoE call that
   can later sit behind a reusable decode loop; optimizing one down kernel is
   confirmed insufficient.

The useful engineering target is no longer just "faster active experts"; it is
"reduce the whole 8.12 ms/token MoE budget while preserving structured gates."
