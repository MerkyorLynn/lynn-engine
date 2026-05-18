# Qwen3.6 35B W4A16 Native MoE P154 Gate/Up Order Probe

Date: 2026-05-19

## Purpose

P153 identified gate/up as the main source of native packed-MoE drift. P154
tests whether changing native gate/up from a full-hidden reduction tree to a
Triton-like `BLOCK_HIDDEN=256` reduction order closes the gap.

## Result

Verdict: **TRITON_ORDER_STILL_DRIFTS**

| Variant | Inter Exact | Inter Max Abs | Mean Gate/Up Latency |
|---|---:|---:|---:|
| existing native inter | 6/18 | 2.44140625e-4 | 0.04134 ms |
| Triton-order native inter | 10/18 | 2.44140625e-4 | 0.06577 ms |
| Triton-order full output | 13/18 | 2.44140625e-4 | n/a |

The hidden-block reduction order is a real contributor: exact rows improved
from 6/18 to 10/18. It is not the whole fix: the variant is slower and still
not bit-exact. Per-item `scale / global_scale` versus precomputed reciprocal
made no measurable difference.

## Notable Rows

| Fixture | Existing Inter Max Abs | Triton-Order Inter Max Abs | Full Output Max Abs |
|---|---:|---:|---:|
| L28/P00 | 2.44140625e-4 | 7.45e-9 | 0 |
| L08/P01 | 2.44140625e-4 | 0 | 0 |
| L39/P00 | 7.63e-6 | 2.44140625e-4 | 2.44140625e-4 |

## Interpretation

The next exactness blocker is likely not down/reduce and not simply reciprocal
scale handling. The remaining candidates are:

1. exact Triton `tl.sum` reduction tree shape inside each 256-column block;
2. `tl.sigmoid` / SiLU approximation semantics;
3. BF16 store rounding at the inter boundary after a slightly different FP32
   accumulator.

## Artifacts

- `reports/qwen36_35b/p154_native_packed_gateup_order_20260519_040430_div.json`

## Next Step

Build a raw gate/up accumulator diagnostic that records gate_acc and up_acc
before SiLU/BF16 store for both Triton and native. That will separate reduction
drift from SiLU/BF16 boundary drift.
