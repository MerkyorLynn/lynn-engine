# Qwen3.6 35B W4A16 Native MoE Gate/Up Debug Matrix

Date: 2026-05-19

This note is the current exactness map for the Native MoE gate/up line.  It is
intended to keep the next kernel experiments narrow: no resident P37/P25 run is
allowed until the fixture-stage Triton contract is exact.

## Known Gates

| Probe | Scope | Exact Result | Speed Signal | Status |
|---|---|---:|---:|---|
| P147 | Triton reference contract | reference ready | gate/up 0.0685 ms, down 0.0265 ms | GREEN reference |
| P152 | native packed full stage vs P147 | 12/18 | native full 0.09036 ms | CLOSED_STAGE_DRIFT |
| P153 | split native packed stage | inter 6/18, down(Triton inter) 15/18, full 12/18 | inter 0.04132 ms, down 0.04946 ms | GATEUP_DRIFT |
| P154 | Triton-like hidden-block reduction order | inter 10/18, full 13/18 | inter 0.06577 ms | still drift, slower |

## Current Diagnosis

The down side is not the main blocker.  When fed the Triton BF16 intermediate,
native down is exact for 15/18 fixtures with max_abs <= 2.384e-7.  The packed
native full-path failures follow native intermediate drift.

The hidden-block reduction order matters but is insufficient.  It fixed several
large rows, including L28/P00 and L08/P01, but introduced or preserved drift on
others such as L39/P00.

## Next Diagnostic

P155 should compare raw gate_acc and up_acc before SiLU/BF16 store:

1. Triton raw gate/up accumulator reference.
2. Native existing-order raw accumulator.
3. Native Triton-order raw accumulator.
4. Post-SiLU BF16 intermediate.

Interpretation:

| P155 Outcome | Next Action |
|---|---|
| raw accumulators drift | match Triton reduction tree or FP4 decode/scale semantics |
| raw exact but post-SiLU drifts | match Triton sigmoid/SiLU approximation |
| raw and post-SiLU exact | rerun P153/P147; down/final-store is next suspect |

## Hard Rule

Do not promote any Native MoE packed candidate to resident P37 unless the P147
stage contract is exact across all 18 fixtures.
