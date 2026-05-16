# P88: Real Lynn Gate/Up FP4 Tile Contract

Date: 2026-05-16

## Summary

P88 moves the SM120 block-scaled FP4 MMA path from synthetic data to real Lynn
27B NVFP4 tensors.

It loads:

- a real decode activation from Lynn 27B;
- a real routed expert id from the layer router;
- real packed gate/up rows from the Lynn-native NVFP4 artifact;
- real packed activation codes from the production activation quantizer.

Then it runs one 16x32 x 8x32 SM120 block-scaled FP4 MMA tile with the P86
E2M1 placement shift and the P87 layout formulas.

P88 intentionally compares code-level dot products first. Real per-16 floating
scales are recorded but not applied yet, so a failure would isolate indexing or
fragment layout rather than scale conversion.

## Result

Report:

```text
reports/p16_155/p88_sm120a_real_gateup_tile_contract.json
```

Configuration:

```text
model       = lynn-27b-variable-recovery-step5000-nvfp4-final
layer       = 28
expert slot = 0
expert id   = 116
row offset  = 0
k offset    = 0
scale byte  = 127
```

Contract:

```text
max_abs_err  = 0
mean_abs_err = 0
all_exact    = true
```

Observed output range:

```text
observed = [-119.25, 102.0]
expected = [-119.25, 102.0]
```

Sample lane:

```text
(m=0,n=0): observed  73.5, expected  73.5
(m=0,n=1): observed 102.0, expected 102.0
(m=8,n=0): observed  73.5, expected  73.5
(m=8,n=1): observed 102.0, expected 102.0
```

## Interpretation

P88 proves that real Lynn packed tensors can be consumed by SM120
block-scaled FP4 MMA when we apply the discovered rules:

1. use `sm_120a`;
2. use the block-scaled FP4 MMA variant;
3. shift E2M1 byte containers left by two bits before MMA;
4. follow CuTe A/B/C register layouts.

This is no longer just a synthetic MMA smoke test. It is a real-model,
real-router, real-packed-weight tile proof.

## Remaining Risk

P88 deliberately does not solve Lynn's current per-16 floating scale contract.

The report records the real scale values for the tested tile:

```text
activation native scale sample = [0.3125, 0.28125, 0.34375, 0.28125, ...]
weight per-16 scale rows       ~= 0.0022 to 0.0047
weight global scale            = 1.0
```

The next problem is scale handling:

- either apply per-16 scales outside/around MMA;
- or generate a second official-style e8m0/block-scaled artifact;
- or build a custom kernel that fuses scale application with grouped active
  expert accumulation.

## Decision

Proceed to P89:

1. extend the real tile to include Lynn per-16 scales in the scalar reference;
2. test a neutral-scale MMA + explicit post-scale strategy for K=32 tiles;
3. quantify the error and cost of per-16 scale application;
4. decide whether the grouped active expert kernel should consume the current
   Lynn-native artifact or require a second official NVFP4 artifact.

P88 is the green light for real kernel construction. The blocker has narrowed
to scale strategy, not instruction/layout correctness.
