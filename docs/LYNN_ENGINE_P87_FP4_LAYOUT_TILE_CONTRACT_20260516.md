# P87: SM120 Block-Scaled FP4 Layout Tile Contract

Date: 2026-05-16

## Summary

P87 proves the SM120 block-scaled FP4 MMA path can compute a non-uniform logical
tile exactly when we follow CuTe's published register layouts and the P86 E2M1
bit-placement shift.

The tested instruction:

```text
SM120::BLOCKSCALED::SM120_16x8x32_TN_VS<
  float_e2m1_t, float_e2m1_t, float, float_ue8m0_t, 32>
```

The tested logical shape:

```text
A: 16 x 32
B:  8 x 32
C: 16 x 8
```

## Result

Report:

```text
reports/p16_155/p87_sm120a_fp4_layout_tile_contract.json
```

Build/runtime:

```text
cuda flags = ["-O3", "--use_fast_math", "-arch=sm_120a"]
scale byte = 127
```

Contract result:

```text
max_abs_err  = 0
mean_abs_err = 0
all_exact    = true
```

The tile used non-uniform synthetic values, not an all-ones shortcut:

```text
observed range = [-2, 4]
expected range = [-2, 4]
```

Sample lane:

```text
(m=0,n=0): observed  4, expected  4
(m=0,n=1): observed -2, expected -2
(m=8,n=0): observed  0, expected  0
(m=8,n=1): observed  2, expected  2
```

## Interpretation

P87 closes the fragment-layout risk left by P83-P86.

We now know:

1. `sm_120a` executes the Blackwell FP4 MMA instruction.
2. CUTLASS E2M1 encoding matches Lynn's E2M1 value table.
3. E2M1 bytes must be shifted left by two bits before MMA.
4. CuTe's A/B/C layouts can be implemented directly and match scalar logical
   math exactly.
5. UE8M0 neutral scale byte 127 works for this neutral-scale contract.

This is the first exact non-uniform tile proof for the native FP4 route.

## Decision

Proceed to P88 with a tiny real Lynn gate/up tile:

1. load one real layer/expert gate/up packed tensor;
2. select 16 output rows and 32 hidden columns;
3. pack shifted E2M1 registers using the P87 layout formulas;
4. compare block-scaled FP4 MMA output against the existing scalar bridge for
   that tile;
5. only after tile parity, design the grouped per-16 active expert FFN.

P87 is a green light for the custom grouped active expert kernel. The next risk
is no longer fragment layout; it is adapting Lynn's real per-16 FP32/e4m3-ish
scale contract into a fast scale path.
