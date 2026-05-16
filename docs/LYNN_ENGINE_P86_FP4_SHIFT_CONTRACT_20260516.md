# P86: SM120 FP4 Shift Contract

Date: 2026-05-16

## Summary

P86 resolves the P83/P85 all-ones mismatch.

CuTe's SM120 MMA traits document a required E2M1 bit-placement transform before
feeding SM120 F8F6F4 / MXF8F6F4 MMA:

```text
0b0000ABCD -> 0b00ABCD00
word transform: (word << 2) & 0x3C3C3C3C
```

P83 and P85 manually filled MMA registers with unshifted Lynn E2M1 nibbles. P86
repeats both the raw MMA and the block-scaled MMA all-ones contracts with this
shift applied.

## Result

Report:

```text
reports/p16_155/p86_sm120a_fp4_shift_contract.json
```

Build/runtime:

```text
cuda flags = ["-O3", "--use_fast_math", "-arch=sm_120a"]
device     = sm_120 R6000
```

Raw shifted MMA:

```text
all +1 x all +1, K=32 -> +32
all +1 x all -1, K=32 -> -32
```

Block-scaled shifted MMA with neutral e8m0 byte 127:

```text
all +1 x all +1, K=32 -> +32
all +1 x all -1, K=32 -> -32
```

Both contracts are exact.

## Interpretation

P86 proves the earlier mismatch was not caused by:

- unsupported SM120 FP4 MMA;
- incorrect CUTLASS E2M1 value encoding;
- invalid UE8M0 scale-factor inputs;
- a dead official NVFP4 route.

The root cause was the missing CuTe FP4 register-placement shift.

This is the first clean mathematical contract for the direct SM120 FP4 MMA
path.

## Decision

Promote the following rule into all future native FP4 expert kernels:

```text
Before MMA: shift every E2M1 byte container left by 2 bits:
  packed_mma_word = (packed_lynn_word << 2) & 0x3C3C3C3C
```

Next step P87:

1. construct a tiny real gate/up selected-row tile;
2. apply the shift to Lynn packed E2M1 register words;
3. use blockscaled neutral scale first, then real UE8M0 scale variants;
4. compare one tile against scalar reference before designing the grouped
   per-16 active expert FFN.

P86 changes the project state: Blackwell native FP4 MMA is now an engineering
layout problem, not an instruction feasibility problem.
