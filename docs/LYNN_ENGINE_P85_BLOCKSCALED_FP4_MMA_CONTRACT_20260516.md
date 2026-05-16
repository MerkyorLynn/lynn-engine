# P85: Block-Scaled FP4 MMA Contract

Date: 2026-05-16

## Summary

P85 moves from the raw FP4 MMA used in P80-P84 to the CuTe Blackwell
block-scaled instruction shape:

```text
cute::SM120::BLOCKSCALED::SM120_16x8x32_TN_VS<
  float_e2m1_t, float_e2m1_t, float, float_ue8m0_t, 32>
```

This is closer to the official NVIDIA NVFP4/microscaling path because the MMA
consumes UE8M0 scale-factor registers instead of only raw E2M1 operand
registers.

## Result

Report:

```text
reports/p16_155/p85_sm120a_blockscaled_fp4_mma_contract.json
```

Build/runtime:

```text
cuda flags = ["-O3", "--use_fast_math", "-arch=sm_120a"]
device     = sm_120 R6000
instruction path = BLOCKSCALED E2M1 x E2M1 -> F32 with UE8M0 scale
```

The block-scaled instruction compiles and runs.

Zero-operand smoke passes:

```text
input accumulator = [1, 2, 3, 4]
zero operands     = [1, 2, 3, 4]
max_abs_err       = 0
```

Scale behavior matches the e8m0 trend seen in P17:

| scale byte for A/B | positive output | ratio |
|---:|---:|---:|
| 126 | 0.5 | 1x |
| 127 | 2.0 | 4x |
| 128 | 8.0 | 16x |

Because both operands use the same scale byte, each +1 scale-byte increment
multiplies the output by 4x.

## Important Boundary

The all-ones mathematical contract still does **not** pass:

```text
all +1 E2M1 x all +1 E2M1, K=32
naive expected = 32
observed at scale 127 = 2

all +1 E2M1 x all -1 E2M1, K=32
naive expected = -32
observed at scale 127 = 10
```

This mirrors P83/P84: the remaining issue is not instruction availability or
scale-factor availability. It is the operand fragment/register layout.

Manual `uint32_t` fill is still not the correct way to represent a mathematical
K=32 tile for this CuTe MMA wrapper.

## Interpretation

P85 proves three useful facts:

1. `sm_120a` is sufficient for the official block-scaled FP4 MMA instruction.
2. UE8M0 scale-factor inputs are accepted and affect output as expected.
3. The direct route must use CuTe's layout/copy abstractions or a derived
   fragment packer; guessing register words is no longer productive.

This is good news for the official route. The kernel target is alive. The next
work item is layout engineering, not another quantization format debate.

## Decision

Do **not** wire block-scaled MMA into active MoE yet.

Proceed to P86:

1. inspect CuTe `MMA_Atom` / `TiledMMA` helpers for SM120 block-scaled FP4;
2. build a minimal CuTe-layout contract test instead of manually filling A/B
   registers;
3. if that passes, move to a tiny real gate/up tile against scalar reference;
4. only then design the grouped per-16 active expert FFN kernel.

The official NVIDIA-style NVFP4 route remains viable, but it must be entered
through the correct fragment layout path.
