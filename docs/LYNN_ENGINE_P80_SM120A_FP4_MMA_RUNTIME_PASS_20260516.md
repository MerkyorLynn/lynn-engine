# P80: sm_120a FP4 MMA Runtime Smoke Pass

Date: 2026-05-16

## Summary

P78 failed under PyTorch's default `sm_120` target.

P79 showed that `sm_120a` and `compute_120a` can compile the CuTe E2M1 MMA
source.

P80 answers the runtime question: can a Lynn torch CUDA extension force
`sm_120a` and execute that instruction on the R6000?

Yes.

## Result

Report:

```text
reports/p16_155/p80_sm120a_fp4_mma_runtime_smoke.json
```

Passing variants:

| Variant | Flags | Runtime | Output |
|---|---|---:|---|
| `arch_sm120a` | `-arch=sm_120a` | PASS | `[1, 2, 3, 4]` |
| `gencode_sm120a` | `-gencode=arch=compute_120a,code=sm_120a` | PASS | `[1, 2, 3, 4]` |
| `arch_compute120a` | `-arch=compute_120a` | PASS | `[1, 2, 3, 4]` |

Failing control:

| Variant | Flags | Failure |
|---|---|---|
| `arch_compute120` | `-arch=compute_120` | `PTX JIT compilation failed` |

## Interpretation

The direct CUTLASS/CuTe FP4 MMA route is viable on the R6000, but only when the
native extension opts into the architecture-specific target.

The failure mode is now precise:

- default `sm_120`: rejects `.kind::f8f6f4`;
- generic `compute_120`: can compile PTX but cannot JIT at runtime;
- feature target `sm_120a` / `compute_120a`: compiles and runs.

## Engineering Decision

Add a native CUDA build policy for Blackwell feature targets:

```text
LYNN_NATIVE_CUDA_ARCH=sm_120a
```

or an automatic rule:

```text
if device capability == (12, 0) and FP4 MMA probe passes:
    append -arch=sm_120a / -gencode=arch=compute_120a,code=sm_120a
```

Then proceed to P81: write the first real FP4 MMA tile microbench against Lynn
active-expert shapes.

## P81 Shape Target

Start with the smallest shape that maps to the active expert bottleneck:

- A: selected activations, small M / decode batch 1;
- B: Lynn packed E2M1 weight nibbles;
- scale: keep out of the first MMA smoke, then add per-16 scale multiplication;
- output: compare against the existing scalar bridge reference.

The important shift is that we no longer need to treat direct FP4 MMA as
blocked. The route is open; the feature-target build policy is the missing
plumbing.

