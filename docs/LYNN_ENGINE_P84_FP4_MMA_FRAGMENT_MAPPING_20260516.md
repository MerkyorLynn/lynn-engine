# P84: FP4 MMA Fragment Mapping Sweep

Date: 2026-05-16

## Summary

P84 converts the P83 failure into a usable map. The valid version runs the
warp-level MMA in all 32 lanes and sweeps sparse one-hot A/B register patterns.

The first attempt that only executed lane 0 was invalid for `mma.sync`; the
committed script uses one block per case with all lanes participating.

## Result

Report:

```text
reports/p16_155/p84_sm120a_fp4_mma_register_mapping.json
```

Compact report size:

```text
5.1 KB
```

Target/build:

```text
cuda flags = ["-O3", "--use_fast_math", "-arch=sm_120a"]
cases      = 1538
```

Observed summary:

| Case kind | Count | Nonzero cases | Nonzero count range | Sum range |
|---|---:|---:|---:|---:|
| baseline zero | 1 | 0 | 0 | 0 |
| baseline all-one | 1 | 1 | 128 | 256 |
| A one-hot | 1024 | 512 | 0-8 | -0.5 to 0 |
| B one-hot | 512 | 256 | 0-16 | -1.0 to 0 |

The all-one baseline still reports per-lane accumulators of `2.0`, not the naive
K=32 result. That confirms P83's conclusion: CuTe's FP4 MMA register fragment
layout is not a plain linear packed-byte K vector.

## Interpretation

The current direct-MMA work has now separated three things:

1. instruction availability: PASS (`sm_120a`);
2. raw loop execution: PASS;
3. Lynn packed-row layout: still unresolved.

The main blocker has shifted from "can we call Blackwell FP4 MMA?" to "how do
we arrange Lynn per-16 packed rows into the fragment layout expected by
`SM120_16x8x32_TN`?"

## Decision

Do not wire the current direct MMA into active MoE yet.

Proceed to P85 with one of two approaches:

1. use CuTe layout utilities instead of manual `uint32_t` register construction;
2. write a small fragment packer derived from the P84 sparse map, then verify a
   tiny real tile against scalar reference.

The route remains open, but P84 proves the remaining work is layout engineering,
not toolchain or storage compatibility.

