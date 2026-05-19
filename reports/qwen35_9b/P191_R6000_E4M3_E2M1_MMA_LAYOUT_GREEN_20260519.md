# P191 R6000 E4M3 x E2M1 MMA Layout GREEN

Date: 2026-05-19

## Result

The real SM120a `E4M3 x E2M1 -> F32` MMA path now computes correctly after
switching to the P87/P88 fragment layout formulas.

| Layer | Scalar ms | MMA ms | Speedup | MMA vs scalar cosine | Max abs |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.5260 | 0.0682 | 7.72x | 0.99999982 | 0.03125 |
| 16 | 0.5263 | 0.0687 | 7.66x | 0.99999994 | 0.015625 |

## Fix

- A fragment: use the P87/P88 lane/value mapping and write raw E4M3 bytes.
- B fragment: use the P87/P88 lane/value mapping and shift E2M1 nibbles left by
  two bits before MMA.
- C fragment: write only row `m == 0` using the P87/P88 C coordinate mapping.
- Validation: compare against a raw FP8 x FP4 reference that intentionally
  excludes Lynn per-16 scales, isolating fragment-layout correctness.

## Meaning

This clears the fragment-layout blocker.  The remaining work is production
plumbing:

1. consume the P192 pretransposed sidecar instead of opening model shards;
2. apply Lynn per-16 activation/weight scales around the raw MMA dot;
3. fuse gate/up/down into a resident candidate and run P193/P37/P25 gates.

