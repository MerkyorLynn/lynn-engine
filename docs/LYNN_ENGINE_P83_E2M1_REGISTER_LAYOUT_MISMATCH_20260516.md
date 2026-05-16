# P83: E2M1 Register Layout Mismatch

Date: 2026-05-16

## Summary

P83 intentionally tried a simple all-ones contract:

```text
A = all +1.0 E2M1
B = all +1.0 E2M1
expected output = 32
```

and a signed control:

```text
A = all +1.0 E2M1
B = all -1.0 E2M1
expected output = -32
```

The kernel compiled and ran, but the contract failed.

## Result

Report:

```text
reports/p16_155/p83_sm120a_fp4_mma_allones_contract.json
```

Observed:

```text
positive_expected = 32
positive_min/max  = 2

negative_expected = -32
negative_min/max  = 10
```

## Interpretation

This does not invalidate P80/P82.

P80/P82 proved:

- `sm_120a` is the right feature target;
- the FP4 MMA instruction executes;
- non-zero operands produce finite non-zero accumulators.

P83 shows a different issue:

> The naive mapping from a 32-bit word full of Lynn E2M1 nibbles to CuTe's
> `SM120_16x8x32_TN` A/B registers is not the final mathematical tile layout.

The output values are suspiciously equal to the low nibble byte pattern
(`0x2 -> 2`, `0xA -> 10`) rather than a K=32 dot-product result. That means the
next problem is register/lane layout, not instruction availability.

## Decision

Proceed to P84 register-lane mapping.

P84 should sweep sparse one-hot nibble/register patterns for A and B and observe
which output accumulator changes. Only after that map is understood should
P85 wire real Lynn packed expert rows and per-16 scales.

