# P82: sm_120a FP4 MMA Loop Microbench

Date: 2026-05-16

## Summary

P80 proved a single CuTe E2M1 MMA instruction can execute on R6000 when the
extension is built for `sm_120a`.

P81 wired the target into the shared Lynn native CUDA loader.

P82 validates the next gate: a non-zero raw E2M1 MMA loop can run and be timed
under that shared policy.

## Result

Report:

```text
reports/p16_155/p82_sm120a_fp4_mma_loop_microbench.json
```

Build flags:

```text
["-O3", "--use_fast_math", "-arch=sm_120a"]
```

Timing shape:

```text
blocks = 4096
iters = 512
warp MMA instructions = 2,097,152
```

Observed:

```text
median_ms = 0.032304
min_ms    = 0.030624
mean_ms   = 0.039376
```

Output sample:

```text
[1505, 1506, 1539, 1540, ...]
```

The output is finite and non-zero, so this is no longer the all-zero smoke from
P80.

## Interpretation

The raw FP4 MMA instruction path is now operational enough for real kernel
development:

- feature target policy works;
- CuTe E2M1 MMA loops execute;
- event timing works;
- non-zero E2M1 operands accumulate into FP32 outputs.

## Decision

Proceed to P83: feed Lynn packed E2M1 bytes into the MMA register layout and
compare a small tile against a scalar reference.

P83 should still ignore per-16 scales at first. Get the packed-code-to-register
layout correct, then add scale multiplication as the next layer.

