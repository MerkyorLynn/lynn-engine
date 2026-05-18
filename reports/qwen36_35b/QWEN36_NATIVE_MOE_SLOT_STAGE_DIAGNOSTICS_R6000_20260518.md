# Qwen3.6 35B Native MoE Slot Stage Diagnostics

Date: 2026-05-18

## Summary

P137 split the `native_slot_output_owned_bf16` fixture candidate into gate/up intermediate generation and down weighted-sum. The goal was to determine whether the strict RED result came from slot-vs-unique ordering, native gate/up, or native down.

The answer: the scalar native slot path is fast, but it does not match PyTorch/TensorCore BF16 semantics closely enough for the strict fixture gate.

| Probe | Max Abs / Mean |
| --- | --- |
| native inter vs PyTorch slot inter | 0.125 max_abs |
| native down with PyTorch inter vs PyTorch slot output | 0.00390625 max_abs |
| PyTorch down with native inter vs PyTorch slot output | 0.00390625 max_abs |
| native full vs PyTorch slot output | 0.00390625 max_abs |
| native full cosine min | 0.9999786615371704 |
| native full latency mean | 0.0517 ms |
| native inter latency mean | 0.0164 ms |
| native down latency mean | 0.0375 ms |

## Reduction Shape Probe

Changing Stage 1 from 256 threads to 128 threads did not improve numerical drift and slowed the candidate:

| Stage 1 Threads | Full Max Abs | Full Cosine Min | Full Latency Mean |
| --- | --- | --- | --- |
| 256 | 0.00390625 | 0.9999786615371704 | 0.0517 ms |
| 128 | 0.00390625 | 0.9999786615371704 | 0.0558 ms |

This closes the cheap "thread-count reduction shape" hypothesis. The issue is not fixed by changing the scalar reduction width.

## Interpretation

The native scalar route is not wrong in a gross layout sense; it is consistently close and faster than the current Triton active fixture reference. But it is not a strict replacement:

- Gate/up intermediate values differ from the PyTorch slot reference by up to 0.125 in later fixtures.
- Native down also differs from PyTorch down by up to 0.00390625 even when fed PyTorch intermediates.
- The final error is small in cosine terms but too large for exact-greedy serving promotion.

The next useful implementation direction is not another scalar-thread sweep. It is either:

1. a TensorCore/cuBLAS-like gate/up and down path that matches the PyTorch BF16 GEMM contract more closely, or
2. a fused native path with an explicit service-level drift budget, kept out of default promote until P37/P25/structured gates prove stable.

Artifacts:

- `reports/qwen36_35b/p137_moe_slot_stage_diagnostics_20260518.json`
- `reports/qwen36_35b/p137_moe_slot_stage_diagnostics_s1t128_20260518.json`
