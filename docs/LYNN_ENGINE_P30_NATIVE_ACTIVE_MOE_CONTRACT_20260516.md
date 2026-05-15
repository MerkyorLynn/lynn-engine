# Lynn Engine P30 — Full Native Active-MoE Contract Probe (2026-05-16)

P28 validated native CUDA gate/up:

```text
hidden[2048] + gate_up packed NVFP4 -> inter[top_k, 512]
```

P29 validated native CUDA down:

```text
inter[top_k, 512] + routing + down packed NVFP4 -> hidden[2048]
```

P30 composes both halves and compares the complete active routed expert output
against the current production Triton path.

## Probe

Script:

```text
benchmarks/p30_native_active_moe_cuda_probe.py
```

Reference:

```text
nvfp4_grouped_gate_up_silu(...)
nvfp4_grouped_down_weighted_sum(...)
```

Candidate:

```text
module.gate_up_silu_scalar(...)
module.down_weighted_sum_scalar(...)
```

Model:

```text
/root/autodl-tmp/models/lynn-27b-variable-recovery-step5000-nvfp4-final
```

Representative layers:

```text
4, 16, 28, 36
```

## Result

| Layer | Triton active | CUDA scalar active | Ratio | Cosine | max_abs |
|---:|---:|---:|---:|---:|---:|
| 4 | 0.058587 ms | 0.067760 ms | 0.865x | 1.0 | 0 |
| 16 | 0.058458 ms | 0.065691 ms | 0.890x | 1.0 | 0 |
| 28 | 0.058486 ms | 0.064280 ms | 0.910x | 1.0 | 0 |
| 36 | 0.058059 ms | 0.064678 ms | 0.898x | 1.0 | 1.91e-6 |

## Decision

P30 **passes the full active-MoE native extension contract** and **does not
promote a speed path**.

This result is important because it proves the complete active routed expert
route can be hosted behind Lynn-owned C++/CUDA entrypoints:

```text
hidden
  -> native gate/up scalar
  -> native down scalar
  -> hidden
```

The speed path is still ahead: replace the scalar inner loops inside the same
contract with grouped native-FP4 tensor-core work. But P30 removes the largest
engineering uncertainty before that step: the production packed tensor layout,
scale contract, routing weights, and top-k expert ids are all correctly
understood on the native side.

In short:

```text
P27: native extension builds
P28: gate/up contract passes
P29: down contract passes
P30: full active-MoE contract passes
```

The next phase should focus on the math inside the native kernels, not on
Python/Triton plumbing.
