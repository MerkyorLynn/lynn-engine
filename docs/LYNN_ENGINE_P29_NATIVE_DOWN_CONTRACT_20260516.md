# Lynn Engine P29 — Native CUDA Down Contract Probe (2026-05-16)

P28 proved the native CUDA extension can consume Lynn 27B's grouped packed
NVFP4 gate/up tensors and produce the active expert intermediate:

```text
inter[top_k, 512]
```

P29 covers the second half of the active routed expert path:

```text
inter[top_k, 512]
+ expert_ids[top_k]
+ routing_weights[top_k]
+ down_packed[experts, 2048, 256]
+ down_scale[experts, 2048, 32]
+ down_global_scale[1]
    -> hidden[2048]
```

As with P28, this is a scalar CUDA reference. It is not the final fast path.
The purpose is to prove the native extension data contract for the full active
MoE route before replacing the inner math with grouped native-FP4 work.

## Probe

Script:

```text
benchmarks/p29_native_down_cuda_probe.py
```

Reference:

```text
triton_kernels.nvfp4_moe.nvfp4_grouped_down_weighted_sum
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

| Layer | Triton reference | CUDA scalar | Ratio | Cosine | max_abs |
|---:|---:|---:|---:|---:|---:|
| 4 | 0.027218 ms | 0.030960 ms | 0.879x | 1.0 | 0 |
| 16 | 0.026552 ms | 0.030979 ms | 0.857x | 1.0 | 0 |
| 28 | 0.026819 ms | 0.030526 ms | 0.879x | 1.0 | 0 |
| 36 | 0.026342 ms | 0.029608 ms | 0.890x | 1.0 | 1.91e-6 |

## Decision

P29 **passes the contract gate** and **does not promote a speed path**.

Together, P28 and P29 establish that Lynn's native CUDA extension can consume
the real production packed NVFP4 active expert tensors end to end:

```text
hidden -> gate/up intermediate -> down weighted sum
```

The next step is P30: compose these two callable halves into a full active MoE
native-extension probe, then use that stable contract as the insertion point
for real grouped native-FP4 tensor-core work.
