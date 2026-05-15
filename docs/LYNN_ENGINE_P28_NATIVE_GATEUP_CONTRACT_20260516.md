# Lynn Engine P28 — Native CUDA Gate/Up Contract Probe (2026-05-16)

P27 proved that R6000 can build and launch a Lynn-owned CUDA extension. P28
moves from a smoke kernel to the first real active-MoE tensor contract:

```text
x[2048]
+ expert_ids[top_k]
+ gate_up_packed[experts, 1024, 1024]
+ gate_up_scale[experts, 1024, 128]
+ gate_up_global_scale[1]
    -> inter[top_k, 512]
```

The implementation is intentionally a scalar reference kernel in
`csrc/lynn_native/moe_scalar_kernel.cu`. It does not yet use tensor cores or the
final grouped native-FP4 math. Its job is to prove that the native extension
path understands Lynn's real packed NVFP4 layout, per-16 scale contract, and
top-k routed expert ids.

## Probe

Script:

```text
benchmarks/p28_native_gateup_cuda_probe.py
```

Reference:

```text
triton_kernels.nvfp4_moe.nvfp4_grouped_gate_up_silu
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
| 4 | 0.034829 ms | 0.035634 ms | 0.977x | 1.0 | 0 |
| 16 | 0.034218 ms | 0.035219 ms | 0.972x | 0.99999994 | 2.98e-8 |
| 28 | 0.034387 ms | 0.035194 ms | 0.977x | 0.99999994 | 0 |
| 36 | 0.033926 ms | 0.035226 ms | 0.963x | 1.00000012 | 0 |

## Decision

P28 **passes the contract gate** and **does not promote a speed path**.

This is still a scalar bridge, just now inside a native CUDA extension. The
important result is that the C++/CUDA side can consume the exact same production
packed tensors as the current Triton path and match it numerically across
multiple layers.

The next optimization should replace the scalar inner loop with actual grouped
native-FP4 math while keeping this callable contract intact:

```text
torch extension entrypoint
    -> same inputs / same output / same parity gate
    -> progressively swap scalar math for native FP4 tensor-core work
```

This keeps the route to 155 TPS honest: P28 removes data-contract ambiguity,
but the real speedup still requires a native grouped expert kernel, not just a
different launch wrapper.
