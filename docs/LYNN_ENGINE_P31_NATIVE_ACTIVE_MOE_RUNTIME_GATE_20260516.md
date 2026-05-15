# Lynn Engine P31 — Native Active-MoE Runtime Gate (2026-05-16)

P30 proved the isolated active routed expert contract:

```text
native CUDA gate/up scalar + native CUDA down scalar
    == production Triton active path
```

P31 wires that path into the production MoE function behind an explicit opt-in:

```bash
export LYNN_NATIVE_ACTIVE_MOE_BACKEND=cuda_scalar
```

Default remains:

```bash
export LYNN_NATIVE_ACTIVE_MOE_BACKEND=triton
```

## Probe

Script:

```text
benchmarks/p31_native_active_moe_runtime_gate.py
```

The probe calls the same production function:

```text
engine.moe_packed_nvfp4.moe_forward_decode_packed_nvfp4(...)
```

twice per layer:

```text
LYNN_NATIVE_ACTIVE_MOE_BACKEND=triton
LYNN_NATIVE_ACTIVE_MOE_BACKEND=cuda_scalar
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

| Layer | Triton backend | CUDA scalar backend | Speedup | Cosine | max_abs |
|---:|---:|---:|---:|---:|---:|
| 4 | 0.236502 ms | 0.148867 ms | 1.59x | 1.00000024 | 0 |
| 16 | 0.224349 ms | 0.149584 ms | 1.50x | 1.0 | 0 |
| 28 | 0.220912 ms | 0.149110 ms | 1.48x | 1.00000012 | 0 |
| 36 | 0.226803 ms | 0.148339 ms | 1.53x | 1.0 | 0 |

## Interpretation

P28-P30 showed the isolated CUDA scalar kernels are not faster than the isolated
Triton kernels. P31 measures the production MoE function boundary instead, where
the opt-in native backend avoids some Python/Triton wrapper overhead and becomes
faster at the full function level.

This is a real signal, but not enough to change the default yet.

## Decision

P31 passes as an **opt-in runtime backend** and remains **off by default**.

Before promotion, it needs:

1. full-token decode parity across the 6-prompt suite;
2. full graph / server TPS measurement;
3. no regression in strict tool-call and no-think loop guards.

If those gates pass, `cuda_scalar` may become a production bridge while the
inner loops are upgraded to grouped native-FP4 tensor-core math.
