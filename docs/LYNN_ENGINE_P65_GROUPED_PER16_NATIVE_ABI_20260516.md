# Lynn Engine P65: grouped per-16 native active-MoE ABI

Date: 2026-05-16
Branch: `codex/p16-r6000-155-tps`

## Why P65

P16/P23/P31/P64 all point at the same remaining bottleneck: active routed
experts. Small Triton tile retunes can produce local speed signals, but strict
full-generate gates reject the unsafe variants. The next real step is a grouped
per-16 native-FP4 active expert FFN kernel.

P65 does not pretend the tensor-core kernel exists yet. It freezes the runtime
ABI and shape/layout checks so the future CUTLASS/custom CUDA implementation can
replace the inner math without changing Python or artifact layout.

## Runtime backend

```bash
export LYNN_NATIVE_ACTIVE_MOE_BACKEND=grouped_per16
```

This backend now enters the native CUDA extension and runs shape/layout checks
for the Lynn-native NVFP4 active expert tensors. It intentionally fails after
the checks with a clear message until the kernel implementation lands.

## Contract

Inputs:

| Tensor | Shape | DType | Meaning |
|---|---:|---|---|
| `x` | `[2048]` | BF16 | one decode hidden state |
| `expert_ids` | `[top_k]` | int32 | routed active experts, currently top-k=8 |
| `routing_weights` | `[top_k]` | FP32 | softmax router weights |
| `gate_up_packed` | `[experts, 1024, 1024]` | uint8 | E2M1 packed gate/up rows |
| `gate_up_scale` | `[experts, 1024, 128]` | FP32 | per-16 scales |
| `gate_up_global_scale` | `[1]` | FP32 | global scale |
| `down_packed` | `[experts, 2048, 256]` | uint8 | E2M1 packed down rows |
| `down_scale` | `[experts, 2048, 32]` | FP32 | per-16 scales |
| `down_global_scale` | `[1]` | FP32 | global scale |

Output target:

```text
out: [2048] BF16/FP32 accumulation result, returned as BF16-compatible runtime output
```

## Why this differs from the vendor route

The vendor/official NVFP4 route usually optimizes for standard grouped GEMM
layouts and larger serving batches. Lynn-native grouped per-16 optimizes a
different production point:

- single-user resident decode;
- variable expert physical counts after pruning;
- top-k=8 routed active experts;
- Lynn per-16 scale contract;
- strict greedy stability and low jitter rather than only peak throughput.

Both layouts can coexist. Vendor NVFP4 is the compatibility/ecosystem route;
Lynn-native per-16 is the specialized low-memory, quality-locked route.

## Current status

P65 wires:

- `csrc/lynn_native/bindings.cpp`
  - `active_moe_grouped_per16_contract`
- `csrc/lynn_native/moe_scalar_kernel.cu`
  - full shape/layout guard for the grouped per-16 ABI
- `engine/moe_packed_nvfp4.py`
  - `LYNN_NATIVE_ACTIVE_MOE_BACKEND=grouped_per16` now enters the native
    contract instead of failing in Python

Expected behavior today:

```text
active_moe_grouped_per16_contract passed shape/layout checks, but the grouped
per-16 native-FP4 active expert FFN kernel is not implemented yet.
```

R6000 validation:

```text
P65_CONTRACT_RUNTIME_ERROR
active_moe_grouped_per16_contract passed shape/layout checks, but the grouped
per-16 native-FP4 active expert FFN kernel is not implemented yet. This guarded
ABI exists so the future CUTLASS/custom CUDA kernel can replace only the inner
math without changing Python/runtime layout.
```

This confirms that the native extension builds on sm_120 and that a real layer-0
Lynn-native NVFP4 tensor set satisfies the guarded ABI.

## Next implementation step

Replace the guarded failure with one of two implementation paths:

1. CUTLASS/CuTe grouped FP4 MMA using the Lynn per-16 scale contract.
2. Custom CUDA decode-specialized kernel that keeps per-16 scales exact and
   fuses gate/up SiLU with down accumulation for top-k=8.

Promotion gates remain unchanged:

- P37 full-generate greedy IDs;
- P62 first-divergence tracing if drift appears;
- tool-call / no-think / long-context smoke before serving default.
