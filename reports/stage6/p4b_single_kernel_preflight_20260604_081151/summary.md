# Stage 6 P4B Single-Kernel Fail-Loud Preflight Summary

| Field | Value |
|---|---|
| Verdict | **PASS** (single-kernel fail-loud contract preflight passed; fused kernel still unbanked) |
| Decision | `PASS_SINGLE_KERNEL_FAILLOUD_CONTRACT` |
| Symbol | `active_moe_fused_zero_shadow_single_kernel_contract` |
| Device | `NVIDIA GB10` |
| Capability | `[12, 1]` |
| Torch/CUDA | `2.9.1+cu130` / `13.0` |
| Build dir | `/tmp/lynn_engine_native_build/p4b_single_kernel_20260604_001154` |
| Banked contract preflight | `True` |
| Banked fused kernel | `False` |
| Banked default promotion | `False` |
| Extension loaded | `True` |
| Symbol present | `True` |
| Call returned false | `True` |
| Fail-loud not implemented | `True` |
| Zero-shadow ABI | `True` |
| Packed byte budget | `True` |
| No inter_scratch ABI | `True` |
| Packed weight bytes | `18874376` |
| BF16 shadow-equivalent bytes | `50331648` |
| Packed/BF16 ratio | `0.3750001589457194` |
| Elapsed seconds | `25.386` |

## Tensor ABI

| Tensor | Shape | DType | Bytes | Contiguous |
|---|---|---|---:|---|
| `hidden` | `[1, 2048]` | `torch.bfloat16` | `4096` | `True` |
| `expert_ids` | `[1, 8]` | `torch.int32` | `32` | `True` |
| `routing_weights` | `[1, 8]` | `torch.float32` | `32` | `True` |
| `gate_up_packed` | `[8, 1024, 1024]` | `torch.uint8` | `8388608` | `True` |
| `gate_up_scale` | `[8, 1024, 128]` | `torch.float32` | `4194304` | `True` |
| `gate_up_global_scale` | `[1]` | `torch.float32` | `4` | `True` |
| `down_packed` | `[8, 2048, 256]` | `torch.uint8` | `4194304` | `True` |
| `down_scale` | `[8, 2048, 32]` | `torch.float32` | `2097152` | `True` |
| `down_global_scale` | `[1]` | `torch.float32` | `4` | `True` |
| `out` | `[1, 2048]` | `torch.bfloat16` | `4096` | `True` |

## Error Tail

```text
P4B single-kernel fused zero-shadow contract is not implemented yet; do not bank fused-kernel speed or promote this backend
```
