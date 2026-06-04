# Stage 6 P4B Runtime Bridge Fail-Loud Preflight Summary

| Field | Value |
|---|---|
| Verdict | **PASS** (real runtime bridge reaches P4B fail-loud symbol; fused kernel still unbanked) |
| Decision | `PASS_P4B_RUNTIME_BRIDGE_FAILLOUD` |
| Model | `/home/merkyor/models/Qwen3.6-35B-A3B-lynn-native-w4a16-nvfp4-from-bf16-20260526` |
| Layer | `0` |
| Expected backend | `fused_zero_shadow_single_kernel_contract` |
| Native layer selected | `True` |
| P4B native call delta | `1` |
| P4B last shapes | `{'hidden': [1, 2048], 'expert_ids': [1, 8], 'out': [1, 2048]}` |
| Banked P4B runtime bridge preflight | `True` |
| Banked fused kernel | `False` |
| Banked default promotion | `False` |
| Baseline norm | `0.26544320583343506` |
| Candidate returned | `False` |
| P4B fail-loud not implemented | `True` |
| P4B last shapes out-only | `True` |
| Packed tensors present | `True` |
| Active out scratch present | `True` |
| Active shadows removed | `True` |
| Elapsed seconds | `356.476` |

## Removed Active Shadows

| Key | Shape | DType | Bytes |
|---|---|---|---:|
| `mlp.experts.gate_up_proj` | `[256, 1024, 2048]` | `torch.bfloat16` | `1073741824` |
| `mlp.experts.down_proj` | `[256, 2048, 512]` | `torch.bfloat16` | `536870912` |

## Active Scratch Manifest

| Key | Shape | DType | Bytes |
|---|---|---|---:|
| `mlp.experts._active_inter_scratch` | `[8, 512]` | `torch.bfloat16` | `8192` |
| `mlp.experts._active_out_scratch` | `[2048]` | `torch.bfloat16` | `4096` |

## Packed Tensor Inputs

| Key | Shape | DType | Bytes |
|---|---|---|---:|
| `mlp.experts._gate_up_packed` | `[256, 1024, 1024]` | `torch.uint8` | `268435456` |
| `mlp.experts._gate_up_scale` | `[256, 1024, 128]` | `torch.float32` | `134217728` |
| `mlp.experts._gate_up_global_scale` | `[]` | `torch.float32` | `4` |
| `mlp.experts._down_packed` | `[256, 2048, 256]` | `torch.uint8` | `134217728` |
| `mlp.experts._down_scale` | `[256, 2048, 32]` | `torch.float32` | `67108864` |
| `mlp.experts._down_global_scale` | `[]` | `torch.float32` | `4` |

## Error Tail

```text
P4B single-kernel fused zero-shadow contract is not implemented yet; do not bank fused-kernel speed or promote this backend
```
