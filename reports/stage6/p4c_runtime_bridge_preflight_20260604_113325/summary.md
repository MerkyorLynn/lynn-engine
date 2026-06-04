# Stage 6 P4C Active-Reuse Runtime Bridge Preflight Summary

| Field | Value |
|---|---|
| Verdict | **PASS** (P4C active-reuse runtime bridge returns caller-owned two-phase output) |
| Decision | `PASS_P4C_ACTIVE_REUSE_RUNTIME_BRIDGE` |
| Model | `/home/merkyor/models/Qwen3.6-35B-A3B-lynn-native-w4a16-nvfp4-from-bf16-20260526` |
| Layer | `0` |
| Expected backend | `fused_zero_shadow_active_reuse_contract` |
| Native layer selected | `True` |
| Native backend call delta | `1` |
| Banked runtime bridge preflight | `True` |
| Banked fused kernel | `False` |
| Banked default promotion | `False` |
| Baseline norm | `0.26544320583343506` |
| Candidate norm | `0.26544320583343506` |
| Candidate rel L2 vs baseline | `0.0` |
| Candidate max abs diff vs baseline | `0.0` |
| Packed tensors present | `True` |
| Active scratch present | `True` |
| Active shadows removed | `True` |
| Candidate output returned | `True` |
| Candidate numeric vs Triton | `True` |
| Elapsed seconds | `387.548` |

## Removed Active Shadows

| Key | Shape | DType | Bytes |
|---|---|---|---:|
| `mlp.experts.gate_up_proj` | `[256, 1024, 2048]` | `torch.bfloat16` | `1073741824` |
| `mlp.experts.down_proj` | `[256, 2048, 512]` | `torch.bfloat16` | `536870912` |

## Active Scratch

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

## P4C Boundary

- Candidate gate/up tile_inter: `2`.
- Native call recorded gate/up tile_inter: `2`.
- This banks only `banked_p4c_active_reuse_runtime_bridge=true`.
- It keeps `banked_fused_kernel=false` and `banked_default_promotion=false`.
- It is **P4C**, not P4B: caller-owned `active[top_k,512]` scratch is allowed and must be reported.
