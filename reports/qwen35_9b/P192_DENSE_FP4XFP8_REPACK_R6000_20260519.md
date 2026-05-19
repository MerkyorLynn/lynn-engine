# Qwen3.5-9B Dense FP4xFP8 Offline Repack — R6000 Validation

Date: 2026-05-19
Environment: R6000 (MI300X / SM120), torch 2.10.0+cu128

## Summary

| Metric | Value |
|---|---:|
| Layers repacked | 32/32 |
| Sidecar size | 6048.04 MiB |
| Contract overall | GREEN |
| Contract checked | L00, L08, L16, L31 |
| Contract elapsed | 1.34 s |

## Source Model Integrity

| File | SHA256 |
|---|---|
| `config.json` | `1bfafec061b61e06aade6d1b6bd810b9df90580393fd2444ccf1835de8ac78a6` |
| `model.safetensors.index.json` | `2aa537f7466b7a5fee751d3e5f08d6e8652b47d54c3daf2776a3fba61e3fb972` |
| `lynn_quant_manifest.json` | `a4e3201b379919e4501a40c4eed8262ede5cdcb4881447b0f0e39d00c6a02fda` |

## Sidecar Layout

Per layer (`layer_{NN}.safetensors`) contains 15 tensors:

| Tensor | Shape (gate_proj example) | Dtype | Note |
|---|---|---|---|
| `gate_proj.weight_packed` | [12288, 2048] | uint8 | Original NVFP4 packed bytes [N, K/2] |
| `gate_proj.weight_t_packed` | [2048, 12288] | uint8 | Pretransposed contiguous [K/2, N] |
| `gate_proj.weight_scale` | [12288, 256] | float32 | Per-16 group scale [N, K/16] |
| `gate_proj.weight_global_scale` | scalar | float32 | Global scale |
| `gate_proj.scale_b_native` | [3145728] | uint8 | Swizzled FP8 scale for `torch._scaled_mm` (view as float8_e4m3fn) |

Same 5 tensors repeated for `up_proj` and `down_proj`.

## Contract Results

All checked layers passed bitwise verification:

- `weight_packed` identical to original
- `weight_t_packed` matches reference transpose and is contiguous
- `weight_scale` / `weight_global_scale` identical to original
- `scale_b_native` matches reference swizzle

## How P191 Reads the Sidecar

```python
from safetensors.torch import load_file

side = load_file("sidecar/layer_00.safetensors", device="cuda")

# Original packed layout (for scalar-bridge / Triton kernels)
w_packed = side["gate_proj.weight_packed"]          # [N, K/2] uint8
w_scale = side["gate_proj.weight_scale"]            # [N, K/16] float32
w_global = side["gate_proj.weight_global_scale"]    # scalar float32

# Pretransposed layout (for native _scaled_mm / CuTe kernels)
w_t = side["gate_proj.weight_t_packed"].view(torch.float4_e2m1fn_x2)  # [K/2, N]
scale_b = side["gate_proj.scale_b_native"].view(torch.float8_e4m3fn)   # [rows*groups]

# Example: FP4xFP8 matmul via torch._scaled_mm
out = torch._scaled_mm(
    act_fp8,           # [M, K] float8_e4m3fn
    w_t,               # [K/2, N] float4_e2m1fn_x2
    scale_a=scale_a,
    scale_b=scale_b,
    out_dtype=torch.bfloat16,
)
```

## Artifacts

- Sidecar: `/root/autodl-tmp/reports/qwen35_9b/p192_dense_fp4x_fp8_sidecar/`
- Manifest: `/root/autodl-tmp/reports/qwen35_9b/p192_dense_fp4x_fp8_sidecar/repack_manifest.json`
- Contract report: `/root/autodl-tmp/reports/qwen35_9b/p192b_dense_fp4x_fp8_repack_contract.json`
