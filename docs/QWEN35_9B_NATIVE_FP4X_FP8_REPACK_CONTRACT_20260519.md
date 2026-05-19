# Qwen3.5-9B Native FP4xFP8 Dense Repack Contract

Date: 2026-05-19

## Purpose

P192 is the offline repack step that converts the raw NVFP4 checkpoint into
kernel-ready sidecar layouts for dense FFN gate/up/down.  It does not change
model math; it only precomputes transpose and scale swizzles so that P191 (and
later resident kernels) can load one file per layer instead of chasing generic
manifest keys.

## Input

- Model: `/root/autodl-tmp/models/Qwen3.5-9B-lynn-native-w4a16-nvfp4-v0`
- Format: compressed-tensors NVFP4 v8-RTN with `model.safetensors.index.json`

## Output Sidecar

Directory: `reports/qwen35_9b/p192_dense_fp4x_fp8_sidecar/`

Per layer (`layer_{NN}.safetensors`):

| Tensor | Shape | Dtype | Meaning |
|---|---|---|---|
| `gate_proj.weight_packed` | `[N, K/2]` | uint8 | Original NVFP4 packed bytes |
| `gate_proj.weight_t_packed` | `[K/2, N]` | uint8 | Pretransposed contiguous E2M1 layout |
| `gate_proj.weight_scale` | `[N, K/16]` | float32 | Per-group scale |
| `gate_proj.weight_global_scale` | scalar | float32 | Global scale |
| `gate_proj.scale_b_native` | `[rows*groups]` | uint8 | Swizzled FP8 scale for `torch._scaled_mm`; view as `float8_e4m3fn` |
| ... same for `up_proj` and `down_proj` |

Manifest: `repack_manifest.json`

```json
{
  "schema_version": "qwen35-9b-dense-fp4x-fp8-repack-v1",
  "source_model": "...",
  "layers": {
    "0": {
      "file": "layer_00.safetensors",
      "sha256": "...",
      "tensors": {
        "gate_proj.weight_packed": {"shape": [12288, 2048], "stride": [2048, 1], ...},
        ...
      }
    }
  }
}
```

## Exact Layout Spec

### Pretransposed E2M1 (`weight_t_packed`)

Given original `weight_packed` of shape `[N, K/2]` uint8:

```python
weight_t_packed = weight_packed.t().contiguous()
```

Result is `[K/2, N]` uint8, contiguous, ready to be passed as the B operand to
`torch._scaled_mm` or a CuTe kernel without runtime transpose.
Each byte is already one packed E2M1 x2 element, so the byte transpose is the
same element-level transpose a `float4_e2m1fn_x2` view would express.  PyTorch
currently cannot CPU-copy that float4 dtype, so the sidecar contract uses the
byte-level form.

### Native Swizzled Scale (`scale_b_native`)

Given original `weight_scale` of shape `[N, K/16]` float32:

```python
from engine.nvfp4_runtime import _compact_scale_to_swizzled_fp8
scale_b_native = _compact_scale_to_swizzled_fp8(
    weight_scale.float(), outer_dim=N, k=K
).view(torch.uint8)
```

Result is a 1-D uint8 tensor (view of `float8_e4m3fn`) in the tiled layout
expected by cuBLASLt for FP4 `scale_b`.

## Contract Validation (P192-B)

P192-B loads the original model and the sidecar, then verifies for each
projection:

1. `weight_packed` is bitwise identical to the original.
2. `weight_t_packed` matches the reference transpose.
3. `weight_scale` and `weight_global_scale` are identical.
4. `scale_b_native` matches the reference swizzle.

## Integration with P191

P191 can consume the sidecar by loading `layer_{NN}.safetensors` directly:

```python
from safetensors.torch import load_file
side = load_file("sidecar/layer_00.safetensors", device="cuda")
w_packed = side["gate_proj.weight_packed"]
w_t = side["gate_proj.weight_t_packed"].view(torch.float4_e2m1fn_x2)
scale = side["gate_proj.weight_scale"]
scale_b = side["gate_proj.scale_b_native"].view(torch.float8_e4m3fn)
```

This removes the need for P191 to open the original model shards or compute
transpose/swizzle at load time.

## Artifacts

- `benchmarks/p192_qwen35_9b_dense_fp4x_fp8_repack.py`
- `benchmarks/p192b_qwen35_9b_dense_fp4x_fp8_repack_contract.py`
- `scripts/r6000_qwen35_9b_dense_fp4x_fp8_repack.sh`
- `docs/QWEN35_9B_NATIVE_FP4X_FP8_REPACK_CONTRACT_20260519.md`
