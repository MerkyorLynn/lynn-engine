# P138/P139 Slot-Packed NVFP4 Kernel ABI

## Status

Draft — for future native TensorCore kernel authors.

## Scope

This document specifies the **slot-packed NVFP4 weight layout** produced by `p138_pack_moe_fixture_slots_nvfp4.py` and validated by `p139_moe_slot_packed_contract.py`. It is **not** the production serving path; it is a fixture-level packed format for kernel R&D.

## High-Level Idea

Instead of expanding top-8 expert weights to BF16 (`[8,1024,2048]` + `[8,2048,512]` ≈ 49 MB), we keep the native NVFP4 packed representation (`[8,1024,1024]` uint8 + scales ≈ 15.6 MB). The kernel receives pre-gathered slot-packed weights and skips dynamic expert-table indexing entirely.

## Per-Fixture Tensor Layout

Saved as safetensors (optionally `.safetensors.gz`):

| Key | Shape | Dtype | Bytes | Purpose |
|-----|-------|-------|-------|---------|
| `hidden_in` | `[1, 2048]` | BF16 | 4,096 | Input hidden state |
| `expert_ids` | `[8]` | int32 | 32 | Top-8 expert IDs (metadata) |
| `routing_weights` | `[8]` | float32 | 32 | Top-8 routing weights |
| `slot_gate_up_packed` | `[8, 1024, 1024]` | uint8 | 8,388,608 | Gate+Up fused, packed FP4 |
| `slot_gate_up_scale` | `[8, 1024, 128]` | FP16 | 2,097,152 | Per-16-column scale |
| `slot_gate_up_global_scale` | scalar | FP16 | 2 | Global scale divider |
| `slot_down_packed` | `[8, 2048, 256]` | uint8 | 4,194,304 | Down proj, packed FP4 |
| `slot_down_scale` | `[8, 2048, 32]` | FP16 | 1,048,576 | Per-16-column scale |
| `slot_down_global_scale` | scalar | FP16 | 2 | Global scale divider |

**Total per fixture:** ~15.6 MB (uncompressed) vs ~49 MB BF16 slot weights.

### Compression (`--compress`)

p138 supports `.safetensors.gz` output (~12–14 MB, ~72% reduction vs BF16).

**Rule:** Gzip is **only for offline fixture distribution and storage**. Do **not** decompress on the hot inference path. A production kernel should consume either:
- **Uncompressed safetensors** (fast `mmap` or direct load), or
- **A sidecar memory-mapped buffer** where packed weights are kept in GPU/CPU memory without per-token I/O.

Decompressing gzip during decode adds unnecessary CPU overhead and latency. Pre-decompress offline, or keep fixtures uncompressed on fast storage.

## NVFP4 E2M1 Encoding

Each byte stores two FP4-E2M1 values:

```
byte = [high_nibble << 4] | low_nibble
```

| Bits | Meaning |
|------|---------|
| `b3` | Sign (1 = negative) |
| `b2:b0` | Magnitude index into `[0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]` |

Dequant formula for one value:
```python
value = sign * table[magnitude]
sign = -1.0 if (nibble & 0x8) else 1.0
magnitude = nibble & 0x7
table = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
```

## Per-16 Scale Layout

Group size = 16 input columns. One `FP16` scale per group.

For `slot_gate_up_packed [8, 1024, 1024]`:
- Last dim = 1024 bytes = 2048 FP4 values
- Scale last dim = 2048 / 16 = 128
- Scale shape = `[8, 1024, 128]`

For `slot_down_packed [8, 2048, 256]`:
- Last dim = 256 bytes = 512 FP4 values
- Scale last dim = 512 / 16 = 32
- Scale shape = `[8, 2048, 32]`

## Full Dequant Pipeline

```python
def dequant(packed, scale, global_scale, out_features, in_features):
    # packed: [K, out_features, in_features // 2]  uint8
    # scale:  [K, out_features, in_features // 16] fp16
    # global_scale: scalar fp16

    # 1. Unpack uint8 -> FP4 values (float32)
    unpacked = unpack_fp4_e2m1(packed)  # [K, out_features, in_features]

    # 2. Broadcast per-16 scale
    scale_repeat = scale.repeat_interleave(16, dim=-1)  # [K, out_features, in_features]

    # 3. Apply global scale
    effective_scale = scale_repeat / global_scale

    # 4. Multiply
    return (unpacked * effective_scale).to(bf16)
```

The `global_scale` direction follows the compressed-tensors convention: `effective = scale / global_scale`. All p139 contract tests verify that this pipeline reproduces the BF16 reference exactly (`max_abs == 0`).

## Proposed CUDA Kernel ABI

### Host-side Python binding

```cpp
torch::Tensor moe_slot_packed_nvfp4_decode(
    torch::Tensor x,                    // [2048] BF16
    torch::Tensor routing_weights,      // [8] float32
    torch::Tensor slot_gup_packed,      // [8, 1024, 1024] uint8
    torch::Tensor slot_gup_scale,       // [8, 1024, 128] fp16
    torch::Tensor slot_gup_gs,          // scalar fp16
    torch::Tensor slot_dp_packed,       // [8, 2048, 256] uint8
    torch::Tensor slot_dp_scale,        // [8, 2048, 32] fp16
    torch::Tensor slot_dp_gs            // scalar fp16
);
```

### Expected math

```python
out = zeros([2048], bf16)
for slot in range(8):
    # Stage 1: gate_up_silu (packed -> BF16 intermediate)
    gu = dequant(slot_gup_packed[slot], slot_gup_scale[slot], slot_gup_gs,
                 out_features=1024, in_features=2048)
    gate, up = gu[:512], gu[512:]
    inter = silu(gate) * up   # [512] BF16

    # Stage 2: down (packed -> BF16 output)
    down = dequant(slot_dp_packed[slot], slot_dp_scale[slot], slot_dp_gs,
                   out_features=2048, in_features=512)
    out += matvec(down, inter) * routing_weights[slot]
```

### Strides

All packed tensors are saved `.contiguous()` with row-major layout:
- `slot_gup_packed.stride() = (1024*1024, 1024, 1)`
- `slot_gup_scale.stride() = (1024*128, 128, 1)`
- `slot_dp_packed.stride() = (2048*256, 256, 1)`
- `slot_dp_scale.stride() = (2048*32, 32, 1)`

Kernel may assume contiguous storage.

## Memory Bandwidth Comparison

| Path | Read per fixture | Size |
|------|-----------------|------|
| BF16 slot (`p135`) | gate_up + down | ~49 MB |
| Packed slot (`p138`) | packed + scales | ~15.6 MB |
| **Reduction** | | **68%** |

For a native kernel that fuses unpack + GEMM, effective bandwidth is:
- Read `x` once: 4 KB
- Read packed weights + scales: ~15.6 MB (amortized across 8 experts)
- Write intermediate `[8, 512]`: 8 KB
- Write output `[2048]`: 4 KB

Dominant traffic is the weight read. Packed path saves **~33 MB per token** vs BF16.

## Future Work

1. **TensorCore FP4 GEMM**: On SM100+ (Blackwell), use `mma.sync.aligned.m64n8k32.row.col.f16.f4e2m1.f4e2m1.f16` or equivalent PTX. The packed uint8 layout is directly consumable by hardware.
2. **FP4 act quant**: If activation `x` is also quantized to FP4, the gate/up matvec becomes a pure FP4×FP4 MMA, further reducing memory traffic.
3. **Shared-expert packed slot**: Extend the same packed-slot idea to the shared expert (1 expert, not 8).

## References

- `benchmarks/p138_pack_moe_fixture_slots_nvfp4.py` — fixture exporter
- `benchmarks/p139_moe_slot_packed_contract.py` — decode contract
- `engine/dequant.py` — `unpack_fp4_e2m1_from_uint8` reference
- `engine/nvfp4_runtime.py` — `load_grouped_nvfp4_weight` loader
