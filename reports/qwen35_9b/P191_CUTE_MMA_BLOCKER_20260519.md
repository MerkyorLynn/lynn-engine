# P191 CuTe MMA Fragment Layout Blocker

Date: 2026-05-19
Branch: `claude/p191-r6000-fp4x-fp8-cute-dense-poc`

## Status: AMBER (MMA compiles + runs at speed, fragment layout incorrect)

### What Works

| Item | Status |
|------|--------|
| SM120a compilation | GREEN |
| `cute::SM120_16x8x32_TN<float_e4m3_t, float_e2m1_t, float>::fma` | COMPILES + EXECUTES |
| No illegal instruction / crash | GREEN |
| MMA latency | **0.049ms** (12288 outputs, K=4096) |
| Scalar reference correctness | GREEN (cosine 0.9999 vs BF16 ref) |

### What Doesn't Work

The MMA produces garbage output (cosine -0.015 vs scalar reference). Root cause:
the A-register (E4M3) and B-register (E2M1) fragment packing layout is wrong.

**The instruction expects specific lane-to-element mappings** that are not
documented publicly for the non-blockscaled `SM120_16x8x32_TN` variant.
My naive "linear" packing (thread t holds elements [t*16..t*16+15]) is incorrect.

### Fragment Layout Challenge

For `mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e4m3.e2m1.f32`:

- **A operand** (E4M3, M16×K32, row-major):
  - 4 uint32_t per thread = 16 bytes = 16 E4M3 elements
  - 32 threads × 16 = 512 elements = 16×32 ✓
  - Lane mapping: UNKNOWN (not public for non-blockscaled)

- **B operand** (E2M1, K32×N8, col-major):
  - 2 uint32_t per thread = 8 bytes = 16 E2M1 nibbles
  - 32 threads × 16 = 512 nibbles... but K32×N8 = 256 nibbles needed
  - The extra bits may be padding or the layout packs differently

### Known Approaches to Discover Layout

1. **Brute-force sweep**: Send identity-like patterns through MMA, observe where outputs land
2. **PTX documentation**: NVIDIA PTX ISA 8.x may specify matrix fragment formats
3. **Look at P98 blockscaled variant**: `SM120_16x8x32_TN_VS` has working fill functions
   that demonstrate the warp mapping. The non-blockscaled variant likely uses the same
   spatial layout but different scale handling.
4. **CuTe layout atoms**: `cute/atom/mma_atom.hpp` may have layout definitions

### Next Steps

1. **Adapt P98's fill_a/fill_b pattern** to the non-blockscaled variant:
   - P98 uses `SM120::BLOCKSCALED::SM120_16x8x32_TN_VS` with scale_byte
   - The spatial layout (which lane holds which (m,n,k) element) should be SAME
   - Just strip the scale_byte handling

2. **Validate with a known test vector**: Set A=identity(row), B=ones → expect column sums

### Performance Implication

If fragment layout is resolved, the E4M3×E2M1 path gives:
- 0.049ms for 12288-output gate_proj (vs Triton BF16 at ~0.8ms est)
- This would be **16x faster than scalar reference** and potentially
  competitive with Triton's BF16 dequant path
- The instruction handles both FP8 activation AND FP4 weight in one shot

### Conclusion

**AMBER**: The hardware path exists, compiles, runs at speed. Blocked only on
fragment register layout. This is an engineering problem, not a hardware limitation.
The P98 blockscaled code provides the pattern to follow.
