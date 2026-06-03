**Conclusion: 4a, with a config caveat.** The large routed MoE decode path is already reading packed NVFP4 and dequantizing inside kernels; I do not see a BF16 weight temp in HBM. The remaining steady-state BF16 reads are intentional non-shadow paths: router, shared expert, small constants, and under the checked Stage-6 BASE_ENV, full-attn projections plus linear-attn `out_proj` because `LYNN_PACKED_DECODE=0`. That makes the highest-ROI path launch/dispatch and graph/fusion work, after first ensuring existing packed projection aliases are actually enabled where desired. No new Triton kernel is included because this is not 4b/4c.

**1. Decode Byte Path**

Trace:

- `LynnIncrementalRunner._decode_layer_fast` only hoists dispatch knobs, then calls `_decode_layer` with fixed `moe_fn` / recurrent backend: `engine/resident_runner.py:1256`.
- `_decode_layer` runs input norm, either `decode_linear_attn` or `decode_full_attn`, then post-attn norm and MoE: `engine/full_forward.py:982`.
- Packed alias selection is controlled by `_decode_weight`; it only returns `key + ".packed"` when `LYNN_PACKED_DECODE=1`, or the per-family packed flags are set: `engine/incremental_decode.py:145`.
- The Stage-6 audit script sets `LYNN_PACKED_DECODE=0` while setting only `LYNN_PACKED_DECODE_BACKEND=native_fast_2d`: `scripts/spark_stage6_shadow_byte_audit.py:26`. Backend alone does not attach/use aliases.
- Linear-attn fused native FP4 in-proj is attached separately by `LYNN_LINEAR_ATTN_INPROJ_FUSED_NATIVE_FP4=1`: `engine/resident_runner.py:824`.
- Active MoE decode requires packed aliases and rejects missing ones: `engine/moe_packed_nvfp4.py:965`.
- Routed MoE gate/up and down read `_gate_up_packed/_down_packed` plus scales: `engine/moe_packed_nvfp4.py:787`, `engine/moe_packed_nvfp4.py:854`.
- Shared expert remains BF16 unless packed-shared is explicitly enabled; current fixed path reads fused BF16 shared gate/up and down: `engine/moe_packed_nvfp4.py:889`.
- Native FP4 lm_head repacks once at startup and uses `_scaled_mm` over packed FP4 at decode: `engine/resident_runner.py:923`, `engine/resident_runner.py:955`.

Approximate single-token weight bytes, assuming Qwen3.6 dimensions from code/docs: hidden 2048, 40 layers, 30 linear-attn, 10 full-attn, 256 experts, top-8, expert/shared inter 512, vocab 248320.

| Decode weight group | Current Stage-6 BASE_ENV FP4+scale | Current BF16 read | If projection packed aliases are enabled |
|---|---:|---:|---:|
| Routed MoE active experts, top-8 all 40 layers | ~755 MB | 0 | same |
| Linear-attn fused qkv/z/b/a, 30 layers | ~427 MB | 0 | same |
| Linear-attn `out_proj`, 30 layers | 0 | ~503 MB | ~142 MB FP4+scale, 0 BF16 |
| Full-attn q/k/v/o, 10 layers | 0 | ~545 MB | ~153 MB FP4+scale, 0 BF16 |
| Native FP4 lm_head | ~286 MB | 0 read | same |
| MoE router `mlp.gate.weight`, 40 layers | 0 | ~42 MB | same unless separately quantized |
| Shared expert + shared gate, 40 layers | 0 | ~252 MB | can use existing packed-shared path, but script warns it was slower |
| Conv/norm/small constants | 0 | ~2-3 MB | same |
| **Total** | **~1.47 GB** | **~1.34 GB** | **~1.76 GB FP4+scale, ~0.30 GB BF16** |

So: **yes, BF16 weights are still read in steady-state decode**, but the large 60 GiB dequant shadow is not the active routed-MoE decode source. The largest “current BASE_ENV” BF16 reads are projection aliases not being enabled, not a missing MoE FP4 kernel.

**2. FP4 Kernel Efficiency**

`nvfp4_grouped_gate_up_silu_fast_decode` launches `_grouped_gate_up_silu_fast_decode_kernel`: `triton_kernels/nvfp4_moe.py:760`.

Inside the kernel:

- Loads packed uint8 from `gate_up_packed_ptr`: `triton_kernels/nvfp4_moe.py:316`.
- Extracts nibbles and decodes E2M1 in registers: `triton_kernels/nvfp4_moe.py:326`.
- Loads per-16 scale: `triton_kernels/nvfp4_moe.py:330`.
- Accumulates in FP32 registers: `triton_kernels/nvfp4_moe.py:340`.
- Stores only BF16 intermediate activation, not BF16 weight: `triton_kernels/nvfp4_moe.py:347`.

`nvfp4_grouped_down_weighted_sum` launches `_grouped_down_weighted_sum_kernel`: `triton_kernels/nvfp4_moe.py:936`.

Inside the kernel:

- Loads BF16 inter activation: `triton_kernels/nvfp4_moe.py:394`.
- Loads packed down weight bytes: `triton_kernels/nvfp4_moe.py:409`.
- Decodes nibble to E2M1: `triton_kernels/nvfp4_moe.py:414`.
- Loads scale and accumulates in FP32: `triton_kernels/nvfp4_moe.py:416`.
- Stores BF16 output vector only: `triton_kernels/nvfp4_moe.py:427`.

No full BF16 MoE weight temp is materialized by these Triton kernels.

For dense packed projections, `forward_native_fast_2d` quantizes activation, passes packed FP4 weight view and scale into `torch._scaled_mm`: `engine/nvfp4_runtime.py:348`. The repo does not materialize a BF16 weight temp. Any temp would be internal to PyTorch/CUTLASS, not visible in this code. The local path directly calls `_scaled_mm` with `torch.float4_e2m1fn_x2`: `engine/nvfp4_runtime.py:64`.

**3. Prefill BF16**

Prefill is BF16-shadow dependent today.

- `_prefill_layer` calls `prefill_full_attn` / `prefill_linear_attn`, then `_ffn_forward`: `engine/full_forward.py:631`.
- `prefill_full_attn` uses BF16 `F.linear` for q/k/v/o: `engine/incremental_decode.py:339`.
- `prefill_linear_attn` uses BF16 `F.linear` for qkv/z/b/a/out: `engine/incremental_decode.py:1007`.
- MoE prefill routes through `_moe_forward` / `moe_forward_prefill_optimized`, which uses BF16 expert stacks or per-expert BF16 keys: `engine/full_forward.py:323`, `engine/moe_optimized.py:244`.
- Loader slow-dequants Lynn NVFP4 packed tensors into BF16 `.weight` tensors: `engine/loader.py:348`.
- Loader also stacks expert BF16 gate/up and down tensors: `engine/loader.py:476`.
- Packed MoE decode explicitly rejects non-T=1: `engine/moe_packed_nvfp4.py:972`.
- Packed linear wrappers also reject multi-token inputs: `engine/nvfp4_runtime.py:388`.
- `release_decode_bf16_shadows()` deletes only BF16 tensors covered by packed decode aliases, and warns not to use before prefill unless packed prefill exists: `engine/resident_runner.py:1347`.

This confirms the post-release `KeyError('mlp.experts.1.gate_proj.weight')` is consistent with a fresh prefill trying to read BF16 expert weights, not steady-state decode.

**4. Ranked Lever**

1. **Config correctness first:** `LYNN_PACKED_DECODE_BACKEND=native_fast_2d` alone is insufficient. If the intended RC is “all projection decode packed,” set/verify `LYNN_PACKED_DECODE_LINEAR_ATTN=1` and `LYNN_PACKED_DECODE_FULL_ATTN=1` or `LYNN_PACKED_DECODE=1`, then run a decode-only post-prefill release test. Current Stage-6 script sets `LYNN_PACKED_DECODE=0`: `scripts/spark_stage6_shadow_byte_audit.py:32`.

2. **Highest ROI remains dispatch/graph/fusion, not a new read-4bit MoE kernel.** Existing evidence says traffic levers were dead or negative and the decode census is ~1527 launches/token, with ~half time launch/dispatch: `reports/qwen36_35b/DECODE_LAUNCH_OVERHEAD_CAMPAIGN_20260603.md:118`. README also records launch-bound decode and ~40% CPU dispatch: `README_EN.md:3`.

3. **Remaining BF16 elimination is secondary.** After existing packed projection aliases, BF16 weight reads are mostly router + shared expert, about ~0.30 GB/token. At 240 GB/s, perfect removal is only ~1.2 ms/token. It cannot explain 45→70 by itself.

4. **Do not write 4b kernel now.** The routed MoE Triton kernels already read packed 4-bit and dequantize in-kernel. There is no repo-visible BF16 weight temp to remove.
tokens used
