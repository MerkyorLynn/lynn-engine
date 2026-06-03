Both compile. The code and wiring are in place. Here's the concise deliverable.

## What landed

**`triton_kernels/full_attn_fused.py`** (new) — two single-launch kernels.
**`engine/incremental_decode.py::decode_full_attn`** — wired behind `LYNN_FULL_ATTN_FUSED=1`, coexisting with the current path (3 surgical edits, default unchanged).

## (1) The two fusions

**A. `qk_norm_rope_cache_write_triton` — fuse RoPE + K/V cache-write (3 launches → 1).**
Extends the existing `_qk_norm_rope_pair_kernel` (`triton_kernels/qk_norm_rope.py:64`) with a third head-class. Grid = `H_Q + 2·H_KV`:
- `program < H_Q` → Q head: norm(+1)+partial-RoPE → returned `q_out` (identical math to the pair kernel).
- `H_Q ≤ program < H_Q+H_KV` → K head: norm+RoPE → stored **directly into `K_cache[0, h, write_pos, :]`**.
- `program ≥ H_Q+H_KV` → V head: **raw copy** (no norm/rope) → `V_cache[0, h, write_pos, :]`.

`write_pos` is read from device memory (`tl.load(write_pos_ptr)`) so nothing shape/index-wise is baked at capture time → **CUDA-graph-replayable**, same as the `index_copy_` path it replaces. Per-head strides are passed explicitly (`q.stride(1)=2·head_dim` because `q` is a non-contiguous chunk-then-transpose view) — this also kills the implicit `.contiguous()` copy that `qk_norm_rope_pair_triton`'s `reshape` would have incurred.

**B. `gate_sigmoid_fold_triton` — fold sigmoid-gate + transpose/contiguous (~5 launches → 1).**
Was: `.float()` + `sigmoid` + `.to(dtype)` + `*` + `.transpose(1,2).contiguous()`. One program/Q-head writes the `[1,1,H_Q·D]` o_proj input row directly (the `h·HEAD_DIM` store base *is* the transpose). The o_proj GEMV stays a cuBLAS/packed-NVFP4 call (`_full_attn_o_proj`) — not worth reimplementing at decode shapes.

**Net: ~6 launches saved per full-attn layer × 10 layers/token ≈ 60 fewer launches/token.**

## (2) Wiring (exact, all gated)

- `incremental_decode.py:429` — `fused` flag (`LYNN_FULL_ATTN_FUSED=1 and q.is_cuda and bf16 caches`).
- RoPE block (`~437`): `if fused:` calls the kernel and reassigns `q`; `else:` keeps `_qk_norm_rope_pair_decode`.
- cache-write block (`~466`): `if fused: pass` (kernel already wrote); `elif fixed_shape / else` unchanged.
- gate/o_proj block (`~507`): `if fused and bf16:` calls `gate_sigmoid_fold_triton` → `_full_attn_o_proj`; else the original two-step path.

The fixed-shape masked-SDPA windowing and the variable-slice path both read the cache *after* the write, so both work unchanged with the fused write.

## (3) Token-exactness risks & keeping cos~1

- **V copy: bit-exact.** bf16→fp32→bf16 of an already-bf16 value is identity; matches `index_copy_`.
- **Gate epilogue: bit-exact by construction.** The kernel rounds `sigmoid(gate_fp32)` to bf16 *before* the multiply (`.to(tl.bfloat16).to(tl.float32)`), reproducing `attn_out * sigmoid(gate.float()).to(dtype)` (bf16×bf16, fp32 accumulate, bf16 round).
- **Q/K norm+RoPE: matches the Triton pair path, not eager torch.** The kernel's fp32 `sum(x·x)/HEAD_DIM` vs torch's `pow(2).mean()` differ by ≲1 bf16 ULP after the bf16 store — same delta the existing `LYNN_QK_NORM_ROPE_BACKEND=triton_pair` path already carries. This is the only non-bit-exact-vs-default surface; it does not flip greedy argmax, but should be validated on the existing parity gate before serving-default.
- **Position correctness:** `write_pos = pos_tensor`. Verified equal to the written row for every caller — T=1 decode (`new_position_id == state.seq_len == cached_seq_len`) and the `t1_loop` K2/block verifiers (`position_ids[idx] == state.seq_len+idx == cached_seq_len`). If a future caller passes `new_position_id ≠ cached_seq_len` in variable-slice mode, the fused write would target the wrong row — documented in the code comment.
- **Hard guards / fallbacks:** non-bf16 caches or non-CUDA `q` fall back to the original path; the kernel raises on non-bf16 cache rather than silently mis-storing.

**Suggested A/B:** `LYNN_FULL_ATTN_FUSED=1` (optionally with `LYNN_FULL_ATTN_FIXED_SHAPE=1` for graph capture) vs default, on the greedy-parity + tok/s harness — same lever class as the RMSNorm win. Nothing was run.
