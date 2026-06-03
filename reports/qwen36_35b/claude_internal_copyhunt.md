# Stage-4A copy-hunt — NVFP4 decode `aten::copy_` elision (claude-internal)

**Status:** edits in working tree only. NOT committed. Lead reviews + GPU-verifies + RC-validates.
**Files touched:** `engine/nvfp4_runtime.py`, `engine/incremental_decode.py`.
**Default behavior:** with all new flags unset, behavior is bit-for-bit identical to today.

---

## TASK 1 — bf16-out for the NVFP4 decode matmul (gated, fallback-safe)

### New flag
`LYNN_NVFP4_BF16_OUT` (default `"0"`).

### Root cause recap
Both `forward_native_fast_2d` methods historically did:
```python
torch._scaled_mm(..., out_dtype=torch.float16)[0].float()   # -> fp32
```
The decode caller `_linear` (`engine/incremental_decode.py:88`) then does `out.to(x.dtype)`
with `x` in **bf16**. So per projection there is a real **fp32 → bf16 `aten::copy_`** (one
launch + one memory pass), on the launch-bound Spark GB10 decode path. (Note: the original
code `.float()`s to fp32 first, so the live copy is fp32→bf16, not fp16→bf16 as the census
shorthand put it — the `.float()` is itself an extra fp16→fp32 copy that this change also
removes when bf16-out is active.)

### What changed
- Added module-level helper `_native_decode_scaled_mm(act_view, weight_t, scale_a, scale_b)`
  in `engine/nvfp4_runtime.py`. It returns `(out_2d, is_bf16)`.
  - `LYNN_NVFP4_BF16_OUT=1` → calls `torch._scaled_mm(..., out_dtype=torch.bfloat16)` and
    returns the **bf16** tensor with `is_bf16=True`.
  - otherwise → calls `torch._scaled_mm(..., out_dtype=torch.float16)` and returns `is_bf16=False`.
- Both `forward_native_fast_2d` methods now call the helper and do:
  ```python
  out, is_bf16 = _native_decode_scaled_mm(...)
  return out if is_bf16 else out.float()
  ```
  - **Default (flag unset):** `is_bf16=False` → `out.float()` → fp32 → identical to today.
  - **Flag on:** returns bf16 directly (no `.float()`), so `_linear`'s existing
    `out.to(x.dtype)` is a **no-op** (bf16→bf16) → the per-projection copy disappears.
    (The `.float()` is intentionally skipped in this mode — keeping it would force fp32
    and make `out.to(bf16)` a real copy again, defeating the optimization.)
- `_linear` was **NOT** edited — its `out.to(x.dtype)` is already a no-op when dtypes match.
- `PackedNVFP4Linear.forward`'s separate inline `native_scaled_mm` backend (≈ line 341, the
  non-fast `forward` path, not used by the decode fast path) was left unchanged to stay
  surgical and in-scope.

### Robustness / fallback
- The bf16 `_scaled_mm` attempt is wrapped in `try/except`. On **any** exception (e.g. the
  block-scaled ue4m3xe2m1 NVFP4 path rejecting `out_dtype=bfloat16` on this sm_121 build),
  `_disable_bf16_out()` sets a process-global flag so **all subsequent calls go straight to
  the fp16 path** (no repeated failing `_scaled_mm` launches) and emits a **one-time** stderr
  warning. Never crashes — always falls back to today's exact fp16 behavior.

### Numerics (for the lead's RC pass — not a blocker)
fp32-accumulate → bf16 is **one fewer rounding** than fp32 → fp16 → bf16, so this is
quality-neutral-or-better. Token-exactness is **not** expected and is fine.

---

## TASK 2 — other cheaply-removable copies (conservative scan)

Scope per task: only the **M=1 single-token decode hot path** — `decode_linear_attn` and the
single-token full-attn decode in `decode_full_attn`. K2 / multi-position branches and the
prefill paths (`prefill_full_attn`, e.g. the `attn_out.transpose(1,2).contiguous().view(...)`
at ~line 386 and the `K/V_to_cache.contiguous()` at ~368–369) were **NOT** touched.

### Removed (gated) — `decode_full_attn` o_proj reshape (≈ line 559)
`engine/incremental_decode.py`, non-fused single-token o_proj epilogue:
```python
attn_out = attn_out.transpose(1, 2).contiguous().view(B, 1, H_Q * head_dim)
```
For **T=1** decode, `transpose(1,2)` swaps the head dim with a **size-1** dim. For a
contiguous `attn_out` of shape `[B, H_Q, 1, head_dim]`, the transposed view `[B, 1, H_Q,
head_dim]` is **already contiguous** (PyTorch's `is_contiguous` skips size-1 dims), so the
explicit `.contiguous()` is a redundant copy.

**New flag** `LYNN_DECODE_OPROJ_NOCOPY` (default `"0"`):
- `=1` → `attn_flat = attn_out.transpose(1, 2).reshape(B, 1, H_Q * head_dim)`.
  `.reshape()` is **value-identical** to `.contiguous().view()` (both preserve logical element
  order) and **never crashes** (returns a view when contiguous, copies only if genuinely
  non-contiguous). So it elides the copy in the common case yet stays safe across all SDPA /
  manual_gqa output layouts.
- unset → original `.contiguous().view(...)` (exact current behavior).

**Why gated rather than removed outright:** a bare deletion of `.contiguous()` before `.view()`
would *crash* if the SDPA/gate-multiply output ever lands non-contiguous (layout isn't
contractually guaranteed across `_scaled_dot_product_attention` backends). `.reshape()` removes
that risk entirely, and gating keeps default behavior bit-identical for the lead's A/B + RC.

### Left (listed, not changed)
- **`decode_full_attn` gate epilogue** (≈ line 555):
  `attn_out * torch.sigmoid(gate.float()).to(attn_out.dtype)`. The `gate.float()` /
  `.to(attn_out.dtype)` is a real fp32 sigmoid round-trip for precision, **not** redundant.
  Left as-is.
- **`decode_full_attn` fused path** (≈ lines 548–552): already single-launch via
  `gate_sigmoid_fold_triton` (no `contiguous().view()`); nothing to remove.
- **`decode_linear_attn`** (≈ lines 1137–1276): no redundant `.contiguous()` or
  redundant `.to(dtype)` found. The reshapes there (`mixed_new.transpose(1,2)`, the q/k/v
  `reshape`, the `core_attn_out.reshape(-1, HEAD_V_DIM)` / `flat_y.reshape(...)` feeding
  `_rms_norm_gated_decode`) are genuine layout changes required by the recurrent / RMSNormGated
  kernels, and `reshape` already avoids a copy when it can. Nothing safely removable.

---

## New env flags introduced (both default OFF = today's behavior)
| Flag | File | Effect when `=1` |
|------|------|------------------|
| `LYNN_NVFP4_BF16_OUT` | `engine/nvfp4_runtime.py` | NVFP4 decode matmul returns bf16 directly; removes per-projection fp32→bf16 copy. Auto-falls back to fp16 on bf16-out rejection. |
| `LYNN_DECODE_OPROJ_NOCOPY` | `engine/incremental_decode.py` | full-attn T=1 o_proj reshape uses `.reshape()` instead of forced `.contiguous().view()`; elides redundant copy. |

## Verification done here
```
python3 -c "import ast; ast.parse(open('engine/nvfp4_runtime.py').read()); \
            ast.parse(open('engine/incremental_decode.py').read()); print('PARSE_OK')"
# -> PARSE_OK
```
No GPU run performed (no commit). Lead to GPU-verify + RC-validate.
