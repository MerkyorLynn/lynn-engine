# Stage-3 · linear-attn g/beta-fold into the recurrent kernel (claude-internal, gated, unverified)

**Status:** working-tree only, NOT committed. Default OFF. Needs GPU verification by lead.

## Goal
Fold the per-layer g/beta elementwise computation INTO the existing gated-delta
recurrent decode kernel, eliminating ~4 tiny CUDA launches/layer (sigmoid, add,
softplus, mul) × ~38 linear-attn layers/token. Decode is launch-bound on Spark
GB10, so this is a pure launch-overhead win. The recurrent kernel already does
`exp(g)`, so the only thing that moves is *where* g/beta are produced.

## What changed

### `triton_kernels/gated_delta.py`
- **New `@triton.jit` kernel** `_recurrent_fused_prepare_from_outconv_gqa_gbeta_kernel`.
  It is byte-for-byte identical to the production
  `_recurrent_fused_prepare_from_outconv_gqa_kernel` (qk-l2norm, scale,
  decay `s*g_exp`, `kv_mem`, `delta`, `s_new`, `out`, state-store) **except** the
  source of g/beta. Instead of loading pre-computed `g_ptr`/`beta_ptr`, it takes
  raw `a, b, dt_bias, neg_exp_A_log` (`[NUM_V_HEADS]` each, indexed by head) and
  computes per-head in float32:
  ```
  beta  = tl.sigmoid(b_h)
  g     = neg_exp_A_log_h * log(1 + exp(a_h + dt_bias_h))   # softplus
  g_exp = exp(g)                                            # same exp the old kernel did
  ```
- **New wrapper** `recurrent_gated_delta_fused_prepare_from_outconv_gqa_gbeta(out_conv, a, b, dt_bias, neg_exp_A_log, s_prev)`.
  Same shape/dtype validation, grid `(NUM_V_HEADS, 4)`, block params, and
  `LYNN_LINEAR_ATTN_RECURRENT_INPLACE` handling as the precomputed-g/beta
  wrapper. Returns `(out, s_new)` identically.
- Existing kernels/wrappers (incl. the precomputed `..._from_outconv_gqa`) are
  **untouched**.

### `engine/incremental_decode.py`
- New optional import of the `_gbeta` wrapper in its **own** `try/except` (so its
  absence can never disable the existing precomputed path).
- In `decode_linear_attn`: `dt_bias`/`neg_exp_A_log` are still resolved as
  before, but `beta = b.sigmoid()` / `g = neg_exp_A_log * softplus(a+dt_bias)`
  are now computed **only when not folding** (`if not fuse_gbeta:`). On the
  outconv recurrent path, when folding is enabled we call the new `_gbeta`
  wrapper with raw `a, b, dt_bias, neg_exp_A_log`. Every other path is byte-for-
  byte unchanged.

### `reports/qwen36_35b/claude_internal_gbeta_fold.md`
- This file.

## New env flag
`LYNN_LINEAR_ATTN_FUSE_GBETA` (default `"0"`). Fold is active only when **all** hold:
- `LYNN_LINEAR_ATTN_FUSE_GBETA=1`, AND
- `use_outconv_recurrent` is already true (i.e. `recurrent_backend=triton_fused_prepare`
  + `LYNN_LINEAR_ATTN_GQA_RECURRENT=1` + `LYNN_LINEAR_ATTN_RECURRENT_FROM_OUTCONV=1`), AND
- the `_gbeta` Triton kernel imported successfully.

Otherwise it falls back to the existing precomputed-g/beta path with zero change.
This matches exactly the stack toggled by `scripts/spark_gbeta_fold_ab.py`.

## Bonus micro-win (already realized for the fused path)
When folding, `a`/`dt_bias` stay raw and are cast to fp32 *inside* Triton, so the
per-token `a.float()` / `dt_bias.float()` casts are skipped on the Python side
(no separate dt_bias cache needed). The non-fused path keeps the original casts.

## Risks / things to verify on GPU
1. **Token-exactness is the target, not guaranteed.** The recurrent step is
   byte-identical, but g/beta now come from Triton math instead of PyTorch:
   - `beta`: PyTorch computes `b.sigmoid()` in **bf16** then the old kernel
     upcasts to fp32; the fold computes `tl.sigmoid` in **fp32**. More accurate,
     but not bitwise-identical to the bf16 sigmoid.
   - `g`: PyTorch uses `F.softplus` (with its `threshold=20` branch) in fp32; the
     fold uses naive `log(1+exp(x))` in fp32. Equal for normal magnitudes, last-
     bit differences possible.
   Net: the `TOKEN_EXACT_A_vs_B` check in `spark_gbeta_fold_ab.py` *should* pass
   but could show a divergence after enough decode steps. If it diverges, this is
   the cause — not a logic bug. (If strict bit-parity is required, match PyTorch's
   exact bf16-sigmoid / softplus-threshold semantics in the kernel.)
2. **Softplus overflow:** naive `log(1+exp(a+dt_bias))` has no large-x guard. For
   gated-delta, `a+dt_bias` is small/negative in practice, so this is unlikely,
   but worth a sanity check on extreme inputs.
3. **Pre-existing duplication (left untouched on purpose):** an earlier unwired
   research function `recurrent_gated_delta_fused_prepare_from_outconv_ab_gqa`
   (+ `_recurrent_fused_prepare_from_outconv_ab_gqa_kernel`) already does an
   equivalent fold, but with the manual `1/(1+exp(-b))` sigmoid form and a
   different (unwired) arg order. It's only referenced by
   `benchmarks/p176_qwen36_recurrent_from_outconv_ab_candidate.py`. I did **not**
   touch or remove it (per "do not alter the existing kernel"), so there are now
   two near-identical fold kernels. Recommend the lead pick one canonical impl and
   delete the other once `_gbeta` is verified.

## How to verify
`scripts/spark_gbeta_fold_ab.py` already exists and toggles
`LYNN_LINEAR_ATTN_FUSE_GBETA` OFF→ON on the full RC stack, reporting decode TPS
and `TOKEN_EXACT_A_vs_B`. Run it in the eval docker on Spark.

Parse sanity (local, no GPU):
```
python3 -c "import ast; ast.parse(open('engine/incremental_decode.py').read()); ast.parse(open('triton_kernels/gated_delta.py').read()); print('PARSE_OK')"
```
→ `PARSE_OK`.
