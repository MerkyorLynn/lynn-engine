# Qwen3.6-35B MTP M18 K=2 Layer-Type Sweep Result · 2026-05-20

## Headline

`LYNN_FULL_ATTN_K2_BACKEND=t1_loop` alone is **necessary and sufficient**
at the layer level. **Linear-attention K=2 is innocent** — drift comes
entirely from `decode_full_attn_k2` SDPA. Whether or not
`LYNN_MTP_K2_LINEAR_ATTN_MODE=t1_loop` is set, the K=2 verifier matches
sequential T=1 bit-exactly when full-attn t1_loop is on, and drifts to
the same numbers when it is off.

This nails which segment of the K=2 verifier can be safely batched
(linear-attn), which cannot (full-attn SDPA), and removes the
"linear-attention is the problem" framing from M15/M16.

## Result

Spark Qwen3.6-35B-A3B Lynn-native W4A16 NVFP4, official fused MTP
sidecar, M16 zero-advance bisect setup (event-5 pending=25 / draft=271):

| combo | full-attn | linear-attn | drift | first_bad_layer | logits pos0 max_abs | logits pos1 max_abs |
|---|---|---|---|---|---:|---:|
| `k2_both` (default)        | k2 (SDPA)        | k2 (per-pos internal) | **DRIFT** | L5 linear_attention | 0.9238 | 0.7422 |
| **`t1_full_attn_only`**    | **t1_loop** (2×T=1) | **k2** (per-pos internal) | **EXACT** | — | **0.0** | **0.0** |
| `t1_linear_attn_only`      | k2 (SDPA)        | t1_loop (2×T=1 + state.update interleave) | **DRIFT** | L5 linear_attention | 0.9238 | 0.7422 |
| `t1_both`                  | t1_loop          | t1_loop               | EXACT | — | 0.0 | 0.0 |

JSON: [`reports/mtp/mtp_m18_k2_layer_sweep_20260520_104335.json`](mtp_m18_k2_layer_sweep_20260520_104335.json)

## Interpretation

### `t1_linear_attn_only` ≡ `k2_both` numerically

The fact that `t1_linear_attn_only` and `k2_both` produce **byte-identical
diff numbers** (same first_bad layer, same max_abs, same cosine) proves
two things:

1. `decode_linear_attn_k2` (per-position internal with end-of-block-only
   `state.update_linear_attn_state`) is **numerically equivalent** to a
   true per-position T=1 chain with interleaved state.update — at least
   for the zero-advance event-5 probe. The K=2 vs sequential linear-attn
   chain produces matching hidden state regardless of the state-update
   schedule.
2. The drift seen at first_bad layer 5 (linear_attention) in M15/M16 is
   **downstream propagation** from upstream `decode_full_attn_k2`, not
   linear-attn arithmetic. Layer 3 is the first full_attention layer (per
   Qwen3.6 30/40 hybrid pattern: full at 3,7,11,...,39); its
   K=2-SDPA-induced delta enters layers 4 and 5 as input-channel drift
   and crosses the cosine threshold there first.

### `t1_full_attn_only` is the minimal-impact layer-level fix

With **only** `LYNN_FULL_ATTN_K2_BACKEND=t1_loop` set, K=2 vs sequential
is strict (max_abs = 0.0 at the final logits). Linear-attention can
continue running through `decode_linear_attn_k2` without contributing
drift. This matches the [t1_loop fix from commit `193a392`](../../../../lynn-engine/commit/193a392)
already on main; M18 confirms this fix is also **sufficient** at the
layer level (not just necessary).

The new `LYNN_MTP_K2_LINEAR_ATTN_MODE=t1_loop` knob (introduced in
commit `087b910` for this sweep) is therefore **redundant for layer-
level correctness** and should not be promoted as part of any production
path. It remains useful as a diagnostic toggle.

## Reconciliation with M13's 2/6 exact gap

M13 smoke ran with `LYNN_FULL_ATTN_K2_BACKEND=t1_loop` and showed K=2
batched accept 75.17% but only 2/6 exact prefix-match (mean prefix 65 /
100). M18 proves the **layer-level math is strict** under that env.
Therefore M13's residual sequence-level divergence must originate
**outside the K=2 forward**:

- Most likely: `_lm_head_logits` per-row branch (M11, commit `76a4445`).
  When `all_positions=True` and `h2d.shape[0] != 1`, it iterates `for row
  in h2d` yielding 1D `[K]` tensors, then calls
  `quantize_fp4_m1_native(row.contiguous())`. If
  `quantize_fp4_m1_native` derives scale differently for 1D `[K]` vs 2D
  `[1, K]` input (an ULP-level difference would suffice to flip argmax on
  near-tied tokens), the K=2 verifier's lm_head argmax disagrees with
  sequential's argmax at one position every few hundred rounds —
  consistent with mean prefix 65 / 100.

This was flagged in
[`QWEN36_MTP_M13_PROBE_ENV_GAP_HYPOTHESIS_20260520.md`](QWEN36_MTP_M13_PROBE_ENV_GAP_HYPOTHESIS_20260520.md).
M18's strict layer-level result confirms hypothesis (1) (lm_head
dispatch) over the alternative hypothesis (per-layer drift in
production env) — at least at layer-level resolution; production env may
still introduce additional drift via `LYNN_PACKED_DECODE`/`LYNN_PACKED_SHARED_EXPERT`
fast-path activation, which M18 does not yet test (BASE_ENV inherited
from probe defaults, no production fast-path flags).

## Recommended next steps

1. **Promote `LYNN_FULL_ATTN_K2_BACKEND=t1_loop` as the default** in
   `_decode_layer_k2` (keep current opt-out behavior accessible via
   `LYNN_FULL_ATTN_K2_BACKEND=k2`). This costs ~16% extra layer launches
   in full-attn layers (10 / 40 layers) but unlocks correct K=2 verifier
   math at every layer.
2. **Inspect `quantize_fp4_m1_native` 1D vs 2D scale derivation** —
   either reshape input to `[1, K]` inside the per-row loop, or
   demonstrate scale is shape-invariant.
3. **Re-run M18 with full smoke `BASE_ENV`** (add `LYNN_PACKED_DECODE=1`,
   `LYNN_PACKED_SHARED_EXPERT=1`, `LYNN_LINEAR_ATTN_INPROJ_FUSED_NATIVE_FP4=1`,
   `LYNN_FULL_ATTN_QKV_FUSED=1`) to verify `t1_full_attn_only` remains
   strict under production fast paths.
4. **Defer true batched `decode_full_attn_k2` SDPA fix** as a
   longer-horizon kernel-level effort. Layer-level work is now
   bottlenecked by lm_head per-row + production-env probe gap, not by
   full-attn k2 SDPA rewriting.

## Cross-reference

- M9 MoE per-position fix: [commit `db08d7e`](https://github.com/MerkyorLynn/lynn-engine/commit/db08d7e)
- M11 native FP4 lm_head per-row: [commit `76a4445`](https://github.com/MerkyorLynn/lynn-engine/commit/76a4445)
- M12 lm_head opt-in: [commit `49320a1`](https://github.com/MerkyorLynn/lynn-engine/commit/49320a1)
- M12 t1_loop full-attn fallback: [commit `193a392`](https://github.com/MerkyorLynn/lynn-engine/commit/193a392)
- M16 bisect: `reports/mtp/QWEN36_MTP_M16_K2_BISECT_RESULT_20260520.md`
- M17 canonical scaffold: `reports/mtp/QWEN36_MTP_M17_CANONICAL_K2_RESULT_20260520.md`
- M18 knob + probe: [commit `087b910`](https://github.com/MerkyorLynn/lynn-engine/commit/087b910)
- M13 env gap hypothesis: `reports/mtp/QWEN36_MTP_M13_PROBE_ENV_GAP_HYPOTHESIS_20260520.md`
- Memory: `project_lynn_engine_t1_only_kernel_contract_20260519`
