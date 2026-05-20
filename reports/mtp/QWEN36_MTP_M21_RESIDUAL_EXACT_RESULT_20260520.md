# Qwen3.6-35B MTP M21 Residual-Exact Bisect Result · 2026-05-20

## Headline

**Neither `LYNN_MTP_K2_LM_HEAD_MODE=bf16` nor `LYNN_MTP_K2_ACCEPT_SOURCE=canonical_t1`
fixes the K=2 batched exact-match gap.** Both switches change the K=2
verifier's accept signal numerically (lm_head dispatch path / accept
argmax source) but exact stays at 0/6 in apples-to-apples eager
comparison. Per the M21 acceptance rules, the residual bug lives in
**speculative commit / state rollback / new_ids emission** — *not* in
the K=2 lm_head per-row path and *not* in the K=2 accept argmax.

The per-event trace pinpoints the failure mode: after a K=2 forward is
rejected and `restore_recurrent_conv` (lean snapshot: recurrent + conv +
seq_len only, **no KV**) is called, the subsequent canonical T=1
re-decode of the pending token produces a different argmax than a clean
T=1 decode of that same token from the same prefix. **The K=2 forward
leaves residual state that the lean restore does not undo.**

## Setup

Spark Qwen3.6-35B-A3B Lynn-native W4A16 NVFP4, official fused MTP
sidecar, canonical 6-prompt smoke set, max_new=128, runner shared across
configs in one process.

`BASE_ENV` (held constant for all configs):

```
LYNN_MOE_IMPL=packed_nvfp4
LYNN_LINEAR_ATTN_RECURRENT_INPLACE=1
LYNN_NATIVE_FP4_LM_HEAD=1
LYNN_PACKED_DECODE_BACKEND=native_fast_2d
LYNN_PACKED_DECODE=1
LYNN_PACKED_SHARED_EXPERT=1
LYNN_LINEAR_ATTN_INPROJ_FUSED_NATIVE_FP4=1
LYNN_FULL_ATTN_QKV_FUSED=1
LYNN_FULL_ATTN_K2_BACKEND=t1_loop      # M18 layer-level necessary fix
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

All configs additionally set `LYNN_MTP_SHADOW_VERIFY=0` and run
**eager** (no `LYNN_LINEAR_BLOCK_GRAPH`) — the linter normalized the
baseline_greedy config to eager so the bisect compares eager-vs-eager,
removing graph-vs-eager numerical drift from the picture. M19/M20
smoke compared graph baseline (LINEAR_BLOCK_GRAPH=1) against eager
spec; that graph-vs-eager axis is a separate residual we are not
investigating here.

## Result

| Config | Exact | Mean prefix | Accept | Effective TPS |
|---|---:|---:|---:|---:|
| baseline_greedy (eager) | 6/6 | 128.0 | — | 25.0 decode |
| spec_k1_batched_default | 0/6 | 1.0 | 0.7411 | 23.2 effective |
| spec_k1_batched_lmhead_bf16 | 0/6 | 1.0 | 0.7269 | 22.1 effective |
| spec_k1_batched_canonical_t1_accept | 0/6 | 1.0 | 0.7590 | 15.6 effective |
| spec_k1_batched_both_switches | 0/6 | 1.0 | 0.7512 | 14.5 effective |

Raw report: [`reports/mtp/mtp_m21_residual_exact_20260520_114144.json`](mtp_m21_residual_exact_20260520_114144.json)

### What the accept-rate dial does

Each switch demonstrably changes the K=2 verifier's accept signal —
none of the configs are no-ops, none collapses to the same accept rate.

* `lmhead_bf16` (0.7269) — replacing the native FP4 per-row lm_head with
  BF16 `F.linear` shifts the accept rate **down** by ~1.4pp. That is
  numerical evidence that the M11 per-row `quantize_fp4_m1_native`
  branch returns slightly different logits than the equivalent BF16
  F.linear path for the same 2-position input.
* `canonical_t1_accept` (0.7590) — using the *canonical* sequential T=1
  argmax for the accept comparison (rather than the K=2 forward's
  pos-0 lm_head argmax) shifts the accept rate **up** by ~1.8pp. The
  K=2 verifier is over-rejecting on ~2pp of events relative to the
  reference.
* `both` (0.7512) — combining the two switches puts the accept rate
  midway between, confirming the two are independent numerical
  perturbations.

Despite all three perturbing the accept signal, the end-to-end
exact-match rate stays pinned at 0/6 with mean prefix = 1 token. That
is the M21 bisect's positive result: **the post-M18 layer-level fix
exposes a separate bug downstream of the accept comparison.**

## Where the trace pinpoints the bug

`event_summary` for prompt 0 default config:

```
n_events: 77 (52 accept, 25 reject)
tokens_committed: 129
first_divergence_event_idx: 1
first_divergence_after_accept_or_reject: reject
first_divergence_token_offset: 1
prefix_match_len: 1
```

Event 0 of prompt 0 is a REJECT (`draft="**"` mismatch). The K=2
forward rejected, lean restore was applied, and `decode_one_to_logits_and_hidden(state, 271)`
ran a T=1 re-decode. The result was committed as token 271 → matches
baseline at offset 0 ✓.

Event 1 of prompt 0 is also a REJECT. Spec commits token **248068**;
the eager baseline commits token **238434** at the same offset.

Both events compute their next token via a canonical T=1 decode of
the same pending token (271) starting from a state that *should* be
identical to the baseline's post-prefill state. The lean restore puts
recurrent + conv + seq_len back to pre-K=2; the K=2 forward's KV
writes at `[seq_len, seq_len+1]` get overwritten at offset `seq_len`
by the subsequent T=1 decode of pending. Logically there should be no
visible state difference. Numerically there is — the two "canonical
T=1 decodes" of the same token from the same prefix yield different
argmax.

The lean restore is hiding a state mutation. Candidates:

1. **In-place packed kernel buffers**: when `LYNN_LINEAR_ATTN_RECURRENT_INPLACE=1`
   the recurrent kernel writes through the snapshot's source tensor.
   If the K=2 path's per-position internal threading captures an
   intermediate state by reference and never lets the snap's
   `copy_(...)` overwrite it, restore would be a no-op.
2. **Shared scratch / autotune state in `quantize_fp4_m1_native`**:
   M11's per-row branch (`for row in h2d: quantize_fp4_m1_native(row.contiguous())`)
   passes 1D `[K]` tensors. If the kernel internally maintains a
   warmup or scale cache keyed on shape and 1D vs 2D is treated
   differently, subsequent calls with 2D `[1, K]` may read stale
   scratch.
3. **Triton autotune cache**: separate keys per shape, so K=2 (T=2)
   and T=1 paths have distinct cache entries — but if a kernel's
   captured workspace pointer crosses the K=2 / T=1 boundary, that
   could persist.
4. **CUDA graph slot caching on the runner**: even with
   `LYNN_LINEAR_BLOCK_GRAPH=0`, `runner._linear_block_graph_slot` may
   already exist from prior usage or a graph capture inside the K=2
   forward itself. Subsequent eager T=1 calls may behave differently
   when this slot is alive vs absent.

The M21 probe does not yet distinguish among (1)–(4). The clean next
action is a **full-state snapshot** (KV + recurrent + conv + seq_len
+ any runner-side scratch) for the K=2 REJECT path, and verify the
T=1 re-decode of pending then matches the eager baseline T=1 decode
bit-exactly.

## Bisect verdict against M21 acceptance rules

| Test | Result | Verdict |
|---|---|---|
| `lmhead_bf16` flips exact to 6/6 | No (still 0/6) | Native FP4 lm_head per-row dispatch is **not** the sole bug. |
| `canonical_t1_accept` flips exact to 6/6 | No (still 0/6) | K=2 accept logits/argmax is **not** the sole bug. |
| Neither switch flips exact | Confirmed | **Root cause: speculative commit / state rollback / new_ids emission.** |

The MD `root_cause` field in the JSON is therefore
`neither_switch_fixes__state_commit_rollback_or_new_ids_emission`.

## Promotion status

**NOT promoted.** Per M21 acceptance rules, batched spec is not
promoted until exact=6/6 with TPS ratio above baseline. Neither
condition is met. `LYNN_FULL_ATTN_K2_BACKEND=t1_loop` remains opt-in;
defaults unchanged.

## Recommended next probe (M22)

1. Replace `restore_recurrent_conv` (lean) on the K=2 REJECT path with
   the runner's existing full-state snapshot/restore
   (`_snapshot_state` / `_restore_state` at
   `engine/resident_runner.py:719+`). Run the M21 probe again.
2. If exact jumps to 6/6 with the full snapshot, lean restore is
   confirmed as the bug — the K=2 path leaves residual state outside
   recurrent/conv/seq_len. Audit what survives.
3. If exact stays at 0/6 even with full snapshot, the residual bug is
   in `runner.outside` / `runner.layer_weights` / kernel-side scratch
   that no snapshot captures — bisect by replacing K=2 forward with a
   no-op (or running it but never reading its output) to see whether
   the *act* of running K=2 perturbs subsequent T=1 decode.
4. Either path tells us the exact axis to harden before any K=2
   batched promotion.

## Cross-reference

- M9: [commit `db08d7e`](https://github.com/MerkyorLynn/lynn-engine/commit/db08d7e) MoE per-position
- M11: [commit `76a4445`](https://github.com/MerkyorLynn/lynn-engine/commit/76a4445) native FP4 lm_head multi-row
- M12: [commit `49320a1`](https://github.com/MerkyorLynn/lynn-engine/commit/49320a1) lm_head opt-in; [commit `193a392`](https://github.com/MerkyorLynn/lynn-engine/commit/193a392) full-attn K=2 t1_loop fallback
- M18 layer sweep: [commit `32731a9`](https://github.com/MerkyorLynn/lynn-engine/commit/32731a9)
  showed `t1_full_attn_only` is **strict** at layer level
- M19 / M20 smoke: 2/6 exact (graph baseline, eager spec) — a different
  reference axis than M21 (eager baseline, eager spec)
- M21 switches + probe: [commit `f517e36`](https://github.com/MerkyorLynn/lynn-engine/commit/f517e36)
- Memory: `project_lynn_engine_t1_only_kernel_contract_20260519`
