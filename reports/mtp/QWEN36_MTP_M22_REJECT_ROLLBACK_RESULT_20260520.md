# Qwen3.6-35B MTP M22 Reject-Rollback Full-State Probe Result · 2026-05-20

## Headline

**`LYNN_MTP_K2_REJECT_ROLLBACK=full_state` is a no-op.** Replacing the
lean `restore_recurrent_conv` (recurrent + conv + seq_len) with the
runner's full `_snapshot_state` / `_restore_state` (KV + recurrent +
conv + seq_len) on the K=2 REJECT path produces **bit-identical**
end-to-end behavior — same exact-match (3/6), same mean prefix (44.0),
same accept rate (0.7485), same per-prompt first-divergence event and
kind. **The K=2 forward's residual state lives outside everything the
runner currently snapshots.**

Per M22 acceptance: `full_snapshot_insufficient__residue_outside_captured_state`.

## Setup

Spark Qwen3.6-35B-A3B Lynn-native W4A16 NVFP4, official fused MTP
sidecar, canonical 6-prompt smoke set, `--max-new 64`. Apples-to-apples
**eager** (no `LYNN_LINEAR_BLOCK_GRAPH`) + **shadow off**. Single
resident runner across all configs.

## Result

| Config | Exact | Mean prefix | Accept | Effective TPS | TPS ratio vs baseline |
|---|---:|---:|---:|---:|---:|
| baseline_greedy (eager) | 6/6 | 64.0 | — | 31.70 | 1.000 |
| spec_k1_sequential | **6/6** | 57.3 | 79.18% | 26.42 | 0.833 |
| spec_k1_batched_default | 3/6 | 44.0 | 74.85% | 24.50 | 0.773 |
| spec_k1_batched_fullrollback | **3/6** | 44.0 | 74.85% | 24.07 | 0.759 |

Raw report: [`reports/mtp/mtp_m22_reject_rollback_20260520_123627.json`](mtp_m22_reject_rollback_20260520_123627.json)

### Per-prompt first divergence — default vs fullrollback

| Prompt | default first_div_event / kind / prefix / exact | fullrollback first_div_event / kind / prefix / exact |
|---|---|---|
| 0 (Q4_K_M vs NVFP4) | event 17 / reject / 28 / ✗ | event 17 / reject / 28 / ✗ |
| 1 (中文 speculative) | event 26 / accept / 39 / ✗ | event 26 / accept / 39 / ✗ |
| 2 (Fibonacci) | — / — / 64 / ✓ | — / — / 64 / ✓ |
| 3 (train 60mph) | — / — / 64 / ✓ | — / — / 64 / ✓ |
| 4 (JSON Tokyo)¹ | (see JSON) | (see JSON) |
| 5 (MoE router) | (see JSON) | (see JSON) |

The first-divergence event index and kind are **identical** across the
two configs for every prompt, confirming full-state rollback does not
change the verifier's output stream.

¹ Two of the three exact-matching prompts are short-completion (#3
Fibonacci, #4 JSON Tokyo) whose generation terminates well below the
64-token budget. The remaining exact prompt (#3 train) is also short.
Long-form prompts diverge — same as M19/M20 graph-baseline runs.

## Overhead of full snapshot/restore

- Mean event step seconds: default 0.07137s, fullrollback 0.07266s
  → +1.8% per spec event when reject_rollback=full_state is on
- Effective TPS: default 24.50 / fullrollback 24.07 → −1.7%

The full snapshot/restore is cheap (single KV clone is ~1ms on Spark
GB10), but offers zero correctness improvement on this set.

## What M22 rules out

1. **KV cache residue is not the bug.** Even with KV fully restored
   pre-K=2-forward, exact-match stays pinned at 3/6 with identical
   per-event traces. The K=2 forward's KV writes at
   `[seq_len, seq_len+1]` are correctly overwritten by the subsequent
   T=1 re-decode of pending (state.seq_len = pre+0 → write at pre+0
   visible to attention, KV[pre+1] invisible).
2. **`runner._snapshot_state` is not the missing dimension.** Whatever
   the K=2 forward perturbs survives a full restore.

## Where the residual must live

By elimination — captured state (KV + recurrent + conv + seq_len) is
not the bug; `_lm_head_logits` BF16 path is not the bug (M21);
`canonical_t1` accept argmax is not the bug (M21); the K=2 forward's
own hidden state is bit-exact under `t1_loop` for a single round
(M18). The residual must be **process state outside every captured
tensor**:

1. **Triton autotune cache** keyed on (shape, dtype, device). The K=2
   path calls T=1 kernels twice in a per-position loop (M9 MoE, t1_loop
   full-attn, per-position linear-attn). The autotune cache may pick
   a different "best" kernel after observing K=2-burst-of-T=1 vs
   genuinely sequential T=1 — subsequent T=1 calls then take a slightly
   different code path.
2. **CUDA workspace allocator** with `expandable_segments=True`. After
   the K=2 forward releases its activation buffers, the allocator's
   free-list / segment layout differs from a clean post-prefill state.
   Subsequent T=1 forwards get different memory layouts → different
   rounding behavior in reduction kernels (FP4 quantization, RMSNorm
   accumulation, SDPA softmax).
3. **Native FP4 lm_head workspace** (`quantize_fp4_m1_native` per-row
   branch). The K=2 path calls it with 1D `[K]` row inputs; T=1 calls
   it with 2D `[1, K]`. Workspace buffer reuse may leave subtly
   different state.
4. **Runner-level lazy caches** (e.g., `_linear_block_graph_slot`)
   may be initialized lazily by some code path the K=2 forward
   indirectly triggers.

## Sequential is healthy — so the K=2 batched primitive itself is the dial

`spec_k1_sequential` is 6/6 exact at 0.83× baseline TPS (lossy verifier
overhead, expected). `spec_k1_batched_default` is 3/6 exact at 0.77× —
same TPS class, half the correctness. The two paths share the SAME
post-event T=1 re-decode helper (`decode_one_to_logits_and_hidden`),
but only the batched path calls `_decode_layer_k2_fast` first. **Just
calling the K=2 forward is what introduces the residue.**

This narrows the next bisect to: instrument `_decode_layer_k2_fast`
to record what process-side state it changes vs a pure T=1 chain
(autotune cache misses, allocator stats, lazy buffer creation).

## Promotion status

**NOT promoted.** Exact-match stays at 3/6; TPS ratio for batched
(0.77) is below baseline. `LYNN_MTP_K2_REJECT_ROLLBACK=full_state`
ships as an opt-in diagnostic toggle only.

`--max-new 128` re-run not warranted: M22 acceptance gate requires
exact=6/6 at `--max-new 64` before extending. We hit 3/6 with full
rollback identical to default; doubling the window won't change the
conclusion.

## Recommended next probe (M23)

1. **Reset Triton autotune cache** between spec events (or disable
   autotune entirely) and re-run the M22 comparison. If exact jumps
   to 6/6, autotune is the residue source.
2. **Force eager allocator** (`PYTORCH_CUDA_ALLOC_CONF=` empty or
   without `expandable_segments`) and re-run. If exact jumps, the
   allocator's segment state is the residue.
3. **Call K=2 forward but discard output** — i.e., run
   `_decode_layer_k2_fast` then ignore its results and use the
   sequential T=1 chain output. If exact jumps to 6/6, the *act* of
   running the K=2 forward perturbs subsequent T=1; if stays at 3/6,
   the K=2 forward is causally orthogonal (very unlikely given we've
   eliminated all captured state already).

## Cross-reference

- M18 layer sweep (full-attn k2 SDPA = sole layer-level drift): [commit `32731a9`](https://github.com/MerkyorLynn/lynn-engine/commit/32731a9)
- M19/M20 smoke (2/6 exact graph baseline)
- M21 lm_head + canonical_t1 bisect: [commit `432a29e`](https://github.com/MerkyorLynn/lynn-engine/commit/432a29e)
- M22 switch + probe: [commit `7e2704b`](https://github.com/MerkyorLynn/lynn-engine/commit/7e2704b)
- Memory: `project_lynn_engine_t1_only_kernel_contract_20260519`
