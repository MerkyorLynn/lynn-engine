# R6000 Full-Attention Graph Runtime ABI

Date: 2026-05-17

## Why This Exists

The R6000 Config D service path is now pinned:

```text
safe decode: ~98-101 tok/s
token wall: 10.17 ms
linear-block graph replay: 6.55 ms
10 eager full-attention layers: 3.11 ms
host gap: 0.15 ms
```

The remaining gap is not mostly Python overhead. A C++ token loop alone cannot
recover the missing `~3.5 ms/token`.

P9H/P9I/P9J changed the full-attention decision:

| Probe | Result |
|---|---:|
| P9H layer31 fixed-position graph | 3.90x replay speedup, exact output/KV parity |
| P9I layers 3/15/31/39, positions 10/14/32 | 12/12 exact parity, 4.09x mean replay speedup |
| P9J mutable input buffer | 4/4 exact parity after swapping graph input, 4.21x mean replay speedup |
| P9L runner slot smoke | `_capture_full_attn_layer_graph_slot` exact parity on R6000 |
| P9M pre-captured slot on populated KV | layer31 position39 exact output/KV parity, 0.295 ms replay |
| P9N hybrid token probe | 10 linear graphs + 10 full-attn slots, greedy pass, 9.53 ms one-shot |
| P9O layerwise diff | first strict drift at full-attn slot layer3; linear block0 is exact |
| P9P single-state layerwise diff | confirms P9O drift is not a two-prefill artifact |
| P9Q layer3 capture-mode probe | layer3 alone is exact for real-capture and pre-capture |
| P9R graph-pool/order probe | fresh full slot after linear capture is exact; stale slot drifts |
| P9S capture-order token probe | linear-first capture stays greedy-safe but not strict-exact |
| P9T separate-state token probe | separate linear/full graph states stay greedy-safe but not strict-exact |

This makes full-attention reusable graphing the strongest non-MTP R6000 speed
lever found today.

## Non-Goal

Do not revive `LYNN_FULL_TOKEN_GRAPH_SLOT=1` as the serving path. Spark already
measured that strict-slot mode at about `10 tok/s` because it captures a whole
decode graph every token:

```text
capture every token -> ~80 ms capture + ~10 ms replay
```

That is a diagnostic path only.

## Required ABI

The serving graph path needs a resident graph state, not per-request graph
state copies:

```text
runner owns:
  LynnInferenceState graph_state
  per-layer graph slots for full-attention layers
  mutable input/output buffers per slot
  static pos tensors or position-keyed graph families

request path:
  reset graph_state
  prefill writes KV/recurrent/conv state directly into graph_state
  decode replays graph slots against graph_state
  unsupported position/backend -> eager fallback
```

Copying full-attention KV prefixes into a separate graph state at every token is
not acceptable; it would erase the replay win.

## Slot Shape

Minimum Python-side slot:

```text
FullAttentionLayerGraphSlot:
  layer_idx: int
  seq_len: int
  input_buf:  [1, 1, 2048] bf16
  output_buf: [1, 1, 2048] bf16
  pos_tensor: [1, 1] int64
  graph: torch.cuda.CUDAGraph
```

Replay contract:

```text
input_buf.copy_(h)
graph.replay()
h = output_buf
```

P9J verifies that this input buffer is genuinely mutable: changing `input_buf`
changes graph output while preserving exact eager parity for both inputs.

P9L then verifies the same contract through the actual runner scaffold:

```text
reports/p16_155/p9l_r6000_full_attn_runner_slot_smoke_20260517_140335.json
layer31 position10
case-A output/KV max_abs: 0
case-B output/KV max_abs: 0
graph output delta A->B rel_l2: 1.241
```

P9M strengthens the serving ABI: a full-attention graph slot can be captured
before real request KV exists, then replayed after prefill and 32 eager prefix
tokens have populated the same cache tensors. The R6000 report is:

```text
reports/p16_155/p9m_r6000_precaptured_full_attn_graph_configd_20260517_144739.json
layer31 position39
graph_ms: 0.2946
output/KV write max_abs: 0
```

P9N combines the existing 10 linear-attention block graphs with 10 pre-captured
full-attention slots for one whole decode token:

```text
reports/p16_155/p9n_r6000_hybrid_full_attn_graph_slots_configd_20260517_144814.json
one_shot_graph_ms: 9.5298
one_shot_graph_tps: 104.93
greedy_pass: true
strict_logit_pass: false
```

This is a graph-composition proof, not yet the target server number. P9O then
locates the strict drift:

```text
reports/p16_155/p9o_r6000_hybrid_full_attn_graph_layerwise_diff_configd_20260517_145103.json
first_drift: full_attention_slot layer3
linear block0 diff: max_abs 0
layer3 slot diff: max_abs 0.02124, rel_l2 0.1479
```

The next runtime patch should therefore focus on the first full-attn slot's
capture/replay state contract, not on the linear block graph.

P9P repeats that check with one shared state restored from the same prefill
base, matching the P9N execution model:

```text
reports/p16_155/p9p_r6000_hybrid_full_attn_graph_single_state_diff_configd_20260517_145702.json
first_drift: full_attention_slot layer3
linear block0 diff: max_abs 0
layer3 slot diff: max_abs 0.02124, rel_l2 0.1479
```

So the layer3 drift is a real composition issue, not an artifact of comparing
two independently-prefilled states.

P9Q/P9R narrow the bug further. Layer3 alone is exact when tested as a single
slot:

```text
reports/p16_155/p9q_r6000_full_attn_slot_capture_mode_layer3_configd_20260517_150212.json
real_capture_exact: true
precapture_exact: true
```

But the graph-pool/order probe shows stale captured slots can drift after other
graphs are captured, while a fresh slot captured after the linear graph capture
is exact:

```text
reports/p16_155/p9r_r6000_full_attn_slot_graph_pool_order_layer3_configd_20260517_150435.json
pre_slot_before_linear_capture_diff.max_abs: 0.02490
same_pre_slot_after_linear_capture_diff.max_abs: 0.046875
fresh_pre_slot_after_linear_capture_diff.max_abs: 0
```

P9S then tests the naive "capture linear first, full-attn second" whole-token
order. It remains greedy-safe but strict logits get worse:

```text
reports/p16_155/p9s_r6000_hybrid_full_attn_graph_slots_order_configd_20260517_150629.json
one_shot_graph_ms: 9.6734
greedy_pass: true
strict_logit_pass: false
logit_diff.max_abs: 3.53125
```

Current read: full-attn slots are individually viable, but the mixed graph
family needs explicit graph-pool/state ownership rather than ad hoc capture
ordering.

P9T then separates the linear graph state from the full-attention KV graph
state. It does not fix strict drift:

```text
reports/p16_155/p9t_r6000_hybrid_graph_separate_state_configd_20260517_151417.json
one_shot_graph_ms: 9.5324
greedy_pass: true
strict_logit_pass: false
logit_diff.max_abs: 3.53125
```

So simple state separation is not enough. The next proof should isolate CUDA
graph memory-pool ownership or capture full-attn slots in a fresh graph pool per
family; if that still drifts, the production path should move toward a native
static full-attn layer boundary instead of composing PyTorch CUDAGraph objects.

## Graph Key

Graph slots must be invalidated by:

```text
model fingerprint
layer_idx
seq_len / position family key
dtype
LYNN_MOE_IMPL
LYNN_MOE_FAST_FIXED
LYNN_NATIVE_DOWN_BACKEND
LYNN_PACKED_DECODE*
LYNN_QK_NORM_ROPE_BACKEND
LYNN_FULL_ATTN_DECODE_BACKEND
native FP4 lm_head is irrelevant for per-layer slots
```

The first implementation should be conservative: if any key is not exactly
recognized, run the existing eager full-attention layer.

## Position Problem

Current `_decode_layer` calls `decode_full_attn(... cached_seq_len=state.seq_len)`.
That Python integer fixes the KV slice length during CUDA graph capture:

```text
K_used = K_cache_full[:, :, :cached_seq_len + 1, :]
```

Therefore a graph captured at position `P` is not a general graph for position
`P+1`. There are two viable paths:

1. position-keyed graph family for common serving positions;
2. static-window/native full-attention kernel where write position and mask are
   dynamic graph inputs.

Path 1 is faster to wire and should be used for proof-of-service on repeated
bench prompts. Path 2 is the production architecture.

## Expected Speed Envelope

P26 measured ten eager full-attention layers at `3.11 ms/token`. P9I/P9J show
about `4.1x` replay speedup for the full-layer boundary. If this transfers to
the service path:

```text
full-attn budget: 3.11 ms -> ~0.76 ms
saved:            ~2.35 ms/token
base token wall:  10.17 ms -> ~7.82 ms
base TPS:         ~98 TPS -> ~128 TPS
```

That still does not close 155 alone. It makes the required MTP multiplier much
smaller:

```text
128 TPS x 1.22 accepted-token multiplier = 156 TPS effective
```

So the merged target is now clear:

```text
R6000 base runtime: full-attn graph/static boundary to ~125-130 tok/s
A100 MTP: saved sidecar accept >=55%, then serving verifier >=1.20x effective
combined: >155 effective tok/s
```

## Implementation Order

1. Give linear-block and full-attn graph families explicit graph-pool/state
   ownership; then make P9P/P9S strict at layer3.
2. Add opt-in `LYNN_FULL_ATTN_LAYER_GRAPH=1` with eager fallback.
3. Reuse the existing `LynnInferenceState` as the resident graph state for
   single-stream serving.
4. Capture per-layer slots for one or more positions after prefill warmup.
5. Add a parity gate comparing eager full-attn layers vs graph slots for the
   same request.
6. Run repeated-prompt service TPS with position-family hits.
7. Only after that, start the static-window/native kernel refactor.

No full-active W4A8 promotion depends on this path until generation gates stay
AMBER/GREEN.
