# Stream B — full-attn layer graph reuse pool spec

Date: 2026-05-18

Author: Claude (clean-room Stream B lane)

## Problem

The decode loop in ``engine/resident_runner.py`` has three branches
(around line 1351):

1. ``LYNN_FULL_TOKEN_GRAPH_SLOT=1`` — capture-per-token whole-token
   graph. The method ``_capture_full_token_graph_slot`` comment is
   explicit: *"not yet wired into default ``generate()``: future-window
   graph families drift, while current-position graph slots are
   strict."* So this path pays a full re-capture every token and is
   slower than eager.
2. ``linear_block_graphs is None`` — pure eager forward through 40
   layers. Slow on host_gap and launch overhead.
3. ``linear_block_graphs is not None`` — the **current safe-default
   Spark Config D path** — linear-attention blocks replay from
   ``_get_reusable_linear_block_graphs``, **full-attention layers
   still run eager via ``_decode_layer_fast``**.

In path 3, the eager full-attention layers issue ~6–8 kernel launches
each (QKV proj × 3, qk_norm × 2, RoPE, KV cache write, SDPA, attn-out
gate, o_proj) × 10 full-attn layers = ~60–80 launches per token. The
linear-attention blocks replay one CUDA graph per 3-layer block, so
launch overhead there is already amortised.

Meanwhile ``llama.cpp`` on the same R6000 hits **207 single-stream
TPS** at short context on the Q4_K_M-imatrix GGUF, while Lynn-native
W4A16 NVFP4 on R6000 sits at **116.9 TPS**. The 90-TPS gap is largely
**kernel-launch + boundary-fragmentation cost on the full-attn
layers**, not host gap (host_gap is 0.16 ms/token per P26).

## Goal

Add a ``_get_reusable_full_attn_layer_graph_slots`` analog of
``_get_reusable_linear_block_graphs`` so that decode loop branch 3
becomes:

```text
for bi, block in enumerate(linear_block_graphs):
    block["input"].copy_(h)
    block["graph"].replay()
    h = block["output"]
    full_slot = full_attn_graph_slots[bi]
    full_slot["input"].copy_(h)
    full_slot["graph"].replay()
    h = full_slot["output"]
```

i.e. **every layer in the safe-default decode loop is a CUDA-graph
replay**, end-to-end. Host gap stays where it is (already 0.16 ms);
GPU launch overhead and per-op dispatch cost drops by 60–80 launches /
token, which is the realistic ROI window for closing the 117 → 207
TPS gap.

## Why not just turn on ``LYNN_FULL_TOKEN_GRAPH_SLOT=1``?

The existing whole-token slot captures the entire 40-layer +
embedding + lm_head graph at the current ``state.seq_len``. Because
``state.seq_len`` changes every token, the slot is **valid for one
token only** — every step pays a fresh capture cost (P13 smoke
docstring documents this). On R6000 this re-capture cost dominates,
so the toggle is research-only today.

Per-layer slots have a tighter scope. The full-attention KV slice
that mutates with ``state.seq_len`` is the only piece that needs
re-capture; the linear-attention blocks already manage their state
buffers cleanly. By bucketing on ``seq_len`` (e.g. recapture every
N = 256 tokens), we amortise the capture cost over N decode steps.

## Acceptance gates (2026-05-18 hand-off)

The candidate **must** carry **all three** gates together (microbench
latency alone is a research artifact and cannot promote):

| Gate | DEFAULT promote | AMBER promote |
|---|---|---|
| P37 exact-greedy | 3/3 required | drift OK |
| Hard structured | 40/40 | 70/70 (opt-in only) |
| P25 512-token decode TPS | ≥ 108 | ≥ 118 |

Stream B competitive bar (post-deliverable):
- safe-default-stage target **125+ TPS** (≥ Atlas published Qwen3.6
  no-MTP single-stream baseline) before claiming win.
- Stream B sprint **118 TPS** stays as the AMBER bar.
- 207 TPS = llama.cpp Q4_K_M same-hardware reference; 122 TPS is the
  final stack-with-Stream-A target.

A graph-replay candidate that fails P37 3/3 closes immediately. Graph
replay is byte-identical kernel sequence by construction, so a P37
fail means the wrapper Python is mutating state outside the capture
window — that bug must be fixed, not papered over with AMBER bar.

## Lifecycle design

### Capture trigger

``_get_reusable_full_attn_layer_graph_slots(state, h_seed, pos_tensor)``
is called from the decode loop just before token generation starts
(same call site as ``_get_reusable_linear_block_graphs``). It returns
``(slots, capture_seconds, created_now)`` where ``slots`` is a list
of length ``num_full_attn_layers`` (= 10 for Qwen3.6-35B-A3B).

### Bucket boundary

A ``slot`` is valid for a range of ``state.seq_len`` values. The
bucket size lives behind an env var
``LYNN_FULL_ATTN_LAYER_GRAPH_BUCKET`` (default 256). When
``state.seq_len`` crosses a bucket boundary, the slot is invalidated
and re-captured next step.

The bucket size is configurable so a sweep can find the right
trade-off between capture cost and KV-slice shape stability. Smaller
buckets = more captures = more time spent re-capturing; larger buckets
= fewer captures but each replay assumes a KV slice longer than the
true window (which is fine because attention output indexes by current
position, not by buffer length — only the SDPA kernel launch shape
needs to be valid).

### Per-request state

Each request gets its own copy of the ``state.k_cache``,
``state.v_cache`` because graph replay assumes stable tensor
addresses. The reuse pool keeps the **graph + input/output buffers**
resident across requests; the **per-request KV cache** is copied into
the slot before decode (mirrors ``_copy_linear_state``).

### Invalidation

The slot is invalidated when:
- ``state.seq_len`` crosses a bucket boundary (most common)
- ``state.max_seq_len`` changes (rare; serving restart)
- The runner reloads weights

The slot is **not** invalidated by argmax / sampling / chat template
changes — only by structural KV-cache layout changes.

## Code surface (next session, after R6000 verify)

New file: ``engine/full_attn_layer_graph_pool.py``

```python
class FullAttnLayerGraphPool:
    """Per-layer full-attn graph slots with bucket-aligned reuse."""

    def __init__(self, runner: LynnIncrementalRunner, bucket: int = 256):
        self.runner = runner
        self.bucket = bucket
        self._slots: dict[int, FullAttentionLayerGraphSlot] = {}
        self._slot_bucket: dict[int, int] = {}

    def get(
        self,
        state: LynnInferenceState,
        h_seed: torch.Tensor,
        pos_tensor: torch.Tensor,
        layer_idx: int,
    ) -> FullAttentionLayerGraphSlot:
        cur_bucket = int(state.seq_len) // self.bucket
        slot = self._slots.get(layer_idx)
        if slot is None or self._slot_bucket.get(layer_idx) != cur_bucket:
            slot = self.runner._capture_full_attn_layer_graph_slot(
                state, h_seed, pos_tensor, layer_idx
            )
            self._slots[layer_idx] = slot
            self._slot_bucket[layer_idx] = cur_bucket
        return slot

    def invalidate(self) -> None:
        self._slots.clear()
        self._slot_bucket.clear()
```

Modification to ``engine/resident_runner.py`` (single function in
decode loop branch 3):

```python
elif linear_block_graphs is not None:
    for bi, block in enumerate(linear_block_graphs):
        block["input"].copy_(h)
        block["graph"].replay()
        h = block["output"]
        full_layer = bi * 4 + 3
        if full_attn_layer_graph_pool is not None:
            slot = full_attn_layer_graph_pool.get(state, h, pos_tensor, full_layer)
            slot.input.copy_(h)
            slot.graph.replay()
            h = slot.output
        else:
            h = self._decode_layer_fast(h, pos_tensor, state, full_layer)
```

A new env toggle ``LYNN_FULL_ATTN_LAYER_GRAPH_POOL=1`` gates the new
path. Default is ``0`` until the promotion gate passes.

## Risk register

| Risk | Mitigation |
|---|---|
| Bucket boundary causes P37 drift due to KV slice shape mismatch | Bench at multiple bucket sizes (128 / 256 / 512); pin bucket = 256 for first candidate |
| Graph replay aliases mutable state (e.g. RoPE cache table not pinned) | Pre-prewarm RoPE cache table during pool init via ``triton_kernels/full_attn_rope_cache.get_global_cache().prewarm(...)`` |
| Slot capture cost dominates if bucket too small (similar to existing whole-token slot) | Bucket ≥ 256 + per-layer scope (smaller than whole-token slot, so capture cost is ~10% of whole-token) |
| Per-request state copy overhead | Already paid by linear-block reuse pool, no new overhead |
| Stream A (DeepSeek) MoE work overlaps | Scope-locked: this spec touches ``engine/resident_runner.py`` (decode loop) + a new helper, never ``csrc/lynn_native/*`` or ``engine/moe_packed_nvfp4.py`` |

## Steps to ship (in order)

1. ✅ This spec doc landed (this commit).
2. Land a candidate env file
   ``scripts/qwen36_candidate_env_full_token_graph_slot_baseline.env``
   that enables the existing ``LYNN_FULL_TOKEN_GRAPH_SLOT=1`` (with
   ``LYNN_ROUTER_TOPK_SORTED=1`` precondition) — this lets Codex
   measure the **capture-per-token** baseline on R6000 after the 30-min
   hold. This number quantifies the ROI room for the per-layer reuse
   pool below.
3. Wait for the baseline gate (3 fields per Stream C wrapper).
4. Implement ``engine/full_attn_layer_graph_pool.py`` (scaffold).
5. Wire into ``resident_runner.py`` decode loop branch 3 behind
   ``LYNN_FULL_ATTN_LAYER_GRAPH_POOL=1``.
6. Land
   ``scripts/qwen36_candidate_env_full_attn_layer_graph_pool.env``.
7. Codex R6000 gate runs → DEFAULT / AMBER / closed decision via
   ``scripts/stream_b_promotion_report_card.py``.
8. If DEFAULT, flip safe-default in a separate commit.

## Not in scope

- Whole-token graph slot reuse — kept research-only. The per-layer
  pool is the smaller, more parity-safe path.
- MoE / linear-attn block graphs — already reuse via existing pool.
- Native CUDA / C++ rewrite — Phase A / kernel island work, lives in
  Stream A. This spec stays purely in Python + existing CUDA graph
  infrastructure.
- HTTP / scheduler / serving layer — host_gap is already 0.16 ms, not
  the bottleneck.
