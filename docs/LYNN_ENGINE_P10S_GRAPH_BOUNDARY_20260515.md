# Lynn Engine P10-S Graph Boundary Notes (2026-05-15)

This note records the R6000 graph-path findings after the 27B NVFP4 step5000
model reached the 100 TPS class in benchmark gates.

## Ground Truth

Model:

`/root/autodl-tmp/models/lynn-27b-variable-recovery-step5000-nvfp4-final`

Runtime flags:

```bash
LYNN_MOE_IMPL=packed_nvfp4
LYNN_LINEAR_ATTN_RECURRENT_BACKEND=triton_fused_prepare
LYNN_LINEAR_ATTN_RECURRENT_INPLACE=1
LYNN_LINEAR_STATE_UPDATE=inplace
LYNN_QK_NORM_ROPE_BACKEND=triton_pair
LYNN_RMSNORM_GATED_BACKEND=triton
LYNN_LINEAR_ATTN_INPROJ_FUSED_NATIVE_FP4=1
LYNN_NATIVE_FP4_LM_HEAD=1
```

Validated numbers on RTX PRO 6000 Blackwell:

| Gate | Result |
|---|---:|
| Full 40-layer graph replay-only | 107.15 tok/s |
| Full decode + final norm + native FP4 lm_head graph | 103.49 tok/s |
| 4-layer hybrid graph slots, strict single-token parity | 93.49 tok/s |
| Current OpenAI server stable decode path | ~88-89 tok/s |

The 100 TPS class path exists. The remaining work is productionizing graph
reuse without losing greedy parity across many decode tokens.

## Important Trap

When chaining commands like:

```bash
env LYNN_MOE_IMPL=packed_nvfp4 python bench_a.py && python bench_b.py
```

only `bench_a.py` receives the environment override. `bench_b.py` silently
falls back to the default MoE path. In this case it hit:

```text
torch.unique(...).tolist()
CUDA error: operation not permitted when stream is capturing
```

That failure was not a packed NVFP4 failure. Always wrap each benchmark command
with the full env or export the variables first.

## What Is Safe Today

### P9-O / 4-layer hybrid graph slots

The 4-layer graph slot covers:

```text
linear_attention, linear_attention, linear_attention, full_attention
```

It is strict for a single fixed decode position:

```text
max_abs = 0.0
greedy_pass = true
graph_tps = 93.49
```

This is a safe graph unit, but it captures a fixed `cached_seq_len`. It should
not be reused across arbitrary future tokens unless the full-attention sequence
length discipline is solved.

### Full graph replay-only

The 40-layer graph replay-only gate reaches 107 TPS, but it is a benchmark
upper bound. It is not yet a production server path because the graph-state
trajectory must remain greedy-identical over long generations.

## What Is Not Safe Yet

### Sequential full-token graph family

`p9k_sequential_capture_graph_family_greedy.py` with native FP4 lm_head:

```text
avg_graph_replay_tps = 93.66
greedy_pass = false
first drift appears after the early stable window
```

The first several tokens preserve top-1 parity, then graph and eager paths
diverge. This means the plumbing is close, but cross-token graph-state drift is
still real.

### Windowed graph family

`p9i_windowed_graph_family_greedy.py` is diagnostic only. Its current window
capture implementation captures multiple future positions from the same window
base state rather than sequentially advancing state inside the window. Do not
treat it as a production route.

## Next Engineering Target

To move stable serving from ~89 TPS toward 100+ TPS:

1. Keep the current OpenAI server default on the stable path.
2. Add an opt-in graph gate, not a default serving path, for any new full-token
   graph work.
3. Investigate full-attention decode graphability:
   - Can `cached_seq_len` be represented with tensor state or fixed slot family?
   - Can KV write/read slices be made graph-safe per position without Python
     integer specialization?
   - Can we cheaply refresh graph state at bounded windows without falling back
     to eager decode as the source of truth?
4. Only promote a graph path to serving after:
   - multi-prompt greedy parity,
   - long generation parity,
   - tool-call prompt parity,
   - stable memory under repeated requests.

## Current Decision

Do not force full-token CUDA graph into production yet. The correct near-term
path is:

```text
stable serving 88-89 TPS
  -> strict 4-layer graph slots as guarded opt-in
  -> graph-state discipline fix
  -> full-token graph serving
  -> 100+ TPS production path
```

## P10-T Follow-Up: Single-Position Graph Is Safe

After P10-S, we reran `p9j_single_position_graph_after_prefix.py` with the
same native NVFP4 runtime env at increasingly later positions:

| Prefix tokens before capture | Position | Graph TPS | Diff |
|---:|---:|---:|---|
| 8 | 15 | 94.19 | max_abs 0.0, top1 match |
| 16 | 23 | 94.11 | max_abs 0.0, top1 match |
| 32 | 39 | 94.51 | max_abs 0.0, top1 match |

This resolves the ambiguity from the windowed graph-family failures:

- Capturing a **single** full-token graph at the current real state is strict.
- Capturing a whole future window from one base state is not strict.
- Therefore the production design should be a lazy per-position/per-state graph
  slot cache, not an upfront future-window graph family.

The next implementation target is an opt-in server path roughly shaped as:

```text
prefill real request state
for each decode step:
  if graph slot for current position/state-shape exists:
    replay
  else:
    capture exactly this position from current real state, replay once, cache
```

This preserves correctness first. Capture amortization can be optimized later
via common-position warmup, short-window recapture, or prompt-prefix reuse.
