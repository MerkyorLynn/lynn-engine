# Stream B owned-modules roadmap

Date: 2026-05-18

Stream B in `docs/QWEN36_W4A16_KERNEL_REFACTOR_PLAN_20260518.md` calls for
moving the full-attention + linear-core hot path off ad-hoc state and into a
small set of explicitly-owned modules. This file tracks that list so the
serving banner, dev tooling, and gate scripts converge on one ownership model.

## Promotion bar (2026-05-18 hand-off, fixed)

Every owned-module deliverable must report **all three** at promotion time.
No headline is allowed to claim a win on average / microbench TPS alone.

| Gate | DEFAULT promote | AMBER promote |
|---|---|---|
| P37 exact-greedy | 3/3 (required) | may drift; document |
| hard structured | 40/40 | 70/70 (stricter; opt-in only) |
| P25 512-token decode TPS | ≥ 108 TPS | ≥ 118 TPS |

122 TPS is only a target after Stream A (native MoE island) + Stream B
(attention + linear-core) candidates each pass DEFAULT individually and
are then combined through the full gate.

Stream B sprint target this cycle is **118 TPS** (AMBER) at the same exact
P37/structured bar. P26 phase profile must show ≥ 5% reduction in either
full-attention or linear-block total ms/token vs the safe-default baseline.

## Module ledger

### 1. ✅ `triton_kernels/full_attn_rope_cache.py` — landed 2026-05-18

`FullAttnRoPECache` class with `prewarm` / `lookup` / `reset` / `info`. Replaces
the old module-level `_ROPE_TABLE_CACHE` dict + `_build_rope_cos_sin_cached`
function in `engine/incremental_decode.py`. Backward-compatible private-name
re-export so the two call sites in `incremental_decode` and the p123 probe do
not change.

Status:
- Cache toggle `LYNN_FULL_ATTN_ROPE_CACHE` default remains `"0"` until R6000
  promotion gate proves the path passes DEFAULT bars at default-on.
- Candidate env file: `scripts/qwen36_candidate_env_rope_cache_default.env`
- Run command: see the file header.

### 2. Pending — `triton_kernels/full_attn_mask_cache.py`

Today's SDPA decode path uses `is_causal=True` (no explicit mask materialized),
so this module is **not urgent**. It becomes useful when:

- a candidate switches from `F.scaled_dot_product_attention` to a manual
  attention kernel that needs the mask buffer pre-allocated; or
- prefix-cache / partial-prefix paths need a buffered mask owner.

Until then, scaffold only. Do not allocate the mask buffer in the safe route
just because the module exists; that would regress steady-state mem with no
TPS reward.

Acceptance, when promoted to land:
- module owns the mask buffer for the worst-case `max_seq` config
- `prewarm(max_seq, head_dim, dtype, device)` returns the buffer ready
- `is_causal=True` path stays the default; explicit mask is a candidate

### 3. Pending — `triton_kernels/full_attn_decode_workspace.py`

Workspace owner for the decode step's reusable scratch tensors:

- `pos_tensor` (1×1 long) — currently re-allocated each step unless CUDA-graph
  captured with a pinned buffer.
- `new_token_tensor` (1×1 long) — same shape.
- `qkv_split_views` — views into the projection output that the call site
  re-derives each step.

The graph-capture path in `engine/resident_runner.py` already pins these
implicitly. The owned-workspace module makes them visible + bench-able, and
removes the per-call `torch.tensor([[…]])` allocation in the non-graph path.

Acceptance criteria:
- P37 3/3 exact
- P25 512 ≥ 108 (DEFAULT) or ≥ 118 (AMBER)
- `host_gap_ms` in P26 drops vs the safe-default baseline (currently
  0.16 ms/token, so the savings are small — accept ≥10% reduction inside the
  workspace path, NOT a wall-TPS regression)

This is intentionally narrow — host-gap is already small; the win here is
predictability + CUDA-graph friendliness, not raw TPS.

### 4. Pending — linear-core boundary candidates

Per p124's `target_deltas_ms` block, the four production segments and their
plan-doc targets are:

| segment | target ms/layer | notes |
|---|---:|---|
| `fused_inproj_native_fp4` | 0.077 | native FP4 path; do not regress |
| `recurrent_fused_prepare` | 0.036 | with/without GQA recurrent variant |
| `conv_update_decode` | 0.030 | conv1d update; in-place candidate exists |
| `gated_rmsnorm_decode` | 0.020 | Triton kernel route is opt-in |

Stream B boundary candidates land as `scripts/qwen36_candidate_env_<lc_*>.env`
files. Each must report all three gates (P37 / structured / P25 @ 512); none
goes to default on microbench latency alone.

## Negative list (do not retry, per plan doc)

- naive QKV row concat — only 1.013× and failed P37 0/3
- manual GQA attention path on the safe default — opt-in only via
  `LYNN_FULL_ATTN_DECODE_BACKEND=manual_gqa`, never the default

## Gate discipline reminder

Every Stream B status update we publish must read like:

> Candidate **X**: P37 exact = `3/3` / `2/3` / `0/3` ; hard structured =
> `40/40` / `38/40` ; P25 512-token decode TPS = `NN.NN` (safe default
> 107). Decision: DEFAULT / AMBER / closed.

If a status update has only "microbench `NN`% faster" without those three
fields, it is a research artifact, not a promotable candidate.
