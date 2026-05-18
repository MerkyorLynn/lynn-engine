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

### 2. ✅ `triton_kernels/full_attn_mask_cache.py` — landed 2026-05-18 (scaffold)

`FullAttnMaskCache` class with `prewarm` / `lookup` / `reset` / `info`, plus
module-level singleton + `prewarm_global_mask` free function. Owns the
`[max_seq, max_seq]` causal-mask buffer for future explicit-mask candidates
(manual GQA path, prefix-cache batched decode, sliding window).

Status:
- **Scaffold only** — read by `benchmarks/p126_workspace_mask_cache_probe.py`,
  not wired into the safe-default `decode_full_attn` path (which keeps
  `is_causal=True` for SDPA).
- Wiring waits on a candidate kernel that actually needs an explicit mask
  buffer + R6000 promotion gate proving P37 3/3 + structured 40/40 + P25
  512 ≥ 108 TPS DEFAULT bar.
- Cache env: `LYNN_FULL_ATTN_MASK_CACHE_MAX_SEQ` (default 65536) — only
  consulted if a future `LYNN_FULL_ATTN_MASK_CACHE` toggle is added; the
  current default route does not look at the cache.

### 3. ✅ `triton_kernels/full_attn_decode_workspace.py` — landed 2026-05-18 (scaffold)

`FullAttnDecodeWorkspace` class with `prewarm` / `get_pos_tensor` /
`get_new_token_tensor` / `set_position` / `set_token` / `reset` / `info`,
plus module-level singleton `get_global_workspace` + `reset_global_workspace`.

Workspace owner for the decode step's reusable scratch tensors:

- `pos_tensor` (1×1 long) — currently re-allocated each step in the
  non-graph fallback paths (`engine/incremental_decode.py:348` and
  `engine/resident_runner.py:739`/`:1093`).
- `new_token_tensor` (1×1 long) — same shape; serving loop at
  `engine/resident_runner.py:1263-1264` already pre-allocates + reuses
  via `.fill_()`, so wiring is incremental (consolidate ownership only).

The graph-capture path in `engine/resident_runner.py` already pins these
implicitly. The owned-workspace module makes them visible + bench-able,
and removes the per-call `torch.tensor([[…]])` allocation in the
non-graph fallback paths.

Status:
- **Scaffold only** — read by `benchmarks/p126_workspace_mask_cache_probe.py`,
  not wired into `incremental_decode` or `resident_runner` this commit.
- Wiring is a separate commit that must pass DEFAULT promote gate first.

Acceptance criteria when wired:
- P37 3/3 exact
- P25 512 ≥ 108 (DEFAULT) or ≥ 118 (AMBER)
- `host_gap_ms` in P26 drops vs the safe-default baseline (currently
  0.16 ms/token, so the savings are small — accept ≥10% reduction inside
  the workspace path, NOT a wall-TPS regression)

This is intentionally narrow — host-gap is already small; the win here
is predictability + CUDA-graph friendliness, not raw TPS.

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

## Promotion-discipline tooling (2026-05-18, landed)

Two scripts encode the promotion bar so every candidate result carries the
three required gates together:

- `scripts/stream_b_promotion_report_card.py`: ingests the Codex Stream C
  wrapper JSON (from `scripts/r6000_qwen36_candidate_promotion_gate.sh`)
  and emits a deterministic markdown decision card. Decision codes:
  `DEFAULT_promote` / `AMBER_promote` / `AMBER_only` / `closed` /
  **`research_artifact_only`** (used when any one of the three required
  gates is missing — no microbench-only number can promote).
- `scripts/stream_b_candidate_sweep.sh`: walks `scripts/qwen36_candidate_env_*.env`,
  prints each as a sweep entry with the recommended wrapper invocation,
  and optionally aggregates finished gate JSON outputs (DEFAULT / AMBER /
  closed / research counts).

Smoke-tested 2026-05-18 with three synthetic gate JSONs (clean DEFAULT,
P37-drift AMBER, microbench-only research artifact). The report card
correctly refuses to issue a promote decision when any required gate
field is missing.

Sample card output (DEFAULT scenario):

```
# Candidate: `rope_cache_default` — promotion decision

| Gate | Result | DEFAULT bar | AMBER bar |
|---|---|---|---|
| P37 exact-greedy | **3/3 ✓** | 3/3 required | drift OK |
| Hard structured | **100.0%** | 100% (40/40) | 100% (70/70) |
| P25 512 decode TPS | **113.50 TPS** (+6.07% vs safe) | ≥ 108 & ≥ safe + 1% | ≥ 118 & ≥ safe + 5% |

## Decision: 🟢 DEFAULT_promote
```

## Candidate env files registered for the sweep

| File | Expected P37 | Expected structured | Decision class |
|---|---|---|---|
| `qwen36_candidate_env_rope_cache_default.env` | GREEN (byte-equivalent) | 40/40 | DEFAULT-eligible |
| `qwen36_candidate_env_amber_sharedgate_convinplace.env` (Codex) | RED (known) | 40/40 (verified 2026-05-18, P25 114.04) | AMBER-only |
| `qwen36_candidate_env_rope_cache_plus_amber.env` | RED (inherits amber) | 40/40 (expected) | AMBER-only — does rope stacking lift P25 above 118? |

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
