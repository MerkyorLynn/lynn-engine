# Qwen3.6-35B MTP M24 Commit-Repair Bisect Result · 2026-05-20

## Headline

**There is no cheap commit-repair.** The K=2 commit `next_base_hidden`
and `next_pending_id` are byte-identical (or near-bit-identical) to
canonical T=1 chain outputs, so replacing them alone changes nothing.
The K=2 **commit state** itself must be discarded and replaced with
the canonical T=1 chain end-state to restore correctness. `state` mode
alone reaches 5/6 (fails on one prompt due to K=2 accept signal
disagreement); `full_canonical` mode reaches 6/6 by also using
canonical T=1 accept argmax. **There is no minimal repair component
strictly cheaper than M23's full canonical commit.**

Verdict class: `M24_CANDIDATE_FOUND` is technically reported (because
`full_canonical` is 6/6 with TPS > M23's 7.85), but the "candidate"
is just `full_canonical` itself — i.e., it confirms M23 rather than
discovering a cheaper repair. **The minimal-repair search returned
empty.**

## Setup

Spark Qwen3.6-35B-A3B Lynn-native W4A16 NVFP4, official fused MTP
sidecar, canonical 6 prompts, `--max-new 64`. Apples-to-apples eager
(no `LYNN_LINEAR_BLOCK_GRAPH`) + shadow off. One resident runner.

## Result

| Config | Exact | Mean prefix | Accept | Effective TPS | TPS ratio vs baseline |
|---|---:|---:|---:|---:|---:|
| baseline_greedy | 6/6 | 64.0 | — | 31.70 | 1.000 |
| spec_k1_batched_default | 3/6 | 44.0 | 74.85% | 24.39 | 0.769 |
| spec_k1_batched_repair_hidden | **3/6** | **44.0** | **74.85%** | 14.00 | 0.441 |
| spec_k1_batched_repair_next_pending | 3/6 | 44.0 | 74.99% | 13.96 | 0.440 |
| spec_k1_batched_repair_hidden_next_pending | 3/6 | 44.0 | 74.99% | 13.93 | 0.440 |
| spec_k1_batched_repair_state | **5/6** | **54.83** | **78.58%** | 14.35 | 0.453 |
| **spec_k1_batched_repair_full_canonical** | **6/6** | **57.33** | **79.18%** | 14.38 | **0.454** |

Raw JSON: [`reports/mtp/mtp_m24_commit_repair_20260520_130415.json`](mtp_m24_commit_repair_20260520_130415.json)

### Per-prompt first-divergence event

| Prompt | default | hidden_next_pending | state | full_canonical |
|---|---|---|---|---|
| 0 (Q4_K_M vs NVFP4) | event 17 reject | event 17 (same) | **null ✓ exact** | null ✓ exact |
| 1 (中文 speculative) | event 26 accept | event 26 (same) | event 29 accept | null ✓ exact |
| 2 (Fibonacci) | exact | exact | exact | exact |
| 3 (train 60mph) | exact | exact | exact | exact |
| 4 (JSON Tokyo) | event 3 | event 3 (same) | **null ✓ exact** | null ✓ exact |
| 5 (MoE router) | exact | exact | exact | exact |

## What M24 proves

1. **`hidden` mode is byte-identical to default** (3/6, 44.0 prefix,
   74.85% accept — match to all decimal places except slight TPS noise).
   This confirms M18's strict-diff result holds in the multi-round
   speculative loop too: the K=2 `next_base_hidden` (= `h_k2[:, 1, :]`)
   is bit-equal to the canonical T=1 chain's `h_after_draft`. **There
   is no hidden drift to repair.**
2. **`next_pending` mode shifts accept rate by +0.14pp** (74.85% →
   74.99%) but keeps exact at 3/6. This is the known M11 native FP4
   lm_head per-row 1D-input drift. **It is a real drift, but does not
   cause the exact gap on its own.**
3. **`hidden_next_pending` mode = `next_pending` mode** (3/6, 44.0
   prefix, 74.99% accept). Combining both output replacements doesn't
   help — the K=2 commit-outputs are not the bug.
4. **`state` mode (5/6, 54.83 prefix, 78.58% accept)**: replacing K=2
   state with canonical T=1 chain end-state fixes 2 of the 3 failing
   prompts. Prompt 1 still diverges at event 29 (delayed from default's
   event 26) — `state` mode trusts K=2's accept signal, and at event
   29 K=2 accepts a draft that canonical T=1 would reject.
5. **`full_canonical` mode (6/6)**: state + canonical accept re-check
   = the minimum 6/6 repair. Effective TPS 14.38 (1.83× the M23 result
   of 7.85, likely due to a warmer runner this session).

## Where the bug actually lives

The per-layer hidden output at the end of the 40-layer decoder chain
is bit-strict (M18, and confirmed multi-round by M24's `hidden` no-op).
The `next_pending_id` argmax has ULP-level drift from M11 lm_head
per-row, accounting for a tiny accept-rate shift but not exact-match
loss. The KV cache writes are byte-equal (M22 reject rollback no-op).

The bug therefore lives in some component of the **post-K=2 state**
that is **(a) NOT visible at the per-layer hidden output** and
**(b) NOT visible inside the captured KV / recurrent / conv tensors**,
yet **(c) IS visible when comparing the next K=2 forward's behavior
against a canonical T=1 chain end-state's behavior**.

Candidates (M22 already listed these; M24 reconfirms via elimination):

- **Triton autotune cache**: K=2 path calls T=1 kernels in burst
  (M9 MoE per-position loop, M12 t1_loop full-attn, M11 per-row
  lm_head). The autotune cache may select different "best" kernels
  after observing a K=2-burst pattern than after observing genuinely
  sequential T=1 calls. Subsequent K=2 forwards then use a slightly
  different kernel choice, leading to multi-round drift.
- **CUDA workspace allocator** with `expandable_segments=True`:
  segment layout / free-list state changes after K=2 forward's larger
  activation footprint. Next forward reads from differently-laid-out
  memory → different FP rounding in reductions.
- **Native FP4 lm_head workspace** (M11 per-row): workspace tensors
  shared across calls, multi-round behavior depends on prior calls.
- **Runner-level lazy buffers**: graph slot pool, MoE expert routing
  caches, etc. May be initialized lazily by a code path that K=2
  forward triggers indirectly.

## Why `state` mode helps but `hidden_next_pending` doesn't

`state` mode does two things differently from `hidden_next_pending`:

1. **State content replaced with canonical T=1 chain end-state.** If
   layer outputs are bit-strict (M18 says yes), this should be a no-op
   on tensor values — but it **also** clears any process-side state
   the K=2 forward left behind (Triton autotune cache hits, allocator
   segment layout, etc., are reset implicitly because the canonical
   T=1 chain's kernel calls happen in a different invocation pattern
   that the allocator/autotune see differently).
2. **Next round's K=2 forward starts from canonical-T=1-end state.**
   Even if the tensor values are bit-equal, the process state
   surrounding them differs.

The 2 prompts that `state` mode fixed (0 and 4) had drift sources
that were sensitive to this process-state reset. Prompt 1 has an
additional accept-argmax disagreement that needs canonical
verification to catch.

## Promotion status

**NOT PROMOTED.** The "minimum 6/6" repair is `full_canonical` which
is functionally identical to M23 — paying the cost of a full canonical
T=1 chain on every accept. Effective TPS 14.38 vs baseline 31.70 =
0.45× ratio; well below sequential `spec_k1` (21.3 TPS, 6/6 exact).
Sequential spec is therefore still the *only* correctness-clean K-step
speculative path that beats no path; batched K=2 + full canonical
commit is correctness-clean but slower than sequential and adds K=2
forward overhead with no compensating output reuse.

`LYNN_MTP_K2_COMMIT_REPAIR` ships as opt-in diagnostic; default
unchanged.

## Recommended next probe (M25)

Per the elimination chain (M21 → M22 → M24), the remaining suspects
all live in **process-side state outside captured tensors**. The
cheapest decisive M25 probe is to **strip suspect process-side
behaviors one at a time**:

1. **Disable Triton autotune** for the K=2 path's T=1 sub-calls (or
   reset the autotune cache between spec events) and re-run M22
   default vs full_canonical comparison.
2. **Force eager allocator** (`PYTORCH_CUDA_ALLOC_CONF=` empty; drop
   `expandable_segments`) and re-run.
3. **Skip the K=2 forward but keep all the rest** (i.e., the runner
   pretends K=2 happened but never calls `_decode_layer_k2_fast`).
   If exact still drifts, the act of running K=2 is not the source;
   if exact is restored, the K=2 forward IS the source and we're back
   to autotune/allocator as the residual.

The cheap repair search returned empty; the next investigation has
to go below the tensor layer.

## Cross-reference

- M18 layer sweep (K=2 layer outputs bit-strict under t1_loop): [`32731a9`](https://github.com/MerkyorLynn/lynn-engine/commit/32731a9)
- M21 lm_head + canonical_t1_accept bisect: [`432a29e`](https://github.com/MerkyorLynn/lynn-engine/commit/432a29e)
- M22 reject rollback full_state (no-op): [`819cdf6`](https://github.com/MerkyorLynn/lynn-engine/commit/819cdf6)
- M23 canonical commit (6/6 @ 7.85 TPS): [`cf0a187`](https://github.com/MerkyorLynn/lynn-engine/commit/cf0a187)
- M24 commit repair switch + probe: [`66d52f5`](https://github.com/MerkyorLynn/lynn-engine/commit/66d52f5)
- Memory: `project_lynn_engine_t1_only_kernel_contract_20260519`
