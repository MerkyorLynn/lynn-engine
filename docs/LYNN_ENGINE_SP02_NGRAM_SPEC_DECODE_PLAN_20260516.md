# Lynn Engine SP-02 · N-gram Speculative Decoding for Spark sm_121

Date: 2026-05-16
Branch: `spark/sm121-port` (Spark-only)
Status: **PLAN** — implementation gated on SP-01 microbench result

## Goal

Close the +42% peak TPS gap to SGLang FP8+MTP on Spark sm_121 (62.51 peak)
without training new draft heads. The SGLang peak comes from its built-in
NEXTN MTP head; Lynn 27B has no internal MTP head (distilled), so we use
**prompt n-gram lookahead** (a.k.a. REST-style speculative decoding) which
needs:

- zero model retraining
- zero new kernels
- only changes to the decode orchestration

```text
Baseline  Lynn 27B NVFP4 (post-SP-01)     ~45-50 mean / ~50 peak (projected)
Baseline  SGLang FP8+MTP 35B              49.97 mean / 62.51 peak
Target    SP-01 + SP-02 combined          55+ mean / 65+ peak  ← beats SGLang
```

Expected per-domain gain envelope from n-gram lookahead literature
(Yang et al. 2024, lookahead-decoding) on similar-size MoE models:

| Domain                  | Acceptance | Expected TPS multiplier |
|-------------------------|-----------:|-------------------------:|
| Code generation         | 50-70%     | 1.8-2.5×                 |
| Tool-call / JSON output | 40-60%     | 1.5-2.0×                 |
| Chinese long-form chat  | 20-35%     | 1.2-1.5×                 |
| Repetitive instruct     | 60-80%     | 2.0-3.0×                 |

If we hit even the conservative 1.3× on top of SP-01-tuned 45 mean:
**45 × 1.3 = 58.5 mean** — clearly beats SGLang 49.97 mean.

## Why N-gram, Not Draft Model / EAGLE / MEDUSA

- **Draft model** (e.g. a small Lynn variant): needs training + tuning, extra
  weight load, doubles memory pressure on the already-tight 73G/97G Spark
  budget.
- **EAGLE / MEDUSA**: need extra trained heads. Lynn 27B distilled doesn't
  have them; bootstrapping costs days on A100.
- **N-gram lookahead**: pure decode-time orchestration. Uses prompt + already
  generated tokens as the candidate corpus. Ships in days, not weeks.

The trade-off: n-gram acceptance is lower than a trained draft model
(typically 25-50% vs EAGLE's 60-80%), but it costs nothing to deploy and
acceptance is materially higher on structured / repetitive outputs that
dominate tool-call and code workloads.

## How It Plugs Into Existing Lynn Engine Plumbing

Critical observation: `engine/resident_runner.py` ALREADY has the state-
snapshot / restore primitives that spec decoding needs. They were built for
P14 state-refresh probes:

- `_snapshot_state(state)` — capture KV cache + linear-attn recurrent state
  at any point
- `_restore_state(state, snap)` — roll back to a snapshot
- `_snapshot_linear_state` / `_restore_linear_state` — linear-only fast path
- `_copy_linear_state(dst, src)` — fast in-place state copy

That means SP-02 does NOT have to redesign the state management layer. The
new work is:

1. **N-gram corpus builder** — a small trie over already-emitted tokens
   plus the prompt tokens. Updated each decode step.
2. **Draft proposer** — given current token and trie, return a candidate
   continuation of length `L` (typically `L = 4-8`).
3. **N-token verifier** — forward `L` candidate tokens through the existing
   prefill / multi-token forward path. Compare each predicted next-token
   distribution's greedy argmax against the candidate; find first mismatch
   index `M`.
4. **Commit / rollback** — accept `M` candidate tokens into the output
   stream, advance state by `M+1` (the model's own argmax at position M).
   Snapshot taken before verification is used to roll back the part of state
   advance beyond `M+1`.

## Phase Plan

### SP-02-A: trie + proposer prototype (Python only, no model)

- Build `engine/spec_ngram_trie.py` with `NgramTrie` class:
  - `update(token: int)` — append token to corpus, update trie nodes
  - `propose(suffix: list[int], max_len: int) -> list[int]` — find best
    continuation matching last `n` tokens
- Unit test on synthetic token streams to validate trie correctness +
  acceptance rate on V8 / V9 / tool-call prompt outputs (replayed).

Gate: trie can propose >= 4-token continuations with >= 30% mean acceptance
on V8 stage4 output corpus.

### SP-02-B: multi-token verify path

- Add `LynnIncrementalRunner.verify_draft(draft_tokens: torch.Tensor) -> int`
  - Takes draft of length L
  - Forwards through model (uses existing prefill-style path if available,
    else a new mini-prefill graph slot for batch=L sequences)
  - Returns greedy-match prefix length M
- Reuses `_snapshot_state` / `_restore_state` to allow rollback
- Microbench: latency of verify(L=4) vs single-step decode × L

Gate: verify(L=4) latency ≤ 2.0× single-step decode (so even at 50%
acceptance we break even or win).

### SP-02-C: end-to-end loop integration

- Modify the generation loop in `server/openai_http.py` (or wherever the
  per-token loop lives) to:
  1. Build / update trie at each step
  2. Propose draft (skip if proposer returns empty)
  3. Verify
  4. Emit accepted tokens via SSE
  5. Continue from the post-match position
- Env gate: `LYNN_SP_NGRAM_SPEC=1` opt-in, default off

Gate: 6-prompt smoke runs to completion, no `<think>` loop, token count and
final-text length consistent with non-spec path.

### SP-02-D: tune lookahead parameters

- Sweep `(n_gram_order, max_draft_len, min_count_threshold)`:
  - n_gram_order ∈ {2, 3, 4}
  - max_draft_len ∈ {2, 4, 6, 8}
  - min_count_threshold ∈ {1, 2, 3} (how many times an n-gram must have
    been seen before being a draft candidate)
- Bench on:
  - V8 (Chinese chat / math)
  - V9 (strict / multi-domain)
  - tool-call 15-stage1 (JSON repetition)
  - 50-prompt coding (high acceptance regime)

Gate: pick (order, len, thresh) tuple that maximizes mean TPS across all
four workloads while keeping V8 ≥ 70% and tool-call ≥ 75%.

### SP-02-E: integrate with SP-01 + measure

- Re-run the 20-prompt mixed-stability bench with **both** env vars on:
  - `LYNN_SP_TRITON_AUTOTUNE=1`
  - `LYNN_SP_NGRAM_SPEC=1`
- Compare against:
  - Lynn 27B baseline (42.85 mean / 44 peak)
  - SGLang FP8+MTP (49.97 mean / 62.51 peak)
- Goal: ≥ 55 mean / ≥ 65 peak

## Risk Surface

1. **Verify path may not exist as multi-token forward.** Lynn engine was
   built for prefill-then-single-token-decode. If the current prefill path
   cannot be invoked mid-decode without re-prefilling the full sequence,
   we need a new short-prefill mode. This adds engineering scope; budgeted
   in SP-02-B.

2. **Linear attention state advancement under speculation.** Linear-attn
   layers carry recurrent state. Speculating L tokens advances the state by
   L; on rejection we need to roll back to position M+1. State snapshot
   exists per layer but currently snapshots the whole layer; we may need a
   per-token rollback for linear-attn correctness. Risk mitigation: use
   `_snapshot_linear_state` BEFORE each verification, restore + replay M+1
   tokens on partial accept.

3. **CUDA graph compatibility.** Full-token graph slot is batch=1 only. For
   batch=L verify, we either (a) capture a new graph slot at L=4 / L=8 or
   (b) fall back to eager forward for verify. (b) is fine for SP-02-A/B;
   (a) is an SP-02-E polish step if the eager verify is the bottleneck.

4. **Acceptance below break-even.** If real V8 / Chinese chat acceptance
   stays below 25% with L=4, the per-step overhead may erase the win. The
   SP-02-D sweep is designed to catch this; we'll keep L=4 even with low
   acceptance because Chinese chat acceptance is still > 20% in published
   results.

## Promotion Gate

SP-02 stays opt-in until:

1. **20-prompt mixed-stability** mean ≥ 50 TPS (beats SGLang 49.97 mean)
2. **Single-stream peak** ≥ 55 TPS (closes most of SGLang's 62.51 peak)
3. **Quality preserved**:
   - V8 ≥ 70%
   - V9 ≥ 30% (currently 38.33%, so up to -8pp headroom)
   - tool-call 15-stage1 ≥ 75%
   - 6-prompt coherence pass
4. **No regression in long-ctx** (16k smoke still passes)

Combined with SP-01 promotion (kernel autotune passed all gates), this
becomes the new Spark sm_121 default.

## What Comes After

If SP-01 + SP-02 still fall short of 65+ peak on Spark, escalate to:

- **SP-03**: custom CUDA persistent-CTA grouped FP4 GEMM (similar to Codex's
  P47-C in spirit but Spark-tuned). This is multi-week scope.
- **SP-04**: hybrid draft model — train a tiny 1B Lynn draft from V Pro
  Distill checkpoints. Needs A100 time.

But the realistic expectation is SP-01 + SP-02 closes the gap. SGLang's
62.51 peak is MTP-driven; n-gram lookahead is its no-training analogue.

## Discipline

Same as SP-01:

- All work on `spark/sm121-port`
- All metrics labeled Spark sm_121 (not comparable to R6000)
- R6000 main line `codex/p16-r6000-155-tps` is Codex's; do not touch
- All Spec implementations are opt-in via env vars until promotion gates pass
