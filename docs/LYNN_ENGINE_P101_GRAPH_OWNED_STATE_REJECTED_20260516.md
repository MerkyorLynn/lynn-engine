# Lynn Engine P101: graph-owned authoritative state rejected

## Result

P101 retested the P14-C graph-owned authoritative decode-state idea on a clean
P100 worktree.

The result is a hard reject:

```text
reports/p101/p101_graph_owned_authoritative_64.json

same_ids:          false
pass:              false
min_cosine:        0.6536
min_top10_overlap: 0
```

The failure is not a P100 regression. Earlier P14-C/P35 reports were also
`pass=false`; their replay TPS looked attractive, but the generated sequence
drifted.

## What This Clarifies

There are three different graph contracts, and they should not be mixed:

| Contract | Meaning | Status |
|---|---|---|
| Current-position capture/replay | Capture exactly the real current state and replay once | strict in P13-style gates |
| Fixed-position refresh/replay/commit | Copy real state into graph state, replay one slot, copy back | top-1 can match, but logits are not exact in P14-B |
| Graph-owned authoritative sequence | Copy prefill state once and let graph-owned state advance across tokens | **fails** in P14-C/P35/P101 |

P101 closes the third route. It is not production-safe.

## Why It Matters

The tempting number was the replay-only speed:

```text
~75-85 tok/s replay equivalent
```

But that number is not a shippable decode path because the graph-owned state
sequence diverges. It should not be used in README tables as a production
candidate unless it is clearly labeled as a failed replay-only probe.

## Decision

Do **not** promote graph-owned authoritative decode state.

Keep only these safe graph directions:

1. reusable linear-block graph path, which already preserves greedy behavior;
2. current-position full-token graph slots for diagnostics;
3. future graph/server work that preserves BF16 activation semantics and proves
   strict greedy-id parity end-to-end.

## Next Route

R6000 runtime work should move away from graph-owned mutable-state tricks and
toward production-safe consolidation:

- reduce Python/server dispatch without changing the numerical path;
- keep the existing linear-block graph path as the safe reusable graph primitive;
- reserve W4A4 / activation-quant-aware changes for the A100 MTP/retrain and
  re-quant artifact line.
