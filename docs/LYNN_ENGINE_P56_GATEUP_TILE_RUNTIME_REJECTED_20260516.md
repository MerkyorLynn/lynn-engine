# Lynn Engine P56: gate/up tile-inter runtime gate rejected

Date: 2026-05-16
Branch: `codex/p16-r6000-155-tps`

## Why P56

P55 found a promising local gate/up kernel shape:

```text
tile_inter=2
  local gate/up speedup: 1.09-1.14x vs Triton
  local diff: max_abs=0
```

P56 wires that shape as an opt-in runtime backend:

```bash
export LYNN_NATIVE_GATEUP_BACKEND=cuda_tile_inter
export LYNN_NATIVE_GATEUP_TILE_INTER=2
```

Only gate/up is replaced. Down remains Triton, shared expert remains BF16, and
the active-MoE backend remains `triton`. This isolates the P55 change.

## Gate

```text
benchmarks/p37_moe_config_generate_gate.py
reports/p16_155/p56_gateup_tile_inter_generate_gate.json
```

Candidate must:

- match baseline greedy token IDs;
- improve or at least not regress decode TPS;
- avoid no-think / `!` loop failure patterns.

## Result

Performance signal is strong:

```text
baseline median:   ~99.15 tok/s
candidate median:  114.43 tok/s
median speedup:    1.154x
```

But quality/parity fails:

```text
new_ids_all_match: false
prompt_001: "，!!!!!!!!!!!!!!!!..."
prompt_002: "\n\n!!!!!!!!!!!!..."
```

The failure pattern is the same class as earlier unsafe native/graph shortcuts:
a local exact-ish kernel-level result can still perturb full decode enough to
fall into a degenerate greedy loop.

## Decision

Do **not** promote `LYNN_NATIVE_GATEUP_BACKEND=cuda_tile_inter`.

P56 is still valuable because it confirms the tile shape has real runtime
speed potential, but the scalar tile-inter implementation is not a safe
production backend.

## Next path

Keep the P55/P56 lesson, but move the implementation target:

```text
use tile_inter=2 as a shape hint
do not reuse scalar accumulation as production math
build the real grouped per-16 native-FP4 active expert kernel
validate with full-generate gates before promotion
```

In other words, P56 is a speed-signal / safety-rejection milestone. It narrows
the P57/P58 kernel design, but it does not change the default runtime.
