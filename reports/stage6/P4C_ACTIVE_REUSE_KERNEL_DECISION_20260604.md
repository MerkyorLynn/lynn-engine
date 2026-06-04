# Stage 6 P4C - active-reuse kernel decision

Date: 2026-06-04

Verdict: **decision bank only; no new fused-kernel speed is banked here.**

P4B proved that the true out-only, no-`inter_scratch` ABI is reachable and can
be numerically exact, but the two first CUDA shapes also proved what must not be
done next:

- single CTA owns the whole `active[top_k,512]` tensor and reuses it correctly,
  but serializes the output path: **39.54 ms vs P4A 0.283 ms = 0.007x**;
- multi-CTA per-output-tile recompute writes the same correct BF16 output, but
  every output tile recomputes `active[top_k,512]`: **48.34 ms vs P4A 0.279 ms =
  0.0058x**.

Therefore the next kernel is not "make more CTAs." The next kernel must preserve
active reuse.

## Fixed Evidence

| Candidate | Evidence | Decision |
|---|---|---|
| P4A two-stage active scratch | `inter_scratch[top_k,512]`, caller-owned, packed NVFP4, fastest current native reference: **~0.28 ms** synthetic | Keep as lower bound and product fallback. Not P4B because it exposes `inter_scratch`. |
| P4B single-CTA reference | `rel_l2=0.0`, `max_abs=0.0`, **39.54 ms**, **0.007x** vs P4A | Correctness reference only. Closed as speed path. |
| P4B multi-CTA recompute | `rel_l2=0.0`, `max_abs=0.0`, **48.34 ms**, **0.0058x** vs P4A | Closed negative. Recomputing active per output tile is forbidden for the next speed candidate. |

## Constraint

The active tensor is small enough to reuse (`top_k=8`, `I=512`, BF16 = 8192
bytes per decode token), but ordinary CUDA shared memory is CTA-local. Once
output rows are split across CTAs, a CTA-local `active` tile is no longer shared
with peer CTAs. A naive output-row split therefore pays the gate/up dequant-GEMV
cost once per output tile.

This is the P4B structural trap. A candidate that merely parallelizes output
rows while recomputing gate/up active values cannot be promoted, even if it is
token-exact.

## Candidate Ladder

| ID | Shape | Active reuse | ABI | Promotion stance |
|---|---|---|---|---|
| C0 | Single CTA computes gate/up, then all output rows | Full reuse inside one CTA | P4B out-only | Already exact, too slow. Closed as speed path. |
| C1 | Multiple CTAs own output tiles and recompute active | None across CTAs | P4B out-only | Already exact, slower than C0. Closed negative. |
| C2 | Two-stage P4A/CUTLASS-style: compute active once into caller scratch, then down GEMV | Full reuse through caller scratch | P4C, not P4B | Most plausible immediate speed path; honest two-phase active-reuse candidate. |
| C3 | Persistent block/cluster kernel with shared active across output workers | Intended reuse inside one cooperative unit | P4B-like if no external scratch | Plausible only if cooperative-group/cluster shared-memory mechanics and launch constraints are proven on Spark. |
| C4 | CUTLASS/CuTe grouped GEMV kernel pair over packed NVFP4 | Full reuse by design, can later fuse epilogue | P4C first, P4B later | Preferred long-term route, especially for FP4-MMA hardware. |

## Next Implementation Target

The next executable target should be named separately from P4B:

```text
LYNN_NATIVE_ACTIVE_MOE_BACKEND=fused_zero_shadow_active_reuse_contract
```

This target may use caller-owned active scratch or a two-kernel active-reuse
layout. That makes it **P4C**, not the final P4B out-only single-kernel. The
name matters: it prevents a fast two-phase implementation from falsely closing
the harder out-only fused-kernel objective.

Minimum acceptable first P4C gate:

- reads packed NVFP4 gate/up/down weights directly;
- never uses active-expert BF16 resident shadows;
- computes `active[top_k,512]` once per decode token, not once per output tile;
- compares numerically against P4A and P4B references;
- reports byte counts for packed weights, scales, active scratch, and BF16
  shadow equivalent;
- reports speed against P4A synthetic reference and current `~44-45 TPS` RC
  stack;
- keeps `banked_default_promotion=false` until server and RC quality gates pass.

## Forbidden False Positives

The following are not bankable fused-kernel speed evidence:

- a microbench that is token-exact but slower than P4A;
- a multi-CTA output split that recomputes `active[top_k,512]` per tile;
- a two-stage implementation reported as P4B out-only single-kernel;
- a Python/Triton fallback routed through the native backend name;
- any result that omits byte-count, numeric parity, e2e TPS, or RC quality
  boundaries.

## Local Gate

```bash
python3 scripts/test_stage6_p4c_active_reuse_decision_static.py
```

This static gate does not prove speed. It prevents the repo from forgetting the
two measured anti-proofs and the active-reuse boundary before the next CUDA
candidate is written.
