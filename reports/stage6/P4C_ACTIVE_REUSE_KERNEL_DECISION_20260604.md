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
| P4C runtime bridge | Spark real-runner path `PASS_P4C_ACTIVE_REUSE_RUNTIME_BRIDGE`; call delta=1, active shadows removed, candidate vs Triton `rel_l2=0.0/max_abs=0.0` | Banks route/numeric evidence only. Fused speed/default promotion remain closed. |
| P4C active-reuse speed baseline | Spark microbench `PASS_P4C_ACTIVE_REUSE_SPEED_BASELINE_RECORDED`; P4A **271.10 us**, P4C **271.06 us**, **1.00013x**, output/scratch exact | Banks baseline only. Confirms the P4C ABI adds no measurable overhead before replacing the symbol body. |
| P4C component profile | Spark diagnostic `PASS_P4C_COMPONENT_PROFILE_RECORDED`; full **233.39 us**, gate/up **151.67 us** (**63.8%**), down **86.21 us** (**36.2%**), all exact | Banks diagnostic only. First speed candidate should target gate/up. |
| P4C gate/up launch-shape sweep | Spark diagnostic `PASS_P4C_GATEUP_SHAPE_SWEEP_RECORDED`; baseline gate/up **151.69 us**, best shape `tile_inter=2, threads=128` **91.41 us**, **1.659x**, numeric ok | Banks actionable diagnostic only. Next low-risk candidate is replacing the P4C gate/up half with this launch shape, then rerunning P4C speed baseline. |
| P4C gate/up shape full-path candidate | Spark microbench `PASS_P4C_GATEUP_SHAPE_CANDIDATE_RECORDED`; current P4C `tile_inter=8` **267.32 us**, candidate `tile_inter=2` **185.09 us**, **1.444x**, output/scratch exact | Banks opt-in speed candidate only. Next gate is resident-runner/server/RC with `LYNN_NATIVE_GATEUP_TILE_INTER=2`; default promotion remains closed. |
| P4C tile=2 resident-runner preflight | Spark real-runner path `PASS_P4C_ACTIVE_REUSE_RUNTIME_BRIDGE`; `LYNN_NATIVE_GATEUP_TILE_INTER=2`, native call delta=1, recorded tile=2, candidate vs Triton `rel_l2=0.0/max_abs=0.0` | Banks opt-in route/numeric evidence on a real model layer. Next gate is server/RC quality; default promotion remains closed. |
| P4C tile=2 OpenAI server smoke | Spark service path `PASS_P4C_TILE2_SERVER_SMOKE`; P4C native call delta **240**, **40** layers called, completion text-exact **2/2**, released shadows stay released, reload observed **23.25s** | Banks opt-in server smoke only. RC quality, sustained server TPS, fused speed, and default promotion remain closed. |

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
python3 scripts/test_stage6_p4c_runtime_bridge_tools.py
python3 scripts/test_stage6_p4c_active_reuse_microbench_tools.py
python3 scripts/test_stage6_p4c_component_profile_tools.py
python3 scripts/test_stage6_p4c_gateup_shape_sweep_tools.py
python3 scripts/test_stage6_p4c_gateup_shape_candidate_tools.py
python3 scripts/test_stage6_p4c_tile2_server_smoke_tools.py
```

This static gate does not prove speed. It prevents the repo from forgetting the
two measured anti-proofs and the active-reuse boundary before the next CUDA
candidate is written.

## Runnable Runtime Bridge Gate

After syncing the branch to Spark, run:

```bash
scripts/run_spark_stage6_p4c_runtime_bridge_preflight.sh --host dgx-via-ssh
```

Expected bankable decision:

```text
PASS_P4C_ACTIVE_REUSE_RUNTIME_BRIDGE
```

This gate proves the real resident-runner path can select
`LYNN_NATIVE_ACTIVE_MOE_BACKEND=fused_zero_shadow_active_reuse_contract`, remove
active-expert BF16 shadows, call the P4C native symbol exactly once, and return a
caller-owned active-reuse two-phase output numerically close to the current
Triton packed path. It may bank only
`banked_p4c_active_reuse_runtime_bridge=true`; it must keep
`banked_fused_kernel=false` and `banked_default_promotion=false`.

## Runnable Speed-Baseline Gate

After syncing the branch to Spark, run:

```bash
scripts/run_spark_stage6_p4c_active_reuse_microbench.sh --host dgx-via-ssh
```

Banked Spark artifact:

```text
reports/stage6/p4c_active_reuse_microbench_20260604_104254
PASS_P4C_ACTIVE_REUSE_SPEED_BASELINE_RECORDED
```

Measured result: P4A two-stage **271.0976 us**, P4C active-reuse contract
**271.0624 us**, speedup **1.00013x**, output `rel_l2=0.0/max_abs=0.0`,
scratch `rel_l2=0.0/max_abs=0.0`, active scratch **8192 bytes**, packed/BF16
shadow ratio **0.37500016**.

This confirms the P4C boundary itself is not the bottleneck. It still banks only
`banked_p4c_active_reuse_speed_baseline=true`; fused-kernel speed and default
promotion remain closed until a real active-reuse CUDA/CUTLASS candidate replaces
the P4C symbol body and passes e2e/RC gates.

## Runnable Component-Profile Gate

After syncing the branch to Spark, run:

```bash
scripts/run_spark_stage6_p4c_component_profile.sh --host dgx-via-ssh
```

Banked Spark artifact:

```text
reports/stage6/p4c_component_profile_20260604_105640
PASS_P4C_COMPONENT_PROFILE_RECORDED
```

Measured result: full P4C **233.3920 us**, gate/up component **151.6704 us**,
down component **86.2144 us**. Component sum is **237.8848 us**, within
**1.01925x** of the full P4C timing despite component symbols allocating output
tensors. Gate/up is **63.76%** of the component sum; down is **36.24%**. Gate,
down, and composed output all report `rel_l2=0.0/max_abs=0.0`.

This makes the next speed candidate concrete: first replace or optimize the
gate/up half of the P4C active-reuse path. Down is still worth optimizing later,
but attacking it first can only address roughly one third of the current P4C
time.

## Runnable Gate/Up Shape-Sweep Gate

After syncing the branch to Spark, run:

```bash
scripts/run_spark_stage6_p4c_gateup_shape_sweep.sh --host dgx-via-ssh
```

Banked Spark artifact:

```text
reports/stage6/p4c_gateup_shape_sweep_20260604_111603
PASS_P4C_GATEUP_SHAPE_SWEEP_RECORDED
```

Measured result: current gate/up baseline **151.6864 us**. The best existing
launch shape is `tile_inter=2, threads=128` at **91.4112 us**, **1.659x** vs
current, with numeric checks passing (`rel_l2=0.0/max_abs=0.0` for the best
shape; all variants numeric-ok under the gate threshold). This is an actionable
diagnostic, not a promotion: it banks only
`banked_p4c_gateup_shape_sweep=true`, while
`banked_p4c_gateup_candidate=false`, `banked_fused_kernel=false`, and
`banked_default_promotion=false`.

The next low-risk speed candidate should wire the P4C gate/up half to
`tile_inter=2, threads=128`, then rerun the P4C active-reuse speed baseline. If
that fails to improve full P4C timing, move directly to a real CUDA/CUTLASS
gate/up kernel instead of more scalar launch-shape tuning.

## Runnable Gate/Up Shape-Candidate Microbench

After syncing the branch to Spark, run:

```bash
scripts/run_spark_stage6_p4c_gateup_shape_candidate_microbench.sh --host dgx-via-ssh
```

Banked Spark artifact:

```text
reports/stage6/p4c_gateup_shape_candidate_20260604_112635
PASS_P4C_GATEUP_SHAPE_CANDIDATE_RECORDED
```

Measured result: full P4C current `tile_inter=8` **267.3184 us**; full P4C
candidate `tile_inter=2` **185.0912 us**, **1.444x** vs current. P4A reference
shows the same direction (**267.4208 us** → **185.0432 us**, **1.445x**).
Candidate output and active scratch are exact against the P4A candidate-tile
reference (`rel_l2=0.0/max_abs=0.0`).

This banks `banked_p4c_gateup_shape_candidate=true`, not fused-kernel speed or
default promotion. The follow-up resident-runner and OpenAI server smoke gates
now pass with `LYNN_NATIVE_GATEUP_TILE_INTER=2`; the next promotion gate is RC
quality plus sustained server TPS before any default decision.

## Runnable Tile=2 Resident-Runner Preflight

After syncing the branch to Spark, run:

```bash
scripts/run_spark_stage6_p4c_runtime_bridge_preflight.sh --host dgx-via-ssh --gateup-tile-inter 2
```

Banked Spark artifact:

```text
reports/stage6/p4c_runtime_bridge_preflight_20260604_113325
PASS_P4C_ACTIVE_REUSE_RUNTIME_BRIDGE
```

Measured result on the real resident-runner path: layer 0 selected the native
P4C backend, removed BF16 active shadows, used caller-owned active scratch, and
called `fused_zero_shadow_active_reuse_contract` exactly once. The preflight
records `gateup_tile_inter=2` both in the requested config and in the native
call trace. Candidate vs Triton baseline is exact (`rel_l2=0.0/max_abs=0.0`).

This confirms the tile=2 candidate is not just a synthetic microbench trick. It
still banks only route/numeric evidence for the opt-in P4C path; server/RC
quality and default promotion remain open gates.

## Runnable Tile=2 OpenAI Server Smoke

After syncing the branch to Spark, run:

```bash
scripts/run_spark_stage6_p4c_tile2_server_smoke.sh --host dgx-via-ssh --p4c-runtime-pass --gateup-tile-inter 2
```

Banked Spark artifact:

```text
reports/stage6/p4c_tile2_server_smoke_20260604_122443
PASS_P4C_TILE2_SERVER_SMOKE
```

Measured result: the candidate OpenAI server ran
`LYNN_NATIVE_ACTIVE_MOE_BACKEND=fused_zero_shadow_active_reuse_contract` with
`LYNN_NATIVE_GATEUP_TILE_INTER=2` after prefill released the BF16 expert
shadows. `/health` recorded P4C native call delta **240**, **40** layers with
calls, recorded tile **2**, `inter_scratch=[1,8,512]`, `out=[1,2048]`,
released shadows still released, and reload observed at **23.246s** for the
second request. Baseline vs candidate completion text was exact **2/2**.

This banks `banked_p4c_tile2_server_smoke=true` only. The prompt set is a tiny
service smoke and produced short `<think>`-prefixed completions, so it is not a
quality RC, not a sustained TPS gate, and not a default-promotion gate.

## RC-mini Agreement Rejection

Wider server agreement was then run with the same opt-in P4C tile=2 backend:

```bash
scripts/run_spark_stage6_p4c_tile2_server_smoke.sh \
  --host dgx-via-ssh \
  --p4c-runtime-pass \
  --gateup-tile-inter 2 \
  --preset rc-mini \
  --prompt-limit 6 \
  --chat-prompts 2 \
  --max-new 8
```

Negative Spark artifact:

```text
reports/stage6/p4c_tile2_rcmini_agreement_20260604_124019
FAIL_P4C_TILE2_SERVER_SMOKE
```

This is an intentional rejection gate, not a promotion failure for the already
banked basic smoke. The candidate still reached the P4C server path:
`delta_total_calls=2040`, **40** layers called, `gateup_tile_inter=2`, release
and reload counters healthy. The failure is quality agreement:
completion text exact **3/6**, chat text exact **2/2**, and
`server_text_exact=false`.

Observed non-exact examples:

| Prompt class | Baseline prefix | Candidate prefix |
|---|---|---|
| Arithmetic | `<think></think> 42` | `<think> Here's a thinking process` |
| V9 bullets | `<think> Here's a thinking process` | `<think></think> * The` |
| Long-context marker | `<think> The user wants me to` | `<think></think> LYNN-Z` |

Decision: keep P4C tile=2 as opt-in server-smoke evidence only. Do **not**
promote it to RC/default, and do not widen server prompts again until a
first-divergence trace explains the token/layer where Triton and P4C separate.

## Shadow-Cycle First-Divergence Diagnostic

The server-like first-divergence diagnostic was then run with the same candidate
backend and an explicit BF16 shadow lifecycle:

```bash
python3 benchmarks/p33_native_active_moe_first_divergence.py \
  --candidate-backend fused_zero_shadow_active_reuse_contract \
  --native-active-moe-layers 0,1,...,39 \
  --candidate-release-shadows-before-decode
```

Banked diagnostic artifact:

```text
reports/stage6/p4c_tile2_shadow_cycle_first_divergence_20260604_130839
```

Result: the arithmetic prompt stayed top-1 identical for **8/8** reference
steps, including the server-like `reload→release→decode` shadow cycle. The
first hidden drift still starts early: `step=0/layer=13`,
`cosine=0.9999949336`, `rel_l2=0.00362795`, `max_abs=0.0078125`. Logits are
already non-identical (`step0 rel_l2=0.1021`, cosine `0.99487`), and step1
candidate margin compresses from Triton `0.28125` to candidate `0.046875`.

Interpretation: P4C tile=2 is not an immediate first-token failure on the
arithmetic smoke, but it has enough accumulated hidden/logit drift to explain
why wider rc-mini prompts split. This diagnostic supports the same boundary:
P4C tile=2 remains an opt-in diagnostic/basic-smoke path, not RC/default.
