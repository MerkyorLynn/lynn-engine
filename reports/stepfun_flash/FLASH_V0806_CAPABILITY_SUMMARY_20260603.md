# Step 3.7 Flash — capability boundary summary (Lynn CLI v0.80.6, 黑灯工厂) — 2026-06-03

**Setup:** CLI pinned to **pure Step 3.7 Flash** (StepFun cloud 198B-MoE, *no* Brain
fallback), `worker run --approval yolo --sandbox danger-full-access` (autonomous
plan→edit→shell→self-correct). Every task **independently verified by me** (Mac CPU,
or the Spark GPU for kernels). 满血 Flash, not the Q3_K_M local quant.

## Scoreboard
| # | task | domain | result |
|---|---|---|:--:|
| — | algorithms (edit-dist, dijkstra, regex-match LeetCode-Hard, LRU, …) | algo gen | **9/9** + 3/3 agent (Q3_K_M) |
| W1 | NVFP4 E2M1 dequant (numpy) | numeric/bit | ✅ first-try |
| W2 | NVFP4 pack + round-trip (numpy) | numeric + self-fix | ✅ hit bug → self-fixed |
| NC1 | kvcalc package + tests | multi-file + tests | ✅ clean |
| NC2 | fix merge_intervals bug | debug existing code | ✅ diagnosed + fixed |
| NC3 | jsonl → markdown summary | data / IO / glue | ✅ clean |
| NC4 | arithmetic evaluator (no `eval`) | long-horizon, constraint | ✅ first-try, 143-line parser |
| NC6 | CSV parser (no `csv` module) | parsing / state machine | ✅ first-try, all edge cases |
| NC7 | cross-file refactor (rename + sig change ×3 files) | cross-file consistency | ✅ correct |
| NC9 | flatten nested (hidden string trap, self-authored tests) | edge-case + test rigor | ✅ anticipated the trap |
| **NC5** | **thread-safe LRU (doubly-linked-list)** | **pointer data structure** | **❌ FAILED** |
| **W3** | **Triton GEMV kernel** | **GPU / CUDA kernel** | **❌ FAILED** |

**Non-CUDA: 9 / 10 pass. The two failures are the whole story.**

## What Flash does WELL (non-CUDA)
- **Self-contained correct logic + verify-by-running** — its sweet spot. Numeric/bit
  (NVFP4 dequant/pack), algorithms, parsers (CSV, arithmetic), data/glue.
- **The autonomous loop is real:** plans (`update_plan`), writes, runs, reads the error,
  **self-corrects** (W2: `'slice' object not subscriptable` → rewrote → PASS), re-runs
  to green, terminates cleanly when it fits the step budget.
- **Multi-file & cross-file** — packages + tests (NC1), 3-file rename+signature ripple (NC7).
- **Debugging existing code** — reads, finds a subtle bug (NC2 missing sort), minimal fix.
- **Edge-case anticipation / test rigor** — NC9: handled the string-iteration trap
  *and* wrote a string test case **without being told**.
- **Constraint-following** — honored "no `eval`" (NC4) and "no `csv` module" (NC6).
- **State machines / char-by-char parsing** — clean (CSV with quoted commas, embedded
  newlines, `""` escapes).

## What goes WRONG (the boundary)
1. **GPU kernels — a clear difficulty gradient (all verified by nvcc+run on Spark):**
   - **Simple CUDA C++ (SAXPY): PASS first-try.** Full program — kernel + cudaMalloc/
     Memcpy + grid/block + error-check macro + self-check — compiled (`nvcc -arch=sm_121`)
     and printed PASS. Flash knows the standard CUDA host API + simple kernels.
   - **Medium CUDA C++ (parallel reduction): FAILED.** Confabulated a non-existent
     `cudaAtomicAdd` (correct is the device intrinsic `atomicAdd`) **and** wrapped it in
     the host error-check macro (device atomics aren't host CUDA APIs). Given the explicit
     `cudaAtomicAdd undefined` compile error it removed the wrapper but **kept the wrong
     name** — it doesn't reliably know the less-common device intrinsics, and couldn't
     self-correct (no GPU/nvcc on the Mac to iterate).
   - **Complex Triton (GEMV, W3): FAILED.** Real broadcasting bug (`x_tile[:,None].T`),
     plausible-but-wrong; couldn't fix from the error without a GPU.

   **Pattern:** standard boilerplate + simple kernels = fine; **device intrinsics
   (`atomicAdd`), shared-mem reductions, tiled GEMVs, and Triton-specific semantics =
   confabulates plausible-but-wrong code and can't self-correct without a GPU to run on.**
2. **Manual pointer / linked-data-structure surgery (NC5).** Hand-rolled a doubly-linked-
   list LRU (instead of `OrderedDict`), got single-thread eviction right but the
   **get()-recency-refresh pointer logic wrong**, then **debugged it (inspected
   head/tail/map, patched unlink) but ran out of steps before fixing it.** The lock was
   fine — it's not concurrency, it's DLL pointer state + debugging depth. (A stdlib
   `OrderedDict` LRU would have been trivially correct; the hand-roll choice backfired.)

**Pattern:** strong when the task is "write correct logic, verify by running"; weak when
(a) it *can't* run/verify (CUDA on a GPU-less host) or (b) the bug is in stateful pointer
manipulation needing many careful debug steps that exceed the worker's step budget.

## Operational notes (running Flash 黑灯工厂)
- Use **`--sandbox danger-full-access`** — `workspace-write` blocks shell, so the agent
  can't run/verify its own code (W1 stalled on exactly this).
- **Raise the worker step budget** for debugging-heavy / complex-data-structure tasks —
  the tight default `max_steps` exits `failed` even when the work passed on the last step
  (W2) or was nearly done (NC5/NC7).
- **Don't trust `worker.ok` alone — verify the diff + re-run** (NC7 passed despite an
  `ok:false`/`git.diff:0` artifact; W3-fix once returned `ok:true` with zero edits).
- Prefer **stdlib** (e.g. `OrderedDict`) over hand-rolled data structures when correctness
  matters; reserve GPU-kernel work for a GPU-in-the-loop or a stronger kernel model.

## Bottom line
For Lynn, **Step 3.7 Flash reliably owns the non-CUDA engineering layer** — algorithms,
numeric logic, multi-file modules, debugging, parsing, data/glue, refactors, tests — fully
autonomously, with genuine self-correction (9/10 here, all clean). Its two real limits are
**GPU/CUDA kernels** and **manual pointer-heavy data structures**; both trace to the same
root — it can't close the loop (no GPU to run kernels; not enough debug steps for pointer
state). Use it as the autonomous CPU/logic/glue/test engine; keep a human/GPU/stronger
model on first-pass Triton/CUDA and gnarly pointer code. Artifacts: `noncuda_factory/`.
