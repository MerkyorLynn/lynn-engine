# Step 3.7 Flash — Lynn CLI v0.80.6 黑灯工厂 (lights-out) eval on real engine work (2026-06-02)

**Setup:** `@lynn/cli 0.80.6`, CLI **pinned to pure Step 3.7 Flash** (StepFun cloud
`api.stepfun.com/step_plan/v1`, model `step-3.7-flash`, a **198B-MoE**) via BYOK —
*no Brain fallback to MiMo/Spark*, so this measures Flash's *own* capability. Agent
mode = `Lynn worker run --brief … --worktree … --jsonl --approval yolo --sandbox
danger-full-access` (the real lights-out factory: plan → edit files → run shell →
self-correct). Tasks are **real engine work** (the kind this repo needs), each
**independently verified** by me (CPU on Mac, or the real Spark GPU). CLI restored
to defaults afterwards.

## Results
| # | task (real engine logic) | mode | outcome | independent verify |
|---|---|---|---|---|
| ping | instruction follow | `-p` | exact, 1.4s, thinking model | ✓ |
| W1 | NVFP4 E2M1 **dequant** (numpy) | worker, *workspace-write* | code **correct first-try**; but loop blocked — sandbox denied shell | **PASS** (my run) |
| W2 | NVFP4 **pack + round-trip** (numpy) | worker, *danger-full-access* | **full loop**: plan→write→run→**real bug**→**self-fix**→PASS | **PASS** (my run) |
| W3 | **Triton** bf16 GEMV kernel | worker, write-only | wrote 114-line kernel, clean finish — **but a real broadcast bug** | **FAIL on Spark GPU** |
| W3-fix×2 | fix the Triton bug from the GPU error (no GPU to run) | worker | **could not fix**: declared "done" with **zero edits** / read file but never edited | bug persists |

## What Flash is good at
- **CPU/numpy/torch engine logic — excellent.** W1 NVFP4 dequant (E2M1 table,
  nibble unpack, sign bit, per-16 block scale) was **correct on the first write**,
  with a *rigorous* self-test that hand-computes all 16 values incl. the −0.0 edge.
  W2 NVFP4 pack+dequant round-trip was correct too.
- **The autonomous run-fix loop works (when it can execute).** W2: it made a plan
  (`update_plan`), wrote the file, ran it, hit a **real bug it introduced**
  (`TypeError: 'slice' object is not subscriptable`), and **self-corrected**
  (apply_patch failed → fell back to a full rewrite) → re-ran → `PASS`. That is
  the genuine lights-out loop.
- **Fast + cheap:** StepFun cloud, ~1–2 s per step; planning + tool use are clean.

## Where Flash is weak (honest)
- **Triton / GPU-kernel code is less reliable.** W3's kernel had a real
  broadcasting bug (`x_tile[:, None].T` → shape mismatch on the K-reduction); it
  *looked* plausible but doesn't compile on the GPU.
- **Can't fix kernel bugs by reasoning alone.** Given the exact GPU error twice
  (no GPU to iterate on), Flash **never produced a working edit** — once it
  declared success with **no file change at all**, once it read the file and
  still didn't edit. It clearly *relies on the execute→observe→fix loop*; remove
  execution and its kernel debugging collapses.
- **Agent-loop reliability quirks:** `apply_patch` emitted a corrupt patch (fell
  back to rewrite); it repeated an identical command (caught by the CLI loop
  guard); it sometimes returns `ok:true` **without taking the required action**;
  and the worker's `max_steps` budget is tight enough that *passing* work can
  still exit `failed` (W2 PASSed on the last allowed step).

## Operational lessons (for running Flash 黑灯工厂)
1. **Use `--sandbox danger-full-access`, not `workspace-write`** — workspace-write
   blocks the shell, so the agent can't run/verify its own code (W1 stalled on this).
2. **Give it an execution environment.** Flash's strength is the run-fix loop; on
   GPU-kernel tasks it needs an actual GPU to iterate, or it can't converge.
3. **Raise/seed the step budget** for multi-iteration tasks, and don't trust
   `worker.ok` alone — verify the artifact (diff + run).

## Bottom line
As Lynn's product coding head, **Step 3.7 Flash is genuinely strong on standard /
CPU-side engineering with a run loop** — correct dequant/pack logic, careful tests,
real autonomous self-correction. For the **GPU-kernel engine work this repo lives
on (Triton/CUDA on Spark)** it is **weaker and needs the execute-on-GPU loop**;
first-pass Triton had a real bug and it could not fix it from an error message
alone. Practically: let Flash own the CPU logic, scripting, glue, and the
write→run→fix loop where execution is available; keep a GPU-in-the-loop (or a
stronger kernel model) for first-pass Triton/CUDA. Artifacts: `v0806_factory/`.
