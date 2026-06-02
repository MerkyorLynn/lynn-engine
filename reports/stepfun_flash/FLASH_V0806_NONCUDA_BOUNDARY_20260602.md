# Step 3.7 Flash — non-CUDA-kernel capability boundary (Lynn CLI v0.80.6, 2026-06-02)

Pure Flash (StepFun cloud 198B-MoE, no Brain fallback), `worker run --approval yolo
--sandbox danger-full-access`, real tasks, **each independently verified by me** (Mac
CPU run, or Spark GPU for kernels).

## Non-CUDA tasks — 5/5 PASS, all clean (full autonomous write→run→fix loop)
| task | type | result |
|---|---|---|
| W1 NVFP4 dequant (numpy) | numeric / bit logic | **correct first-try**, PASS |
| W2 NVFP4 pack + round-trip (numpy) | numeric + self-fix | hit a real bug → **self-corrected** → PASS |
| NC1 `kvcalc` package + tests | multi-file + tests | planned → 2 files + test → PASS, clean finish |
| NC2 fix `merge_intervals` bug | debug *existing* code | diagnosed missing sort → patched → PASS |
| NC3 jsonl→markdown summary | data / IO / glue | correct sorted table + rel_to_best → PASS |

(+ earlier algorithm eval: 9/9 generation incl. a LeetCode-Hard + 3/3 agent on Q3_K_M.)

## What Flash does WELL (non-CUDA)
- **Numeric / bit-level logic** — NVFP4 E2M1 dequant + pack (nibble packing, sign bit,
  per-16 block scales) correct and careful, with rigorous hand-computed self-tests.
- **Multi-file structure** — package + test + import/path wiring (NC1).
- **Debugging existing code** — reads, diagnoses a subtle bug (unsorted input), minimal
  correct fix (NC2).
- **Data / IO / glue** — json parse, sort, markdown formatting, clean stdlib (NC3).
- **The full autonomous loop** — `update_plan` → write → run → read error → self-correct
  → re-run to green → **clean termination** when the task fits the step budget.

## What goes WRONG
- **GPU / CUDA kernels (the one real capability failure).** W3 Triton GEMV: wrote a
  plausible kernel with a real broadcasting bug (`x_tile[:, None].T`), which only
  surfaced on the Spark GPU; and given the GPU error it **could not fix it** without a
  GPU to iterate on (2 attempts, zero edits — once declared `ok:true` with no change).
  Triton/CUDA is the weak spot: it leans on the execute→observe→fix loop, and that loop
  needs a GPU it doesn't have on the Mac.
- **Operational / loop quirks (not capability).** Occasional `apply_patch` corrupt-patch
  (falls back to `write_file`); rarely returns `ok:true` without taking the action; the
  worker's tight `max_steps` can exit `failed` even after the work passed on the final
  step. Mitigations: use `--sandbox danger-full-access` (workspace-write blocks shell),
  verify the diff + re-run (don't trust `worker.ok` alone), raise the step budget for
  long multi-iteration tasks.

## Bottom line
For **non-CUDA engineering** — algorithm/numeric logic, multi-file modules, debugging,
tests, data/glue, scripting — **Step 3.7 Flash is reliably strong** (5/5 here, all clean,
with genuine self-correction; the lights-out worker works end-to-end). Its boundary is
**GPU-kernel code** (Triton/CUDA), where it writes plausible-but-buggy kernels and can't
self-fix without a GPU. Practical split for Lynn: **let Flash own the CPU / logic / glue /
test layer autonomously**; keep a GPU-in-the-loop (or a stronger kernel model) for
first-pass Triton/CUDA. Artifacts: `noncuda_factory/`.
