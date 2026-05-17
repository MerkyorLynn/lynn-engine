# Lynn Engine Phase A Foundation Infrastructure

Date: 2026-05-17

## Purpose

The C++/Rust ROI doc names five Phase A native kernel islands. Before any of
those islands ship, three foundation pieces have to be in place so that:

- the production Triton/Torch baseline stays the default,
- a native CUDA backend is opt-in per kernel via environment variables,
- every kernel proves bitwise-or-near-bitwise parity against the baseline
  before it is promoted in serving.

See also:

- [LYNN_ENGINE_CPP_RUST_REWRITE_ROI_20260517.md](./LYNN_ENGINE_CPP_RUST_REWRITE_ROI_20260517.md) — Phase A target list and gates.
- [LYNN_ENGINE_MTP_VERIFY_ABI_20260517.md](./LYNN_ENGINE_MTP_VERIFY_ABI_20260517.md) — first Phase A kernel ABI.

## Three Foundation Pieces

| Piece | File | Responsibility |
|---|---|---|
| Kernel toggle registry | `engine/kernel_toggle.py` | Single source of truth for which Phase A kernels use the native backend, parsed from env vars at startup. |
| Parity harness | `benchmarks/kernel_parity_harness.py` | Triton-vs-CUDA cosine + rel-L2 + abs-max validation, persists JSON gate reports. |
| Per-op profiling JSON | (future, follow-up commit) | Per-call timing dumped to JSONL so a Phase A kernel landing can show before/after curves without scraping logs. |

The first two are landed in the foundation commit. The profiling JSON writer
is intentionally deferred to the first kernel landing so we do not invent a
schema before there is real data to write.

## Kernel Toggle Contract

Every Phase A kernel name is registered in `_PHASE_A_KERNELS` in ship order:

1. `mtp_verify` — MTP K=2/K=3 verify ABI + decoder layer
2. `active_moe` — variable-expert active-MoE fused boundary
3. `transposed_decode` — transposed NVFP4/W4A8 decode weight layout
4. `full_attn_boundary` — native/static full-attn layer boundary
5. `mtp_policy` — runtime MTP policy surface

Each is controlled by `LYNN_NATIVE_KERNEL_<KEY>=baseline|native_cuda|native_cpp`.
Default is `baseline`. Unknown values raise loud at startup, not silent.

A native backend MUST fall back to baseline if its CUDA extension is missing
or its parity gate file is stale; never crash the serving loop.

## Parity Gate Contract

Hard thresholds (no exception):

| Metric | Threshold | Reason |
|---|---|---|
| Cosine similarity (output) | >= 0.9999 | Aligned with SP-14 W4A8 mirror math contract (cos 0.9998 was already considered acceptable). |
| Max relative L2 | <= 1e-3 | Aligned with SP-08/SP-14 numeric tolerance and downstream P107/P116 trace parity. |
| Absolute max | informational | Diagnostic only; FP4 outliers can spike abs while staying within cosine/rel-L2 gates. |

Every kernel-specific benchmark drives a list of `ParityCase` through
`ParityHarness.run()` and writes JSON to `reports/phase_a/parity_<kernel>_<ts>.json`.

A failed parity gate is fail-loud: the harness raises `AssertionError`, the
serving toggle stays on `baseline`, and the offending native build is not
shipped.

## Ship Order for the Five Native Islands

The kernel toggle registry encodes the ship order; the foundation commit
does not change which kernel ships first. See the ROI doc for the rationale.
Recommended progression:

1. Land foundation (this commit): toggle registry, parity harness, doc.
2. Land `mtp_verify` ABI prototype and parity cases. Keep `mtp_verify`
   default baseline until P107/P116 parity + K=2 cost gate from the ROI doc
   pass.
3. Land `active_moe` boundary as the next island. P50/P69/P97 evidence drives
   its parity cases.
4. Subsequent islands follow as their per-kernel exit gates are met.

The serving image keeps every kernel on `baseline` until the corresponding
gate JSON shows `passed: true` for the production probe set, at which point
the operator can opt the kernel in by setting the env var.

## Out of Scope (Foundation Commit)

- No CUDA kernel code in this commit. Each Phase A island introduces its own
  `csrc/lynn_native/...` files in a later commit.
- No model loader changes. Toggle parsing happens at module import; the
  serving loop will not re-read env vars per request.
- No Rust. C++/CUDA only for Tier 1 per ROI doc.
- No production behavior change. Default is `baseline` everywhere; this
  commit is observable only via new env-var support and the harness helper.

## Why This Foundation Now

The MTP verify ABI is the first kernel island per the ROI doc. Writing the
verify ABI without a parity harness or a toggle is how silent-correctness
regressions get into serving (see the SP-13/SP-14 W4A8 mirror math contract
history). Landing the harness and the toggle first means every subsequent
Phase A kernel can prove correctness in the same way and ship behind the
same opt-in switch, without inventing per-kernel infrastructure each time.

## How To Extend

Adding a new Phase A kernel:

1. Append the kernel name to `_PHASE_A_KERNELS` in `engine/kernel_toggle.py`.
2. Create the native implementation in `csrc/lynn_native/...` and the Python
   shim in `engine/native_cuda.py` or a kernel-specific module.
3. Add a kernel-specific benchmark under `benchmarks/parity_<kernel>.py`
   that constructs the `ParityCase` list and calls `ParityHarness.run()`.
4. Land a gate JSON under `reports/phase_a/` showing `passed: true`.
5. Document the kernel's exit gate (matching the ROI doc's per-target
   gates) in this file or a per-kernel doc.

Adding a non-Phase-A kernel toggle is NOT supported by this module; use the
existing `engine/native_fp4_policy.py` style for projection-level allowlists.
