# R6000 NVFP4 FP4-MMA — Upstream Diff And GO/NO-GO Gate

Date: 2026-06-04

Verdict: **do not build dense/private NVFP4 as a Lynn moat.** Upstream
llama.cpp is already making NVFP4 a first-class ggml type and has merged
Blackwell native FP4 CUDA plumbing. Lynn's only still-defensible R6000 kernel
investment is **grouped-MoE FP4-MMA**, and only under a hard speed gate.

## ⏰ TONIGHT (tightened owner directive — supersedes the 2-3 window in Hard Gate)

**R6000 tonight keeps exactly ONE value point: `grouped-MoE FP4-MMA REAL SPEED`. Stop everything else now.**

- **Only this**: a grouped-MoE active-prefill FP4-MMA kernel + a **real measured R6000 e2e A/B** that beats the current best Lynn path, RC-exact. Nothing else is in scope tonight.
- **Stop now**: any further census / shape / ABI / contract / numeric-smoke gate. They are DONE and do **not** count as progress. The "many gates, zero speed" pattern ends here.
- **Tight time-box**: the next **1–2 real speed candidates**. If they fail parity or get rejected like the trace-derived R5-C4 → **NO-GO, stop immediately** (do not open more sub-gates).
- **Fork is pre-decided** (做出来就贡献,做不出来就直接抄):
  - ✅ **Banked a real grouped-MoE FP4-MMA speed win** → write it up as an upstream contribution to llama.cpp `#23572` and stop there.
  - ❌ **No speed win in the box** → **STOP the private kernel and directly adopt upstream NVFP4** (use llama.cpp's merged path; redirect R6000 hours to product / Qwen-specific wins).
- **GO evidence = a real e2e A/B speed number only.** No exceptions.

This memo is strategy evidence, not a kernel result. It does not bank R5-C4
speed, decode TPS, server behavior, RC quality, or default promotion.

## Verified Upstream State

| Area | Current state | Source |
|---|---|---|
| NVFP4 ggml type | `GGML_TYPE_NVFP4 = 40`, `GGML_FTYPE_MOSTLY_NVFP4 = 26` are present on master. | `ggml/include/ggml.h` |
| NVFP4 block layout | `block_nvfp4` uses four UE4M3 per-16 sub-block scales plus packed E2M1 values. | `ggml/src/ggml-common.h` |
| Blackwell native NVFP4 CUDA | PR `#22196` was merged on 2026-04-28 as a repost of `#21896`; it includes a Blackwell native NVFP4 CUDA path. | `ggml-org/llama.cpp#22196` |
| N4_0/native FP4 prompt-processing work | PR `#23572` is open. It explicitly reports +40% PP for dense prompt processing, but also says it is a starting point and not yet fully spec-compatible NVFP4 checkpoint support. | `ggml-org/llama.cpp#23572` |
| Quantizer scale/default mapping | PR `#22897` is open; it adds NVFP4 default mapping and emits `.scale` / `.input_scale` tensors required by the CUDA MMA dequant path. | `ggml-org/llama.cpp#22897` |
| Gemma4 text NVFP4 support | PRs `#21971`, `#22804`, and `#23682` are merged for Gemma4 NVFP4 tensors / conversion. | `ggml-org/llama.cpp#21971`, `#22804`, `#23682` |

Important correction: `#23572` is **not** proof that upstream has completed every
NVFP4 checkpoint-compatible path. It is enough to show that dense NVFP4/FP4-MMA
is being actively commoditized upstream; it is not a reason to claim upstream
already solved Lynn's grouped-MoE gap.

## Lynn R6000 State

From the Stage 6 evidence ledger as of this memo:

| Gate | Status |
|---|---|
| `r6000_fp4_mma_census` | Banked bring-up / census only |
| `r5a_layout_bridge` | Diagnostic banked |
| `r5b_e8m0_repack` | Closed negative |
| `r5c*` | ABI / numeric-smoke / value-materialization / parity banked |
| `r5c4_full_active_moe_prefill_speed_ab` | `READY_WAITING_R6000`; no speed artifact banked |
| `r5c4_trace_candidate_rejection` | Closed negative; trace-derived candidate rejected |

Lynn has banked important layout and numeric evidence, but **no R6000 grouped-MoE
FP4-MMA speed win**. `banked_kernel_speed=false` remains the honest state.

## Strategic Diff

| Axis | Upstream llama.cpp | Lynn |
|---|---|---|
| Dense NVFP4 format/runtime | Actively upstreamed and maintained by the ggml community | Redundant if treated as private Lynn IP |
| Dense Blackwell FP4-MMA | Already merged or under active open PR work | Do not compete here unless contributing upstream |
| Grouped-MoE FP4-MMA | No confirmed upstream speed path from the checked PRs | Lynn's only unique R6000 kernel opening |
| Product leverage | broad model zoo, runtime distribution, MIT ecosystem | Lynn product/agent integration and Qwen-specific evidence discipline |

## Hard Gate

Keep investing R6000 only if the next focused push targets **grouped-MoE
FP4-MMA** and produces real speed evidence.

GO requires all of:

- A same-scope grouped-MoE active prefill candidate, not dense-only FP4-MMA.
- Numeric parity against the current W4A16/P3 reference: max error, relative L2,
  cosine, route order, and fault injections all reported.
- Real measured R6000 A/B over the full active-MoE boundary:
  `route/order -> gate/up -> SwiGLU -> down -> top-k weighted sum`.
- A speed win against the current best Lynn path; trace-derived or
  gate/up-only timing does not count.
- Non-claims remain false: no Spark decode TPS, no server/RC behavior,
  no default promotion, no full-transformer prefill.

NO-GO triggers:

- The next 2-3 real speed candidates fail numeric parity or are rejected like the
  trace-derived R5-C4 artifact.
- The candidate only reproduces upstream dense NVFP4 / N4_0 work.
- Upstream lands a grouped-MoE FP4-MMA path first.

On NO-GO:

- Stop private R6000 kernel grinding.
- Package Lynn's R5-A/R5-C layout and UE4M3/CUTLASS learnings as an upstreamable
  contribution note for llama.cpp's NVFP4 work.
- Redirect effort to the durable Lynn moat: product integration, desktop agent
  behavior, and short-term Qwen-specific serving wins.

## Working Rule

**Census / ABI / contract / numeric smoke is not speed.** R6000 remains valuable
only if it produces a same-scope grouped-MoE FP4-MMA prefill A/B that can later
be contributed upstream or used as a Lynn product accelerator.

## References

- https://github.com/ggml-org/llama.cpp/blob/master/ggml/include/ggml.h
- https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-common.h
- https://github.com/ggml-org/llama.cpp/pull/22196
- https://github.com/ggml-org/llama.cpp/pull/23572
- https://github.com/ggml-org/llama.cpp/pull/22897
- https://github.com/ggml-org/llama.cpp/pull/21971
- https://github.com/ggml-org/llama.cpp/pull/22804
- https://github.com/ggml-org/llama.cpp/pull/23682
- `reports/stage6/stage6_evidence_ledger_20260604.json`
- `reports/stage6/R5C4_FULL_ACTIVE_MOE_PREFILL_SPEED_AB_CONTRACT_20260604.md`
