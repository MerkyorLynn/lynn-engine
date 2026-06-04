# Stage 6 R6000 FP4-MMA Bring-Up Runbook

Date: 2026-06-04

Verdict: **runbook/tooling gate only; no kernel is promoted by this document.**

## Banked Artifact

The first clean R6000 census is banked here:

| Field | Value |
|---|---|
| Artifact | [r6000_fp4_mma_census_20260604_164457](r6000_fp4_mma_census_20260604_164457/summary.md) |
| Host | AutoDL 727 |
| Device | `NVIDIA RTX PRO 6000 Blackwell Server Edition` |
| CUDA capability | `[12, 0]` |
| Visible memory | `94.973 GiB` |
| Data workspace | `/root/autodl-tmp`, `880 GiB` free |
| Public source candidates | vLLM source scan: `200` candidates |
| Contract suite | P76/P79/P85/P87/P103 all pass |
| Decision | `PASS_R6000_FP4_MMA_BRINGUP` |

This banks only machine/toolchain/public-kernel readiness. It does not promote a
Lynn kernel, runtime default, or speed claim.

## Machine Choice

Use any available RTX PRO 6000 96GB host with the largest disk headroom.
Do not bind the route to a fixed AutoDL host ID; inventory changes quickly.

| Choice | Decision | Reason |
|---|---|---|
| `RTX PRO 6000 96GB, 800GB-1TB expandable disk` | **Preferred** | Correct Blackwell FP4-MMA target with enough room for model artifacts, builds, Nsight traces, and result artifacts. |
| `RTX PRO 6000 96GB, >=500GB free disk` | Acceptable | Enough for the census gate and initial FP4-MMA POC if caches/artifacts are pruned. |
| `RTX PRO 6000 96GB, <500GB free disk` | Avoid if possible | Too tight for native builds plus model/checkpoint artifacts; use only for minimal capability smoke. |
| `H20-NVLink 96GB` | **Do not use for FP4-MMA mainline** | More expensive and not the Blackwell FP4 Tensor Core target for this track. |

Minimum disk: **500GB free**.

Recommended disk: **800GB-1TB free**. This headroom is the safe default because
model artifacts, Docker layers, CUTLASS/vLLM/TRT-LLM builds, native extension
cache, Nsight traces, and result artifacts accumulate quickly.

## Why This Gate Exists

Spark sm_121 is still valuable for 35B service/memory/MTP/compiled-loop work, but
it cannot realize the native FP4-MMA payoff. R6000/Blackwell is the correct target
for:

- CUTLASS/CuTe block-scaled NVFP4/MXFP4 GEMM.
- Lynn grouped native-FP4 active-expert kernels.
- Public-kernel census before writing custom CUDA from scratch.

## Spark vs R6000 Division

Keep the two machines' evidence lanes separate:

| Machine | Owns | Does not own |
|---|---|---|
| Spark sm_121 | 35B NVFP4 serving lifecycle, 60 GiB shadow-release productization, MTP eager-runtime work, decode hot-loop host-dispatch ROI probes, RC/quality smoke, and regressions for the current ~44-45 TPS service stack. | FP4-MMA speed claims, CUTLASS/CuTe native-FP4 promotion, or grouped-MoE tensor-core conclusions. |
| RTX PRO 6000 96GB | FP4-MMA capability/census, CUTLASS/CuTe/vLLM Marlin-Machete public-kernel map, native grouped-MoE FP4-MMA POC, and cross-device NVFP4 kernel moat work. | Spark decode-speed deliverables, serving-memory claims already banked on Spark, or production defaults before RC quality gates pass. |

Rule: **a result is banked only on the lane that can actually exercise the
hardware feature under test**. Spark can reject bad service/runtime ideas; R6000
is required before claiming native FP4-MMA payoff.

Public references to verify on the machine:

- NVIDIA CUTLASS Blackwell narrow/block-scaled GEMM docs:
  <https://docs.nvidia.com/cutlass/latest/media/docs/cpp/blackwell_functionality.html>
- NVIDIA CUTLASS repository:
  <https://github.com/NVIDIA/cutlass>
- vLLM NVFP4 Marlin API path:
  <https://docs.vllm.ai/en/v0.20.1/api/vllm/model_executor/kernels/linear/nvfp4/marlin/>

## First Command

From the `lynn-engine` repo on the R6000 host:

```bash
python3 scripts/r6000_stage6_fp4_mma_census.py \
  --run-contracts \
  --out reports/stage6/r6000_fp4_mma_census_$(date +%Y%m%d_%H%M%S)/result.json
```

Then summarize:

```bash
python3 scripts/summarize_stage6_r6000_fp4_mma_census.py \
  reports/stage6/r6000_fp4_mma_census_*/result.json \
  --markdown-out reports/stage6/r6000_fp4_mma_census_latest.md \
  --strict-exit
```

## Pass Criteria

The census is banked only if all of these are true:

- CUDA is visible through PyTorch.
- Capability major is `12`.
- GPU memory is R6000-class (>=85 GiB visible).
- Workspace disk free is >=500 GiB.
- Public-kernel census records vLLM/CUTLASS/Triton/FlashInfer availability.
- A vLLM NVFP4/Marlin/Machete/CUTLASS path is visible or explicitly importable.
- The native contract suite is recorded and passes:
  - P76 CUTLASS/CuTe toolchain smoke.
  - P79 NVCC FP4-MMA target matrix.
  - P85 block-scaled FP4-MMA contract.
  - P87 FP4 layout tile contract.
  - P103 FP8-activation x FP4-weight MMA capability.

## Promotion Boundary

Passing this gate means **the machine/toolchain/public-kernel map is ready**.

It does **not** mean:

- A Lynn kernel is faster.
- A Lynn kernel is promoted.
- Runtime defaults change.
- Spark decode speed claims change.

The next gate after PASS is a Lynn NVFP4 grouped-MoE FP4-MMA POC that starts from
CUTLASS/CuTe and the public Marlin/Machete census, rather than hand-scanning tile
shapes from zero.
