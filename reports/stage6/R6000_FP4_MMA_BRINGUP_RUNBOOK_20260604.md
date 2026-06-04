# Stage 6 R6000 FP4-MMA Bring-Up Runbook

Date: 2026-06-04

Verdict: **runbook/tooling gate only; no kernel is promoted by this document.**

## Machine Choice

Use the RTX PRO 6000 96GB host with the largest disk headroom:

| Choice | Decision | Reason |
|---|---|---|
| `723机 RTX PRO 6000 96GB` | **Preferred** | Same weekly price, 24 CPU cores, 120GB RAM, and 943GB expandable disk. |
| `895机 RTX PRO 6000 96GB` | Acceptable fallback | Same GPU, but less disk/CPU headroom. |
| `718机 RTX PRO 6000 96GB` | Avoid if possible | Same price but only 171GB expandable disk. |
| `H20-NVLink 96GB` | **Do not use for FP4-MMA mainline** | More expensive and not the Blackwell FP4 Tensor Core target for this track. |

Minimum disk: **500GB free**.

Recommended disk: **800GB-1TB free**. The 943GB option is the safe default because
model artifacts, Docker layers, CUTLASS/vLLM/TRT-LLM builds, native extension
cache, Nsight traces, and result artifacts accumulate quickly.

## Why This Gate Exists

Spark sm_121 is still valuable for 35B service/memory/MTP/compiled-loop work, but
it cannot realize the native FP4-MMA payoff. R6000/Blackwell is the correct target
for:

- CUTLASS/CuTe block-scaled NVFP4/MXFP4 GEMM.
- Lynn grouped native-FP4 active-expert kernels.
- Public-kernel census before writing custom CUDA from scratch.

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
