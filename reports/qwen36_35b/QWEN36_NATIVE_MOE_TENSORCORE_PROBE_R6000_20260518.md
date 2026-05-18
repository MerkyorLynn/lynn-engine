# Qwen3.6 Native MoE TensorCore Probe R6000 2026-05-18

Verdict: CLOSED.

`native_slot_tensorcore_probe` replaced 16 sequential slot `mm` calls with two batched `torch::bmm` calls. It improved numerical drift versus the scalar custom path but was far too slow for the MoE hot path.

| Candidate | max_abs_max | cosine_min | mean latency | Verdict |
|---|---:|---:|---:|---|
| TensorCore batched-bmm slot probe | 1.953125e-03 | 0.9999891520 | 0.2456 ms | CLOSED |

Reference points:

| Path | Mean latency | Notes |
|---|---:|---|
| Triton active baseline | ~0.059 ms | Current serving reference |
| Fast scalar output-owned | ~0.052 ms | Faster but strict drift remains |
| Strict cuBLAS/torch::mm oracle | ~0.467 ms | Exact but unusably slow |
| TensorCore batched-bmm probe | ~0.246 ms | Numerically acceptable-ish, too slow |

Conclusion: `torch::bmm`/batched cuBLAS is not a viable bridge for this single-token top-8 MoE slot workload. The next viable line is packed slot layout plus a custom output-owned kernel that avoids PyTorch/cuBLAS launch overhead while preserving the slot-order contract.
