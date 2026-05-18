# Qwen3.6 Native MoE TensorCore Probe R6000 2026-05-18

Verdict: CLOSED.

Two PyTorch/cuBLAS-backed TensorCore bridge shapes were tested against the p135/p136 slot-order fixtures. Both preserve the slot-order numerical contract better than early scalar experiments, but both are too slow for the active MoE hot path.

| Candidate | max_abs_max | cosine_min | mean latency | Verdict |
|---|---:|---:|---:|---|
| 2x batched-bmm slot probe | 1.953125e-03 | 0.9999891520 | 0.2456 ms | CLOSED |
| fused gate_up mm + bmm down | 1.953125e-03 | 0.9999891520 | 0.1088 ms | CLOSED |

Reference points:

| Path | Mean latency | Notes |
|---|---:|---|
| Triton active baseline | ~0.059 ms | Current serving reference |
| Fast scalar output-owned | ~0.052 ms | Faster but strict drift remains |
| Strict cuBLAS/torch::mm oracle | ~0.467 ms | Exact but unusably slow |
| fused gate_up mm + bmm down | ~0.109 ms | Cleaner bridge, still slower than Triton |

Conclusion: PyTorch/cuBLAS bridge shapes are not a viable path for this single-token top-8 MoE slot workload. The next viable line is packed slot layout plus a custom output-owned kernel that avoids PyTorch/cuBLAS launch overhead while preserving the slot-order contract.

## P139b Pretransposed TensorCore Probe

Verdict: AMBER_FAST_PRETRANSPOSED.

This variant moves the expensive `reshape/transpose/contiguous` work into a load-time repack step and passes precomputed `W_fused_T [2048, 8192]` plus `W_down_T [8, 512, 2048]` into the hot path. Decode-time work is limited to `mm`, `view`, `silu*up`, `bmm`, and BF16 route reduce.

| Candidate | max_abs_max | cosine_min | mean latency | max latency | Verdict |
|---|---:|---:|---:|---:|---|
| pretransposed fused gate_up + bmm down | 1.953125e-03 | 0.9999890924 | 0.0527 ms | 0.0588 ms | AMBER_FAST_PRETRANSPOSED |

This is the first TensorCore bridge candidate that beats the Triton active baseline on the fixture harness. It is not default-promotable because fixture-level numerical drift remains above strict default thresholds, but it is a useful evidence point for the offline repack direction.
