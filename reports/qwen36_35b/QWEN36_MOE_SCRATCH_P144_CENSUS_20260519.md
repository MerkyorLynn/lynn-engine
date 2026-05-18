# Qwen3.6-35B W4A16 MoE Scratch P144 Census

**Date:** 2026-05-19  
**Probe:** `benchmarks/p144_decode_launch_census.py`  
**Purpose:** compare safe fastfixed MoE with `LYNN_MOE_FAST_FIXED=0` and caller-owned active-MoE scratch.

## Result

| Profile | Runner decode TPS | Profiler wall TPS | CUDA attributed events / token | MoE NVFP4 events | Interpretation |
|---|---:|---:|---:|---:|---|
| safe fastfixed | 15.11 | 2.84 | 16085.42 | 342 | profiler-heavy baseline |
| `LYNN_MOE_FAST_FIXED=0` | 15.57 | 2.89 | 16085.33 | 342 | no meaningful launch/event change |
| `LYNN_MOE_FAST_FIXED=0 LYNN_MOE_ACTIVE_SCRATCH=1` | 14.63 | 2.78 | 16085.58 | 342 | scratch is not a speed lever by itself |

The profiler wall TPS is not a serving number; `torch.profiler` slows this
decode path heavily and attributes both ATen operators and CUDA kernels. The
useful signal here is the relative comparison: caller-owned active-MoE scratch
does not reduce the observed event count or grouped MoE kernel count, and it is
slightly slower in the profiled runner timing.

## Implication

Preallocating the existing Triton gate/up and down outputs is not enough to move
the 107 TPS safe line. The next boundary attempt should not be another scratch
toggle. It needs a real Triton-contract-preserving fused boundary or an
implemented `grouped_per16_fused` replacement that passes resident P37.

## Artifacts

- `reports/qwen36_35b/p144_safe_fastfixed_20260519_030042.json`
- `reports/qwen36_35b/p144_fastfixed0_20260519_030042.json`
- `reports/qwen36_35b/p144_fastfixed0_scratch_20260519_030042.json`
