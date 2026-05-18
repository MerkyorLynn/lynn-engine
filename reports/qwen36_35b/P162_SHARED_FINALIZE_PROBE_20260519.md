# Qwen3.6-35B P162 Shared Finalize Probe

**Date:** 2026-05-19  
**Model:** official Qwen3.6-35B-A3B Lynn-native W4A16 NVFP4  
**Probe:** `benchmarks/p162_qwen36_shared_finalize_probe.py`

## Result

P162 isolates the shared-expert finalize tail:

```text
active + shared * sigmoid(linear(h, shared_gate))
```

It compares the current Torch path with the existing
`torch_scalar_add_triton` backend.

Artifact:
`reports/qwen36_35b/p162_qwen36_shared_finalize_probe_20260519_0520.json`

| Metric | Value |
|---|---:|
| Fixtures | 18 |
| Default finalize mean | 0.02931 ms |
| Torch gate + Triton add mean | 0.03426 ms |
| Split delta | +0.00495 ms |
| Triton add-only mean | 0.01511 ms |
| Split exact | 0 / 18 |
| max_abs_max | 0.015625 |
| Decision | `SHARED_FINALIZE_CLOSED_OR_FLAT` |

## Interpretation

The add-only Triton kernel is faster in isolation, but it is not mathematically
equivalent to the default finalize path. The full `torch_scalar_add_triton`
path is both slower and non-exact. This line should stay closed for default
serving.

Next exact boundary target is router/top-k/softmax allocation and output-buffer
behavior.
