# P81: Native CUDA Architecture Policy

Date: 2026-05-16

## Summary

P80 proved that R6000 can execute CuTe E2M1 FP4 MMA when the extension is built
for `sm_120a`.

P81 wires that target into the shared Lynn native CUDA loader as an opt-in /
auto policy.

## Code

The shared loader now exposes:

```python
native_cuda_arch_flags()
native_cuda_extra_cuda_cflags()
```

Default behavior remains unchanged:

```text
extra_cuda_cflags = ["-O3", "--use_fast_math"]
```

Opt-in behavior:

```bash
export LYNN_NATIVE_CUDA_ARCH=sm_120a
```

Auto behavior:

```bash
export LYNN_NATIVE_CUDA_ARCH_AUTO=1
```

On an R6000 reporting capability `(12, 0)`, auto injects:

```text
-arch=sm_120a
```

## Validation

Report:

```text
reports/p16_155/p81_native_cuda_arch_policy_probe.json
```

Observed on R6000:

| Case | Flags |
|---|---|
| default | `["-O3", "--use_fast_math"]` |
| `LYNN_NATIVE_CUDA_ARCH=sm_120a` | `["-O3", "--use_fast_math", "-arch=sm_120a"]` |
| `LYNN_NATIVE_CUDA_ARCH_AUTO=1` | `["-O3", "--use_fast_math", "-arch=sm_120a"]` |

## Decision

P81 passes. Future FP4 MMA kernels should use the shared native CUDA loader
instead of custom benchmark-local architecture flag hacks.

Default stays portable. `sm_120a` remains explicit until the grouped FP4 kernel
has full quality and serving gates.

