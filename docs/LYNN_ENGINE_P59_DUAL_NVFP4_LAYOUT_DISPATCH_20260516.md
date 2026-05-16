# Lynn Engine P59: dual NVFP4 layout dispatch contract

Date: 2026-05-16
Branch: `codex/p16-r6000-155-tps`

## Why P59

The project now intentionally keeps two NVFP4 directions alive:

| Artifact | Purpose |
|---|---|
| **Lynn-native per-16 variable-expert NVFP4** | Main Lynn engine runtime, preserves physical expert pruning and per-16 scale contract |
| **Vendor-friendly ModelOpt / compressed-tensors NVFP4 v2** | Ecosystem route for SGLang/vLLM/TRT-style loaders and official NVIDIA kernels |

These two artifacts must coexist. The engine must never silently treat one as
the other.

## Code added

P59 adds metadata-only layout detection:

```text
engine/nvfp4_layout.py
benchmarks/p59_nvfp4_layout_dispatch_probe.py
```

The detector classifies a checkpoint before tensor loading:

```text
lynn_native_per16_variable
compressed_tensors_nvfp4
modelopt_nvfp4
packed_fp4_unknown
bf16_or_unquantized
unsupported
```

It reports:

- recommended loader;
- backend family;
- quantization config;
- Lynn manifest presence;
- variable-expert spec and HF-vanilla compatibility;
- suffix counts;
- warnings/blockers.

## Probe command

```bash
python benchmarks/p59_nvfp4_layout_dispatch_probe.py \
  --model /root/autodl-tmp/models/lynn-27b-variable-recovery-step5000-nvfp4-final \
  --model /root/autodl-tmp/models/lynn-27b-variable-recovery-step5000-bf16-final \
  --out reports/p16_155/p59_nvfp4_layout_dispatch_probe.json
```

Expected current result:

```text
Lynn-native NVFP4 -> lynn_native_per16_variable
BF16 final        -> bf16_or_unquantized
```

When a vendor-friendly v2 artifact exists, the same probe should classify it as
`compressed_tensors_nvfp4` or `modelopt_nvfp4`, not as Lynn-native.

## Decision

P59 is a small but important engineering boundary:

```text
one engine
two explicit NVFP4 layout families
zero silent cross-loading
```

This preserves Lynn-native advantages while leaving the official/vendor route
open. It also prevents the old class of bugs where packed FP4 tensor bytes are
accidentally cast to BF16 and treated as real weights.
