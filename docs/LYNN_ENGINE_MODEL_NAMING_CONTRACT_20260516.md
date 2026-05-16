# Lynn Engine Model Naming Contract - 2026-05-16

## Rule

Use **Lynn 27B A3B** for all new public-facing model names, handoff reports,
release aliases, benchmark tables, and README/model-card text.

The short `27B` label is no longer precise enough because the model is a
variable-pruned MoE with about 3B active parameters per token.  `27B A3B`
keeps the dense-parameter scale and active-parameter scale visible at the same
time, and avoids confusion with dense Qwen3.6 27B-family checkpoints.

## Recommended Aliases

Use these aliases for newly exported packages:

```text
lynn-27b-a3b-bf16-final
lynn-27b-a3b-nvfp4-final
lynn-27b-a3b-w4a8-nvfp4-v2
lynn-27b-a3b-w4a8-mtp-nvfp4
```

Legacy paths such as `lynn-27b-w4a8-nvfp4-v2` may remain as compatibility
symlinks while transfers and scripts are already in flight.  Do not rename an
active transfer directory in place; add an `a3b` symlink and update future
scripts to point at the canonical alias.

## Reporting Language

Prefer:

```text
Lynn 27B A3B W4A8 NVFP4
Lynn 27B A3B BF16 final
Lynn 27B A3B qwen3_next_mtp-style head
```

Avoid for new docs:

```text
Lynn 27B
27B NVFP4
27B W4A8
```

Historical documents can keep old labels when rewriting them would create
noise, but every new milestone should use the `27B A3B` form.
