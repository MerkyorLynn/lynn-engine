# Lynn Engine AWQ vs Lynn-Native NVFP4 Note - 2026-05-17

## Decision

Use the existing Spark `Qwen3.6-35B-A3B-AWQ` package as a quality and size
reference, not as the default promotion target. The promotion path remains:

```text
Official Qwen3.6-35B-A3B BF16
  -> Lynn-native W4A16 NVFP4
  -> official MTP sidecar
  -> R6000/GB10 native serving
```

## What Spark AWQ Is

The Spark AWQ package at:

```text
/home/merkyor/models/Qwen3.6-35B-A3B-AWQ
```

is a valid W4A16-style weight-only AWQ package:

- `name_or_path`: `tclf90/Qwen3.6-35B-A3B-AWQ`
- `quant_method`: `awq`
- `bits`: `4`
- `group_size`: `128`
- `version`: `gemm`
- `zero_point`: `true`
- 9 safetensors shards, about 24 GiB

It is not the canonical Qwen official BF16 package and it is not Lynn-native
NVFP4. It is useful because it gives an immediate W4A16 reference point for
disk size, loader behavior, and downstream quality checks.

## Why Not Promote AWQ Directly

AWQ is the safer compatibility baseline, but it is not aligned with the Lynn
155 TPS target:

- It uses AWQ/GEMM int4 layout, not Lynn's manifest-driven NVFP4 E2M1 layout.
- It does not directly exercise the Lynn variable-NVFP4 loader or packed FP4
  runtime path.
- Its `modules_to_not_convert` leaves `mtp`, `linear_attn`, `self_attn`,
  `shared_expert`, `mlp.gate`, and layer 0 outside AWQ conversion.
- It would require separate AWQ-specific loader/kernel work before it can be
  a first-class Lynn runtime target.

## Why Lynn-Native NVFP4 Still Matters

Lynn-native NVFP4 is the format that directly supports the current speed plan:

- manifest-driven loader that already handles Lynn naming/layout changes;
- native FP4 packed weight path for R6000/GB10;
- direct integration point for official 35B MTP sidecar probes;
- one runtime surface for W4A16 quality mode and W4A8 speed experiments;
- cleaner path for future C++/CUDA hot-kernel replacement.

## Practical Framing

AWQ can win as a quick usable artifact if quality is good and no native runtime
work is needed. Lynn-native NVFP4 is the candidate that can plausibly compound
with native kernels, active MoE routing, and MTP to chase the 155 TPS target.

So the matrix is:

| Candidate | Role |
| --- | --- |
| Existing AWQ 24G | W4A16 quality/size reference and fallback |
| Lynn-native W4A16 NVFP4 | Default promotion candidate |
| Lynn-native W4A8 NVFP4 | Speed experiment only until structured/tool-call gates hold |

