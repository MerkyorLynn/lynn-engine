# P172 Linear Core Candidate Diagnostics

Date: 2026-05-19

## Purpose

P172 is a read-only diagnostics scaffold for the P169 Qwen3.6-35B linear-core
fixture contract. It does not load the model, import engine modules, or require
GPU. It parses safetensors headers/data bytes directly to produce a per-tensor
shape/dtype/SHA256 manifest and, when a candidate output directory is supplied,
checks structural admission for future fused linear-core kernels.

## ABI Checks

Every fixture must contain:

- `h_norm`
- `conv_state_in`
- `recurrent_state_in`
- `linear_core_out`
- `conv_state_out`
- `recurrent_state_out`

P172 also records whether optional ABI tensors such as `z`, `core_attn`, and
`core_attn_out` are present.

For candidate-output-dir preflight, the default required output tensors are:

- `linear_core_out`
- `conv_state_out`
- `recurrent_state_out`

Shape/dtype compatibility is required for those candidate tensors. Byte-exact
hash equality against the fixture reference is reported by default and can be
made a hard gate with `REQUIRE_HASH_MATCH=1`.

## Artifacts

- Helper: `benchmarks/p172_qwen36_linear_core_candidate_diagnostics.py`
- R6000 wrapper: `scripts/r6000_qwen36_linear_core_candidate_diagnostics.sh`

Example:

```bash
FIXTURES=/root/autodl-tmp/reports/qwen36_35b/p169_linear_core_fixtures_official_w4a16_20260519_0750 \
CANDIDATE_OUTPUT_DIR=/root/autodl-tmp/reports/qwen36_35b/p171_linear_core_identity_candidate_20260519_080056_onlyfinal \
scripts/r6000_qwen36_linear_core_candidate_diagnostics.sh
```
