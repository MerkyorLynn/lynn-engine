# P169 Qwen3.6 Linear-Core Fixture Contract

Date: 2026-05-19

## Verdict

P169 creates the exact fixture target needed for the next 35B speed work:
`in_proj -> conv -> recurrent/GDN -> gated_norm -> out_proj`.

R6000 validation on first-of-block linear layers passed:

| Metric | Result |
|---|---:|
| Fixtures | 20 |
| Layers | 0,4,8,12,16,20,24,28,32,36 |
| Prompts | 2 |
| Passed | 20/20 |
| max_abs_max | 0.0 |
| cosine_min | 0.999999821 |

The cosine value is below 1.0 only because it is recomputed in FP32 over equal
tensors. The primary contract is bit exactness, and all checked tensors had
`max_abs == 0`.

## Contract Surface

Each fixture stores the real decode-time inputs and reference outputs for one
linear-attention layer:

- `h_norm`
- `recurrent_state_in`
- `conv_state_in`
- `proj_all`
- `out_conv`
- `conv_state_out`
- `core_attn_out`
- `recurrent_state_out`
- `gated_norm_out`
- `linear_core_out`

The self-check reloads the official W4A16 model, recomputes the same core, and
requires bit-exact equality for all checked tensors.

The contract also accepts `--candidate-output-dir`, where a candidate can mirror
fixture filenames and provide any subset of checked tensors. This lets fused
kernel experiments write only `linear_core_out`, `recurrent_state_out`, or
stage-level tensors and receive a fast pass/fail report before resident P37/P25.

## Why It Matters

P168 showed the linear/GDN core is the next large exact boundary:

- in-proj: 2.107 ms/token across 30 layers
- recurrent/GDN: 1.132 ms/token
- conv: 0.984 ms/token

P169 now gives candidate kernels a fast admission target before P37/P25. The
first natural candidate is caller-owned activation quant scratch for the fused
native FP4 in-proj, preserving the existing quant kernel and `torch._scaled_mm`
math while removing per-token scratch allocation/initialization.

## Artifacts

- `benchmarks/p169_qwen36_linear_core_fixture_contract.py`
- `scripts/r6000_qwen36_linear_core_fixture_gate.sh`
- `reports/qwen36_35b/p169_linear_core_fixture_contract_20260519_0750_v2.json`
- `reports/qwen36_35b/p169_linear_core_fixture_contract_20260519_0822_v3.json`

The full fixture tensor directory was left on R6000 under
`/root/autodl-tmp/reports/qwen36_35b/p169_linear_core_fixtures_official_w4a16_20260519_0750`
and is intentionally not committed because it contains large tensor artifacts.
