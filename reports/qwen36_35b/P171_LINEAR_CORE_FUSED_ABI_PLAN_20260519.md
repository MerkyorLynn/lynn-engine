# P171 Linear-Core Fused ABI Plan

Date: 2026-05-19

## Target

The next meaningful 35B speed candidate should fuse the first large
linear-attention boundary:

`in_proj -> conv -> recurrent/GDN`

Do not include `gated_norm/out_proj` in the first ABI. Those are smaller and
add RMSNorm/FP4 projection exactness risk before the largest surface is proven.

## Why

P168 measured the repeated 30-layer cost:

| Segment | Sum Across 30 Layers |
|---|---:|
| fused native FP4 in-proj | 2.107 ms/token |
| recurrent/GDN | 1.132 ms/token |
| conv update | 0.984 ms/token |

Together this boundary covers about 4.224 ms/token of hot work. P170 proved
that fixing only activation scratch saves about 0.194 ms/token and does not
promote, so the next candidate has to remove a larger launch/boundary surface.

## Candidate ABI

Suggested callable boundary:

```text
linear_core_inproj_conv_gdn_decode(
  h_norm,
  inproj_weight_t,
  inproj_scale_b,
  act_packed_scratch,
  scale_a_scratch,
  conv_state_in,
  conv_weight,
  recurrent_state_in,
  neg_exp_A_log,
  dt_bias,
  core_attn_out,
  z_out,
  conv_state_out,
  recurrent_state_out
)
```

Input tensors:

| Tensor | Shape | DType |
|---|---:|---|
| `h_norm` | `[1,1,2048]` | BF16 |
| `inproj_weight_t` | logical `[1024,12352]` | `float4_e2m1fn_x2` view |
| `inproj_scale_b` | swizzled FP8 scale_b | `float8_e4m3fn` |
| `act_packed_scratch` | `[1,1024]` | uint8 |
| `scale_a_scratch` | `[16384]` | `float8_e4m3fn`, initialized to ones |
| `conv_state_in` | `[1,8192,3]` | BF16 |
| `conv_weight` | `[8192,1,4]` | BF16 |
| `recurrent_state_in` | `[1,32,128,128]` | FP32 |
| `neg_exp_A_log` | `[32]` | FP32 |
| `dt_bias` | `[32]` | model dtype, cast exactly as current code |

Caller-owned outputs:

| Tensor | Shape | DType |
|---|---:|---|
| `core_attn_out` | `[1,1,32,128]` | BF16 |
| `z_out` | `[1,1,32,128]` | BF16 |
| `conv_state_out` | `[1,8192,3]` | BF16 |
| `recurrent_state_out` | `[1,32,128,128]` | FP32 |

## Admission Gate

Use P169 before any resident integration:

```bash
python benchmarks/p169_qwen36_linear_core_fixture_contract.py \
  --model /root/autodl-tmp/models/Qwen3.6-35B-A3B-lynn-native-w4a16-nvfp4-v0 \
  --fixtures /root/autodl-tmp/reports/qwen36_35b/p169_linear_core_fixtures_official_w4a16_20260519_0750 \
  --candidate-output-dir /root/autodl-tmp/reports/qwen36_35b/p171_linear_core_fused_candidate_outputs \
  --out /root/autodl-tmp/reports/qwen36_35b/p171_linear_core_fused_candidate_gate.json \
  --check
```

Minimum P169 admission:

- 20/20 fixtures exact.
- `max_abs_max == 0.0`.
- `conv_state_out` and `recurrent_state_out` exact, not only final output.

## Math Guardrails

Do not change:

- FP4 activation quantization, `scale_a/scale_b` layout, or `_scaled_mm` output
  dtype/rounding.
- Conv window order, padding, SiLU location, or state shift semantics.
- `g = neg_exp_A_log * softplus(a.float() + dt_bias.float())` cast order.
- `beta = sigmoid(b)` order.
- q/k l2norm eps, `1/sqrt(128)` scale, or GQA `head // 2` mapping.
- recurrent state accumulation dtype.
- in-place state aliasing before P169 proves it exact.

## Promotion Thresholds

Do not run full service gates unless P169 passes.

| Gate | Minimum |
|---|---:|
| P169 | 20/20 exact, max_abs 0 |
| P37 | exact true, median speedup >= 1.03x |
| P25 default | 512 decode >= 112 TPS preferred, 109 TPS absolute floor |
| structured default | 40/40 |
| AMBER | 70/70 structured and >=118 TPS |

The expected useful range is 0.6-1.2 ms/token saved. Anything near P170's
0.2 ms/token is not worth resident promotion.
