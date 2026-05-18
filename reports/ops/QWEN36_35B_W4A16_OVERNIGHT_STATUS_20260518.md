# Qwen3.6 35B W4A16 Overnight Status

Date: 2026-05-18 01:30 CST

## Route Decision

The default promotion route is now:

```text
official Qwen/Qwen3.6-35B-A3B
  -> Lynn-native W4A16 NVFP4
  -> quality-safe serving profile
```

W4A8 stays as a speed experiment. MTP stays as a gated accelerator: useful only
after iterative accept is real on the exact W4A16 runtime.

## Official 35B Quality Anchor

Spark evaluation on the official N5 package:

| Candidate | MMLU 500 5-shot | GPQA Diamond 198 0-shot |
|---|---:|---:|
| Qwen3.6-35B-A3B BF16 official | 86.40% | 45.45% |
| Qwen3.6-35B-A3B Q4_K_M-imatrix GGUF | 83.00% | 50.00% |
| Qwen3.6-35B-A3B Lynn-native W4A16 NVFP4 | 84.40% | 49.49% |
| Delta Q4_K_M vs BF16 | -3.40pp | +4.55pp |
| Delta W4A16 vs BF16 | -2.00pp | +4.04pp |

This validates the pivot away from custom 27B recovery as the primary quality
route. The model quality problem is no longer broad repair; it is preserving the
official model through native quantization and runtime optimization.

The Q4_K_M-imatrix row was completed by the Spark backstop at 2026-05-18
05:54 CST. The llama.cpp served-name string in the summary still says
`Lynn-27B-A3B-qwen36-q4km-imatrix`, but the backstop log records the actual
artifact as
`/home/merkyor/models/Qwen3.6-35B-A3B-GGUF-imatrix/Qwen3.6-35B-A3B-Q4_K_M-imatrix.gguf`
with a 20G quantized size, so this is the official Qwen3.6-35B-A3B GGUF result,
not a V4-Pro/Flash distill result.

The same Q4_K_M-imatrix artifact was also served through llama.cpp on Spark for
a P25 wall-TPS comparison. The 128-token request was polluted by first-request
startup, while the warm 256/512-token requests were stable:

| Q4_K_M-imatrix llama.cpp | Wall TPS |
|---:|---:|
| 128 tokens | 5.34, cold-start polluted |
| 256 tokens | 70.75 |
| 512 tokens | 71.58 |

This makes the current Lynn-native W4A16 R6000 service line meaningfully faster
than the Spark llama.cpp Q4_K_M reference on the same prompt family: conservative
default is about `107` decode TPS and the controlled structured fast mode is now
about `114` decode TPS.

## W4A16 Artifact

R6000 and Spark both produced the Lynn-native W4A16 NVFP4 package from the
official BF16 base.

| Machine | Path | Size | Quantized / kept |
|---|---|---:|---:|
| R6000 | `/root/autodl-tmp/models/Qwen3.6-35B-A3B-lynn-native-w4a16-nvfp4-v0` | 23G | 553 / 492 |
| Spark | `/home/merkyor/models/Qwen3.6-35B-A3B-lynn-native-w4a16-nvfp4-official-n5` | 23G | 553 / 492 |

Spark W4A16 MMLU/GPQA was run under the quality-safe profile:

```text
LYNN_PACKED_DECODE=0
LYNN_PACKED_SHARED_EXPERT=0
LYNN_LINEAR_BLOCK_GRAPH=0
LYNN_MOE_FAST_FIXED=0
```

Copied summaries:

- `reports/qwen36_35b/spark_qwen36_official_bf16_mmlu_n500_20260518.json`
- `reports/qwen36_35b/spark_qwen36_official_bf16_gpqa_20260518.json`
- `reports/qwen36_35b/spark_qwen36_official_q4km_imatrix_mmlu_n500_20260518.json`
- `reports/qwen36_35b/spark_qwen36_official_q4km_imatrix_gpqa_20260518.json`
- `reports/qwen36_35b/spark_qwen36_official_q4km_imatrix_backstop_20260518.log`
- `reports/qwen36_35b/spark_qwen36_official_q4km_imatrix_llamacpp_p25_20260518_124140.json`
- `reports/qwen36_35b/spark_qwen36_official_w4a16_nvfp4_mmlu_n500_20260518.json`
- `reports/qwen36_35b/spark_qwen36_official_w4a16_nvfp4_gpqa_20260518.json`

## R6000 Serving Snapshot

Quality-safe W4A16 server path:

| Max tokens | Wall TPS | Decode TPS |
|---:|---:|---:|
| 128 | 17.64 | 23.00 |
| 256 | 21.97 | 23.41 |

Fast graph path:

| Profile | Decode TPS | Status |
|---|---:|---|
| Graph reuse with default assign-state update | ~82 | Not promotable: long generation degenerates to repeated `!` |
| Graph reuse with `LYNN_LINEAR_STATE_UPDATE=inplace` | 81-82 | Promising: long Chinese server prompt is coherent |

Native FP4 `lm_head` is not the failure source. Direct runner A/B with
`LYNN_NATIVE_FP4_LM_HEAD=0/1` produced normal JSON and Chinese text. The current
speed blocker narrowed to graph-safe state ownership. The first fix is to make
linear block CUDA graph default to in-place recurrent/conv state updates, because
graph replay is address-bound and should not depend on Python dict tensor
replacement.

R6000 graph+in-place P25:

| Max tokens | Wall TPS | Decode TPS | Preview |
|---:|---:|---:|---|
| 128 | 52.51 | 81.32 | coherent Chinese technical text |
| 256 | 65.70 | 82.10 | coherent Chinese technical text |

Copied report:

- `reports/qwen36_35b/r6000_qwen36_w4a16_p25_graph_inplace_20260518.json`

Follow-up service-path gate on the same graph+in-place profile:

| Gate | Result |
|---|---:|
| P25 128-token wall / decode TPS | 51.66 / 81.23 |
| P25 256-token wall / decode TPS | 65.69 / 81.77 |
| P25 512-token wall / decode TPS | 72.76 / 81.79 |
| OpenAI structured gate | GREEN, 14/14 format-clean |
| Structured gate decode TPS | mean 82.05, min 80.04 |

This covers JSON, tool-call JSON, Python code, OpenAPI YAML, short Chinese
format constraints, and a numeric answer over the real OpenAI-compatible server
path. The graph+in-place profile is now a speed baseline candidate, not just a
runner-only optimization.

Copied reports:

- `reports/qwen36_35b/r6000_qwen36_w4a16_graph_structured_p25_20260518_013525.json`
- `reports/qwen36_35b/r6000_qwen36_w4a16_graph_structured_rerun_gate_20260518_013742.json`

Fast gate/up decode was then revalidated on the official 35B W4A16 package.
`LYNN_NATIVE_GATEUP_BACKEND=triton_fast_decode` keeps the same W4A16 quality
contract and changes only the E2M1 decode expression in the existing Triton
gate/up kernel.

| Probe | Result |
|---|---:|
| P53 gate/up micro, sampled layers | 1.069x mean speedup, max rel L2 0 |
| P37 generate parity | 3/3 exact, promote_default true |
| P37 median TPS | 100.43 -> 102.57 |
| P25 512-token wall / decode TPS | 76.58 / 86.60 |
| Structured OpenAI gate | GREEN, 14/14 format-clean |
| Structured gate decode TPS | mean 87.71, min 86.99 |

Copied reports:

- `reports/qwen36_35b/r6000_qwen36_w4a16_p53_lut_fast_decode_20260518_033404.json`
- `reports/qwen36_35b/r6000_qwen36_w4a16_p37_fast_decode_gate_20260518_033502.json`
- `reports/qwen36_35b/r6000_qwen36_w4a16_fastdecode_graph_p25_20260518_033637.json`
- `reports/qwen36_35b/r6000_qwen36_w4a16_fastdecode_graph_structured_gate_20260518_033637.json`

### Fastdecode + Triton Pair QK/RoPE

The current promoted fast profile adds `LYNN_QK_NORM_ROPE_BACKEND=triton_pair`
on top of graph reuse, in-place state update, and gate/up fastdecode.

| Probe | Result |
|---|---:|
| P27 layer 3 `attn.qk_norm_rope` | 0.361 ms -> 0.129 ms |
| P27 layer 31 `attn.qk_norm_rope` | 0.353 ms -> 0.136 ms |
| P27 layer 3 full decode | 1.027 ms -> 0.758 ms |
| P27 layer 31 full decode | 1.015 ms -> 0.788 ms |
| P26 full-attention layers | 4.08 -> 3.11 ms/token |
| P26 decode TPS | 85.48 -> 93.07 |
| P25 512-token wall / decode TPS | 83.38 / 95.88 |
| Structured OpenAI gate | GREEN, 14/14 format-clean |
| Structured gate decode TPS | mean 96.43, min 96.03 |

Copied reports:

- `reports/qwen36_35b/r6000_qwen36_w4a16_fastdecode_p26_phase_20260518_034111.json`
- `reports/qwen36_35b/r6000_qwen36_w4a16_fastdecode_p27_full_layer3_20260518_034254.json`
- `reports/qwen36_35b/r6000_qwen36_w4a16_fastdecode_p27_full_layer31_20260518_034254.json`
- `reports/qwen36_35b/r6000_qwen36_w4a16_fastdecode_p28_hybrid_block_20260518_034111.json`
- `reports/qwen36_35b/r6000_qwen36_w4a16_fastdecode_tritonpair_p27_full_layer3_20260518_034549.json`
- `reports/qwen36_35b/r6000_qwen36_w4a16_fastdecode_tritonpair_p27_full_layer31_20260518_034549.json`
- `reports/qwen36_35b/r6000_qwen36_w4a16_fastdecode_tritonpair_p26_phase_20260518_034713.json`
- `reports/qwen36_35b/r6000_qwen36_w4a16_fastdecode_tritonpair_p28_hybrid_block_20260518_034713.json`
- `reports/qwen36_35b/r6000_qwen36_w4a16_fastdecode_tritonpair_p25_20260518_034915.json`
- `reports/qwen36_35b/r6000_qwen36_w4a16_fastdecode_tritonpair_structured_gate_20260518_034915.json`

### Fastdecode + Triton Pair + Gated RMSNorm

The current promoted fast profile adds `LYNN_RMSNORM_GATED_BACKEND=triton` on
top of gate/up fastdecode and triton-pair QK/RoPE. This is the first official
35B W4A16 service profile above 100 decode TPS.

| Probe | Result |
|---|---:|
| P26 linear graph blocks | 7.10 -> 6.45 ms/token |
| P26 full-attention layers | 3.11 -> 3.09 ms/token |
| P26 decode TPS | 93.07 -> 99.18 |
| P25 128-token wall / decode TPS | 60.69 / 100.64 |
| P25 256-token wall / decode TPS | 78.64 / 102.11 |
| P25 512-token wall / decode TPS | 85.23 / 102.22 |
| Structured OpenAI gate | GREEN, 14/14 format-clean |
| Structured gate decode TPS | mean 102.61, min 101.84 |

Copied reports:

- `reports/qwen36_35b/r6000_qwen36_w4a16_fastdecode_tritonpair_rmsgated_p26_phase_20260518_040928.json`
- `reports/qwen36_35b/r6000_qwen36_w4a16_fastdecode_tritonpair_rmsgated_p28_hybrid_block_20260518_040928.json`
- `reports/qwen36_35b/r6000_qwen36_w4a16_fastdecode_tritonpair_rmsgated_p25_20260518_041200.json`
- `reports/qwen36_35b/r6000_qwen36_w4a16_fastdecode_tritonpair_rmsgated_structured_gate_20260518_041200.json`

### Linear-Attention Core After Gated RMSNorm

P10-C was rerun after the rmsgated promotion to identify the next safe linear
block target:

| Segment | Layer 0 | Layer 28 |
|---|---:|---:|
| Fused native FP4 in-proj | 0.074 ms | 0.071 ms |
| Recurrent fused prepare | 0.036 ms | 0.036 ms |
| Conv update | 0.032 ms | 0.032 ms |
| QKV split/repeat | 0.026 ms | 0.026 ms |
| Gated RMSNorm | 0.020 ms | 0.020 ms |
| Out projection BF16 | 0.014 ms | 0.016 ms |
| Full recomposed core | 0.332 ms | 0.320 ms |

This confirms the remaining linear-block work is not a single easy env switch.
The next safe target is a real fused linear-attention core boundary, with the
native FP4 in-proj still the largest isolated segment.

Copied reports:

- `reports/qwen36_35b/r6000_qwen36_w4a16_rmsgated_p10c_linear_layer0_20260518_042119.json`
- `reports/qwen36_35b/r6000_qwen36_w4a16_rmsgated_p10c_linear_layer28_20260518_042119.json`

### GQA Recurrent Promotion

`LYNN_LINEAR_ATTN_GQA_RECURRENT=1` is now part of the R6000 W4A16 fast service
profile. It preserves the Triton recurrent path but avoids materializing the
`q/k` repeat for grouped-query linear attention.

| Probe | Result |
|---|---:|
| P26 linear graph blocks | 6.45 -> 6.33 ms/token |
| P26 full-attention layers | 3.09 -> 3.06 ms/token |
| P26 decode TPS | 99.18 -> 100.80 |
| P25 128-token wall / decode TPS | 60.53 / 102.13 |
| P25 256-token wall / decode TPS | 79.27 / 103.16 |
| P25 512-token wall / decode TPS | 85.75 / 102.90 |
| Structured OpenAI gate | GREEN, 14/14 format-clean |
| Structured gate decode TPS | mean 103.17, min 100.70 |

This is a small but quality-safe gain. It does not change the 155 TPS gap shape:
linear blocks still dominate, so the next meaningful speed work remains fused
linear/MoE boundaries rather than service-loop rewrites.
Follow-up P10-C still shows the fused native FP4 in-proj as the largest isolated
linear-core segment at about 0.080 ms/layer, with recurrent and conv each around
0.033-0.036 ms.

P10-C was then aligned with the promoted GQA recurrent path and rerun with a
cached fused-FP4 weight transpose. This is not a large runtime lever, but it
removes a stale measurement artifact: q/k split-repeat is now only about
`0.007 ms/layer`, so the next fused boundary should target the native FP4
in-proj plus the conv/recurrent setup rather than repeat materialization.

| GQA P10-C Segment | Layer 0 | Layer 28 |
|---|---:|---:|
| Fused native FP4 in-proj | 0.0767 ms | 0.0786 ms |
| Recurrent fused prepare | 0.0364 ms | 0.0363 ms |
| Conv update | 0.0328 ms | 0.0327 ms |
| Gated RMSNorm | 0.0200 ms | 0.0199 ms |
| QKV split/no-repeat | 0.0072 ms | 0.0072 ms |
| Full recomposed core | 0.3111 ms | 0.3101 ms |

### Triton Conv Promotion

`LYNN_LINEAR_ATTN_CONV_BACKEND=triton_torch_silu` is now the quality-safe conv
decode path. The pure Triton SILU variants were faster locally but failed the
P37 greedy gate. The promoted variant fuses `cat + depthwise conv + conv-state
shift` in Triton, then keeps SILU on the Torch BF16 path; that preserves greedy
tokens while still removing the expensive grouped `conv1d` call.

| Probe | Result |
|---|---:|
| P10-C torch conv, layer 0 / 28 | 0.0332 / 0.0325 ms |
| P10-C Triton conv, layer 0 / 28 | 0.0260 / 0.0259 ms |
| P10-C Triton-inplace conv, layer 0 / 28 | 0.0241 / 0.0240 ms |
| P37 `triton_torch_silu` greedy parity | 3/3 exact |
| P37 median decode TPS | 102.94 -> 106.24 |
| P37 median speedup | 1.032x |
| P25 128-token wall / decode TPS | 54.67 / 102.74 |
| P25 256-token wall / decode TPS | 79.83 / 104.71 |
| P25 512-token wall / decode TPS | 86.95 / 104.73 |
| Structured OpenAI gate | GREEN, 14/14 format-clean |
| Structured gate decode TPS | mean 104.76, min 103.87 |

The server watch script also now pins
`LYNN_LINEAR_ATTN_RECURRENT_BACKEND=triton_fused_prepare` and
`LYNN_LINEAR_ATTN_RECURRENT_INPLACE=1` explicitly. A manual service rerun that
omitted those envs fell back to the torch recurrent path and dropped to about
90 decode TPS, so this is a reproducibility fix as well as a conv promotion.

### Fast Dispatch Pin

P36 was rerun after the GQA recurrent promotion to make sure the runner-fixed
decode dispatch remains quality-safe on the current baseline. It was exact on
all 3 greedy prompts and gave a small median gain, so the server watch script
now pins `LYNN_DECODE_FAST_DISPATCH=1` explicitly instead of relying only on the
runner default.

| Probe | Legacy dispatch | Fast dispatch |
|---|---:|---:|
| Exact greedy match | 3/3 | 3/3 |
| Mean decode TPS | 100.65 | 101.52 |
| Median decode TPS | 101.06 | 101.79 |
| Median speedup | - | 1.007x |

Copied reports:

- `reports/qwen36_35b/r6000_qwen36_w4a16_gqa_recurrent_p26_phase_20260518_052206.json`
- `reports/qwen36_35b/r6000_qwen36_w4a16_gqa_recurrent_p28_hybrid_block_20260518_052206.json`
- `reports/qwen36_35b/r6000_qwen36_w4a16_gqa_recurrent_p25_server_20260518_052455.json`
- `reports/qwen36_35b/r6000_qwen36_w4a16_gqa_recurrent_openai_structured_gate_20260518_052455.json`
- `reports/qwen36_35b/p10c_gqa_fused_cache_layer0_20260518_072610.json`
- `reports/qwen36_35b/p10c_gqa_fused_cache_layer28_20260518_072610.json`
- `reports/qwen36_35b/p10c_conv_torch_layer0_20260518_075306.json`
- `reports/qwen36_35b/p10c_conv_torch_layer28_20260518_075306.json`
- `reports/qwen36_35b/p10c_conv_triton_layer0_20260518_075306.json`
- `reports/qwen36_35b/p10c_conv_triton_layer28_20260518_075306.json`
- `reports/qwen36_35b/p10c_conv_triton_inplace_layer0_20260518_075306.json`
- `reports/qwen36_35b/p10c_conv_triton_inplace_layer28_20260518_075306.json`
- `reports/qwen36_35b/p37_conv_triton_gate_20260518_075613.json`
- `reports/qwen36_35b/p37_conv_triton_inplace_gate_20260518_075512.json`
- `reports/qwen36_35b/p37_conv_triton_torch_silu_gate_20260518_075752.json`
- `reports/qwen36_35b/p37_conv_triton_torch_silu_inplace_gate_20260518_075752.json`
- `reports/qwen36_35b/p25_server_conv_triton_recurrent_20260518_080129.json`
- `reports/qwen36_35b/structured_gate_conv_triton_recurrent_20260518_080129.json`
- `reports/qwen36_35b/p36_fast_dispatch_gqa_20260518_072216.json`
- `reports/qwen36_35b/r6000_qwen36_w4a16_gqa_recurrent_p10c_linear_layer0_20260518_052949.json`
- `reports/qwen36_35b/r6000_qwen36_w4a16_gqa_recurrent_p10c_linear_layer28_20260518_052949.json`

### Post-Conv Promoted Bottleneck Profile

The current promoted profile was reprofiled after the Triton conv promotion,
with the fast service env pinned explicitly:

```text
LYNN_NATIVE_GATEUP_BACKEND=triton_fast_decode
LYNN_QK_NORM_ROPE_BACKEND=triton_pair
LYNN_RMSNORM_GATED_BACKEND=triton
LYNN_LINEAR_ATTN_GQA_RECURRENT=1
LYNN_LINEAR_ATTN_RECURRENT_BACKEND=triton_fused_prepare
LYNN_LINEAR_ATTN_RECURRENT_INPLACE=1
LYNN_LINEAR_ATTN_CONV_BACKEND=triton_torch_silu
LYNN_DECODE_FAST_DISPATCH=1
```

| P26 phase | Mean |
|---|---:|
| Decode TPS from profiled wall | 102.29 |
| Wall | 9.776 ms/token |
| Linear graph blocks | 6.168 ms/token |
| Full-attention layers | 3.083 ms/token |
| Norm + native FP4 lm_head | 0.331 ms/token |
| Host gap | 0.156 ms/token |

P28 confirms the hot path is still broadly uniform rather than one bad layer:
the 10 linear graph blocks sum to `6.177 ms/token`, while the 10 full-attention
layers sum to `3.197 ms/token`. The hottest linear blocks are layers 4 and 0 at
about `0.634 ms`, followed by the remaining linear blocks around
`0.612-0.618 ms`.

P38/P39 MoE profiling on sampled linear layers now shows:

| MoE segment | Mean ms/layer |
|---|---:|
| Router top-k | 0.038 |
| Active packed experts | 0.113 |
| Active combined | 0.124 |
| Shared BF16 expert | 0.061 |
| Current full MoE | 0.200 |

P38 was then expanded from 6 sampled linear layers to all 30 linear-attention
layers. The full-layer profile confirms the bottleneck is uniform, not a single
bad layer:

| All-linear MoE segment | Mean ms/layer | Approx ms/token across 30 layers |
|---|---:|---:|
| Router top-k | 0.0379 | 1.14 |
| Active packed experts | 0.1212 | 3.64 |
| Shared BF16 expert | 0.0621 | 1.86 |
| Current full MoE | 0.2175 | 6.53 |

This means a selective one-layer patch will not move the service line much.
The next real speed step needs a reusable active/shared expert boundary fusion
that applies to all linear layers while preserving the existing Triton numerical
contract.

This keeps the next safe runtime target unchanged: remove kernel boundaries in
the routed/shared MoE path, then revisit full-attention fusion. The host gap is
only `0.156 ms/token`, so a C++ service-loop rewrite is not the current high-ROI
lever for the 155 TPS gap.

Copied reports:

- `reports/qwen36_35b/p26_phase_conv_promoted_20260518_082209.json`
- `reports/qwen36_35b/p28_hybrid_conv_promoted_20260518_082209.json`
- `reports/qwen36_35b/p38_moe_conv_promoted_20260518_082209.json`
- `reports/qwen36_35b/p39_active_moe_conv_promoted_20260518_082417.json`
- `reports/qwen36_35b/p38_moe_all_linear_20260518_092201.json`

### Shared-Gate Triton AMBER Branch

`LYNN_SHARED_EXPERT_GATE_BACKEND=triton` fuses the shared-expert scalar gate
application into one Triton kernel:

```text
shared * sigmoid(hidden @ shared_expert_gate.T)
```

It is a useful speed signal, but not a default promotion yet.

| Probe | Result |
|---|---:|
| P43 fused shared expert, torch gate | 0.0550 ms/layer |
| P43 fused shared expert, Triton gate | 0.0504 ms/layer |
| P37 greedy exact match | RED, 0/3 exact |
| P37 median decode TPS | 104.81 -> 109.69 |
| P25 128-token wall / decode TPS | 56.79 / 107.64 |
| P25 256-token wall / decode TPS | 82.79 / 109.35 |
| P25 512-token wall / decode TPS | 93.95 / 109.19 |
| Structured OpenAI gate | GREEN, 14/14 format-clean |
| Structured gate decode TPS | mean 108.52, min 102.54 |

The P37 failures are not punctuation collapse; they are normal-looking greedy
text divergences from the different scalar-gate reduction order. Keep this as
an AMBER speed branch until a longer structured/code/tool-call and benchmark
quality sweep says the drift is acceptable.

Two AMBER conv combinations were then checked over the OpenAI structured gate:

| AMBER profile | P25 512 wall / decode TPS | Structured gate |
|---|---:|---:|
| shared-gate Triton + conv Triton | 87.72 / 109.13 | GREEN, 14/14, mean 109.70 TPS |
| shared-gate Triton + conv Triton inplace | 88.38 / 110.04 | GREEN, 14/14, mean 110.73 TPS |

These profiles are useful for speed research and controlled local serving, but
they still inherit the P37 exact-greedy drift from both the shared-gate and pure
Triton conv reorderings. Default promote remains the stricter
`triton_torch_silu` conv plus Torch shared gate path.

A longer 70-request structured gate on the fastest AMBER profile
(`shared-gate Triton + conv Triton inplace`) stayed format-clean:

| Probe | Result |
|---|---:|
| Structured requests | GREEN, 70/70 |
| Mean decode TPS | 110.38 |
| Min / max decode TPS | 101.25 / 111.41 |

The follow-up hard structured gate then used a stricter prompt set covering
nested JSON tool args, JSON Schema, JSON-only function args, Python palindrome,
Python `normalize_city`, OpenAPI YAML, Chinese prefix constraints, number-only
answers, and JSON Patch arrays. This also passed:

| Probe | Result |
|---|---:|
| Hard structured requests | GREEN, 40/40 |
| Mean decode TPS | 114.18 |
| Min / max decode TPS | 110.36 / 115.22 |

The same AMBER profile with RoPE-cache prewarm also improves long-output P25:

| Max tokens | Wall TPS | Decode TPS |
|---:|---:|---:|
| 128 | 56.82 | 111.83 |
| 256 | 82.18 | 112.29 |
| 512 | 95.51 | 113.20 |

P26/P28 on this profile shows the remaining budget is still GPU kernel work:

| AMBER+RoPE-cache P26 phase | Mean ms/token |
|---|---:|
| Linear-attention graph blocks | 5.80 |
| Full-attention layers | 2.72 |
| Norm + native FP4 lm_head | 0.36 |
| Host gap | 0.16 |
| Total wall | 9.07 |
| Decode TPS from wall | 110.22 |

P28 layer timing keeps pointing at the 10 linear blocks: layer 4 is about
`0.596 ms`, layer 0 about `0.595 ms`, and the other linear blocks cluster around
`0.578-0.588 ms`. Full-attention layers mostly sit around `0.26 ms` median.

P38/P39 was then rerun under the same AMBER+RoPE-cache profile to check whether
the structured fast mode changed the MoE shape. It did not: the MoE cost is
still a uniform per-linear-layer budget, not a one-layer defect.

| AMBER+RoPE MoE segment | Mean ms/layer |
|---|---:|
| Router top-k | 0.036 |
| Active packed experts | 0.122 |
| Gate/up split | 0.031 |
| Down split | 0.025 |
| Shared BF16 expert | 0.059 |
| Current full MoE | 0.214 |

Approximate contribution across the 30 linear-attention layers is `6.43 ms`
per token for the full MoE body. That number is intentionally larger than the
P26 graph-block wall slice because P38/P39 isolate sampled layer calls outside
the CUDA graph replay envelope, but the directional conclusion is stable:
MoE boundary fusion is still the next kernel island, while Python/C++ service
loop work is still below the current ROI line.

Copied reports:

- `reports/qwen36_35b/p38_moe_amber_rope_20260518_125020.json`
- `reports/qwen36_35b/p39_active_moe_amber_rope_20260518_125020.json`

So this branch is viable as a controlled structured-serving fast mode and the
current best practical serving candidate. It still remains opt-in for
exact-parity-sensitive gates because exact greedy parity is known to drift.

`LYNN_SHARED_EXPERT_GATE_BACKEND=torch_inplace` was also checked as a safer
middle ground. It keeps Torch `F.linear + sigmoid` for the scalar gate and only
applies the final multiply in-place:

| Probe | Result |
|---|---:|
| P37 greedy exact match | GREEN, 3/3 exact |
| P37 median decode TPS | 104.15 -> 105.05 |
| P37 median speedup | 1.009x |
| P25 512-token wall / decode TPS | 87.12 / 104.81 |
| Structured OpenAI gate | GREEN, 14/14 format-clean |
| Structured gate decode TPS | mean 104.41, min 99.63 |

This is a safe opt-in fallback, but not a default promotion: service TPS is
effectively flat versus the current default P25 `104.73` decode TPS, and the
structured mean is slightly below the default `104.76` run.

`LYNN_MOE_ADD_SHARED_INPLACE=1` then checked the narrower MoE boundary idea of
adding active-MoE output and shared-expert output in-place while keeping all
router, gate/up, down, and shared-expert math unchanged:

| Probe | Result |
|---|---:|
| P37 greedy exact match | GREEN, 3/3 exact |
| P37 median decode TPS | 104.81 -> 105.25 |
| P37 median speedup | 1.004x |
| P25 512-token wall / decode TPS | 86.45 / 104.35 |

This is safe but not useful enough to promote. It confirms that pure allocation
cleanup at the final MoE add boundary is below the noise floor; the remaining
TPS gap needs real shared-expert or active-expert fusion, not only in-place
bookkeeping.

Combining `LYNN_SHARED_EXPERT_GATE_BACKEND=torch_inplace` with
`LYNN_MOE_ADD_SHARED_INPLACE=1` was also checked. The small P37 generate gate
looked positive (`3/3` exact, `1.006x` median speedup), but the real OpenAI
server P25 path regressed:

| Probe | Result |
|---|---:|
| P37 greedy exact match | GREEN, 3/3 exact |
| P37 median decode TPS | 104.37 -> 105.05 |
| P25 512-token wall / decode TPS | 81.46 / 99.04 |

This closes the in-place cleanup branch as a default-speed lever. P37 can catch
greedy drift, but service P25 remains the authority for promotion.

`LYNN_SHARED_EXPERT_GATE_BACKEND=torch_scalar_add_triton` was added as a
stricter shared-gate boundary probe: the scalar gate reduction stays on the
Torch path, while the final `active_moe + shared * gate` is fused in Triton.
This also is not promotable. It improves P37 median decode by only `1.011x`
(`103.57 -> 104.67` TPS) and fails exact-greedy parity, so P25 was intentionally
skipped.

The hard structured prompt set was also used for a one-runner MoE budget
frontier sweep. This checked whether reducing active top-k or skipping the
shared expert could be a pragmatic route to 155 TPS. It is not:

| MoE budget candidate | Median decode TPS | Exact / 10 | Min prefix |
|---|---:|---:|---:|
| top8 + shared, baseline | 107.98 | 10 | 22 |
| top6 + shared | 107.67 | 1 | 1 |
| top4 + shared | 109.18 | 1 | 1 |
| top1 + shared | 110.18 | 0 | 1 |
| top8, skip shared | 113.41 | 2 | 1 |
| top4, skip shared | 114.40 | 1 | 1 |
| top1, skip shared | 116.91 | 0 | 1 |

Even the most destructive budget cut is only `1.083x` over baseline and fails
from the first token on the hard structured set. This closes expert-dropping as
a 155 TPS route; the remaining path has to preserve the full top-8/shared MoE
contract and remove kernel boundaries or add a real accepted speculation path.

Copied reports:

- `reports/qwen36_35b/p43_shared_gate_torch_20260518_083212.json`
- `reports/qwen36_35b/p43_shared_gate_triton_20260518_083233.json`
- `reports/qwen36_35b/p37_shared_gate_triton_20260518_083325.json`
- `reports/qwen36_35b/p25_shared_gate_triton_20260518_083519.json`
- `reports/qwen36_35b/structured_gate_shared_gate_triton_20260518_083519.json`
- `reports/qwen36_35b/p25_amber_sharedgate_triton_20260518_085130.json`
- `reports/qwen36_35b/structured_gate_amber_sharedgate_triton_20260518_085130.json`
- `reports/qwen36_35b/p25_amber_sharedgate_triton_inplace_20260518_085210.json`
- `reports/qwen36_35b/structured_gate_amber_sharedgate_triton_inplace_20260518_085210.json`
- `reports/qwen36_35b/structured_gate_amber70_sharedgate_triton_convinplace_20260518_091420.json`
- `reports/qwen36_35b/structured_hardgate_amber_20260518_123744.json`
- `reports/qwen36_35b/r6000_qwen36_w4a16_p25_server_decode_tps_20260518_124449_amber_rope_p25.json`
- `reports/qwen36_35b/p26_amber_rope_profile_20260518_124620.json`
- `reports/qwen36_35b/p28_amber_rope_profile_20260518_124620.json`
- `reports/qwen36_35b/p37_shared_gate_torch_inplace_20260518_090215.json`
- `reports/qwen36_35b/p25_shared_gate_torch_inplace_20260518_091020.json`
- `reports/qwen36_35b/structured_gate_shared_gate_torch_inplace_20260518_091050.json`
- `reports/qwen36_35b/p37_moe_add_shared_inplace_20260518_091905.json`
- `reports/qwen36_35b/p25_moe_add_shared_inplace_20260518_092010.json`
- `reports/qwen36_35b/p37_moe_inplace_combo_20260518_092445.json`
- `reports/qwen36_35b/p25_moe_inplace_combo_20260518_092538.json`
- `reports/qwen36_35b/p37_shared_scalar_add_triton_20260518_125634_shared_scalar_add.json`
- `reports/qwen36_35b/r6000_qwen36_w4a16_moe_topk_budget_hard_20260518_125907_topk_budget_hard.json`

Negative probes from the same loop:

- Router Triton top-k is not a full-path win: sampled full router regressed from
  0.046 ms to 0.052 ms.
- `tl.dot` gate/up is not a win on layer 28: 0.080 ms versus 0.033 ms current.
- Active MoE tile sweep did not beat the current default enough to promote.
- `LYNN_FULL_ATTN_DECODE_BACKEND=manual_gqa` is slower than SDPA on the current
  rmsgated fast profile: P26 decode drops from 99.18 to 87.65 TPS, full-attn
  layers rise from 3.09 to 3.94 ms/token, and host gap rises from 0.16 to
  0.30 ms/token.
- `LYNN_FULL_ATTN_QKV_FUSED=1` is now available as an opt-in research switch,
  but it is not promotable: P37 median decode improves only `1.013x`
  (`104.33 -> 105.67` TPS) and exact-greedy parity is `0/3`. Concatenating the
  BF16 q/k/v projection rows changes matmul reduction behavior enough to move
  greedy text, so full-attention speed needs a numerically stricter kernel path
  instead of this quick row-fusion shortcut.

Copied reports:

- `reports/qwen36_35b/p37_fullattn_qkv_fused_20260518_085919.json`
- `reports/qwen36_35b/r6000_qwen36_w4a16_manualgqa_p26_phase_20260518_041631.json`
- `reports/qwen36_35b/r6000_qwen36_w4a16_manualgqa_p28_hybrid_block_20260518_041631.json`

### Native Kernel Island Readiness

The next target is a Lynn-owned grouped active/shared MoE kernel, not another
Python service-loop rewrite. Toolchain and ABI gates are now checked on R6000:

| Probe | Result |
|---|---:|
| P76 CUTLASS/CuTe SM120 smoke | GREEN, headers found, compile ok |
| P70 grouped-per16 fused ABI | GREEN, shape/layout checks pass and fail-loud replacement point is reserved |
| P38 current full MoE | 0.213 ms/layer sampled mean |
| P39 router top-k | 0.038 ms/layer sampled mean |
| P39 gate/up fastdecode | 0.031 ms/layer sampled mean |
| P39 down | 0.026 ms/layer sampled mean |
| P39 shared BF16 | 0.060 ms/layer sampled mean |

P97 interval decomposition on layer 28 shows there is real native-down speed in
the kernel island, but the clean contract is currently in the quantized
activation reference domain, not the W4A16 BF16-activation serving contract:

| Variant | Gate ms | Down ms | Total ms | Speedup |
|---|---:|---:|---:|---:|
| Triton gate/up + Triton down | 0.0569 | 0.0328 | 0.0901 | 1.00x |
| P93 split16 gate/up + Triton down | 0.0582 | 0.0266 | 0.0848 | 1.06x |
| P93 split16 gate/up + native down tile1 | 0.0578 | 0.0225 | 0.0803 | 1.12x |

The same P97 probe was repeated on seven linear-attention layers. All layers
passed the quantized-activation contract, and `native_down_tile1` stayed the
best interval variant:

| Layer | Triton total ms | Tile total ms | Speedup |
|---:|---:|---:|---:|
| 0 | 0.0879 | 0.0784 | 1.12x |
| 8 | 0.0908 | 0.0878 | 1.03x |
| 16 | 0.0882 | 0.0803 | 1.10x |
| 24 | 0.0922 | 0.0841 | 1.10x |
| 28 | 0.0885 | 0.0804 | 1.10x |
| 32 | 0.0907 | 0.0801 | 1.13x |
| 36 | 0.0899 | 0.0809 | 1.11x |
| Mean | 0.0897 | 0.0817 | 1.10x |

That confirms the next native kernel should borrow the down-tile lesson while
preserving the W4A16 numerical contract. Directly promoting the quantized
activation composition would repeat the W4A8 quality failure mode.

Runtime gate with `LYNN_NATIVE_DOWN_TILE_HIDDEN=1` confirms that conclusion:
the candidate reaches 111.08 median decode TPS, but all three greedy prompts
diverge into repeated exclamation marks. Tile size is not the quality fix.
Follow-up isolation narrows the failure mode:

- P49 true decode-state local down comparison is GREEN: tile1 is 1.25x faster
  than Triton down, with max rel L2 `4.7e-6` and cosine `1.0`.
- P50 first-divergence with linear-block graphs disabled keeps top-1 logits
  aligned for eight reference-fed decode steps, although hidden drift first
  appears at layer 15.
- P37 graph-off generation still fails exact greedy parity on all three prompts,
  but it no longer collapses into repeated punctuation. The remaining issue is
  long-run greedy sensitivity from tiny down-order differences, not an obviously
  broken down tile kernel.
- Single-layer graph-on allowlist is also closed: layers 0/8/16/24/28/32/36 all
  fail 3/3 greedy parity with `LYNN_NATIVE_DOWN_BACKEND=cuda_tile` and
  `LYNN_NATIVE_DOWN_TILE_HIDDEN=1`, while speed is flat to negative
  (`0.99x-1.004x`). There is no safe selective down-tile promotion.
- Triton scale-hoist is numerically clean but not a speed lever: P53 keeps
  active cosine at `1.0`, but down slows from `0.025 ms` to `0.078 ms` and
  combined active MoE falls to `0.16x`. Do not add it as a runtime backend.

This confirms the remaining MoE opportunity is mostly boundary fusion: one
kernel boundary for routed active experts and, later, a shared-expert fusion.
The individual gate/up and down kernels are already small enough that more tile
sweeps are low ROI unless they remove launches, intermediate tensors, or
separate scheduling boundaries.

Two tempting native tile runtime candidates are explicitly blocked by generation
quality:

| Candidate | Median Decode TPS | Speedup | Gate |
|---|---:|---:|---|
| `LYNN_NATIVE_DOWN_BACKEND=cuda_tile` | 108.58 | 1.07x | RED, 0/3 greedy IDs match |
| `LYNN_NATIVE_ACTIVE_MOE_BACKEND=grouped_per16_nonatomic` | 124.30 | 1.23x | RED, 0/3 greedy IDs match |

Both candidates collapse the three P37 prompts into repeated exclamation marks.
Keep them as kernel signals only. The production path needs a fused kernel that
preserves the Triton numerical contract, not a direct promotion of the old
scalar/native tile references.

The grouped-per16 active-MoE signal was then checked with a stricter
first-divergence harness. P33 now supports arbitrary candidate backends and
native-active-MoE layer allowlists, so the same reference-fed prompt can compare
full linear-attention, coarse layer groups, and individual layers against the
Triton baseline.

| P33 grouped-per16 scope | Result |
|---|---|
| All linear-attention layers | RED: first top-1 divergence at step 2, `318 -> 1393`; first hidden drift at layer 15 |
| Layers `0,1,2,4,5,6,8,9,10,12,13,14` | RED: first top-1 divergence at step 2, `318 -> 1393` |
| Layers `16,17,18,20,21,22` | RED: first top-1 divergence at step 2, `318 -> 1393` |
| Layers `24,25,26,28,29,30` | RED: first top-1 divergence at step 2, `318 -> 1393` |
| Layers `32,33,34,36,37,38` | RED: first top-1 divergence at step 2, `318 -> 1393` |
| Single layer `0` | RED: first top-1 divergence at step 2, `318 -> 1393` |
| Single layer `16` | P33 GREEN over this one prompt, but hidden drift appears later at layer 19 |
| Single layer `24` | RED: first top-1 divergence at step 2, `318 -> 1393` |
| Single layer `32` | P33 GREEN over this one prompt |

The apparent safe single-layer windows are not promotable. A follow-up P37
multi-prompt generation gate with grouped-per16 enabled only on layers `16,32`
improved median decode TPS by `1.017x` (`104.42 -> 106.18` median, `104.11 ->
106.21` mean), but all three prompts diverged and collapsed to repeated `!`
tokens after the first newline. This closes the simple layer-allowlist rescue.
The grouped-per16 kernel remains a useful speed ceiling signal, but promotion
needs a numerically stricter active-MoE implementation rather than choosing a
subset of layers.

A final MoE block-shape sweep checked whether the current Triton path had an
easy safe retune hiding behind `LYNN_MOE_FAST_FIXED`. It does not:

| Candidate | P37 exact | P37 median TPS | Service/P25 result |
|---|---:|---:|---|
| `LYNN_MOE_FAST_FIXED=0` | GREEN, 3/3 | `104.51 -> 105.13` | Not promoted: P25 512 decode TPS `103.88`, below current default |
| `FAST_FIXED=0`, gate hidden block `64` | RED | `104.48 -> 98.42` | Closed |
| `FAST_FIXED=0`, gate inter block `16` | RED | `103.64 -> 90.07` | Closed |
| `FAST_FIXED=0`, down hidden block `16` | RED | `104.27 -> 99.03` | Closed |
| `FAST_FIXED=0`, down inter block `256` | RED | `104.43 -> 104.35` | Closed |
| `FAST_FIXED=0`, gate inter `16`, hidden `128` | RED | `104.23 -> 91.53` | Closed |

The service result is the authority here. Even the only exact P37 candidate is
flat-to-negative on P25, so the default stays `LYNN_MOE_FAST_FIXED=1` with the
current block shape. The next MoE work must change the kernel boundary, not just
retune the existing Triton block sizes.

Copied reports:

- `reports/qwen36_35b/r6000_qwen36_w4a16_p76_cutlass_cute_20260518_035340.json`
- `reports/qwen36_35b/r6000_qwen36_w4a16_p70_grouped_fused_guard_20260518_035519.json`
- `reports/qwen36_35b/r6000_qwen36_w4a16_p38_moe_multilayer_fastdecode_20260518_035645.json`
- `reports/qwen36_35b/r6000_qwen36_w4a16_p39_active_moe_inner_fastdecode_20260518_035645.json`
- `reports/qwen36_35b/r6000_qwen36_w4a16_p74_active_moe_budget_fastdecode_20260518_040044.json`
- `reports/qwen36_35b/r6000_qwen36_w4a16_p97_active_moe_interval_layer28_20260518_053300.json`
- `reports/qwen36_35b/r6000_qwen36_w4a16_p97_active_moe_interval_layer0_20260518_053622.json`
- `reports/qwen36_35b/r6000_qwen36_w4a16_p97_active_moe_interval_layer8_20260518_053622.json`
- `reports/qwen36_35b/r6000_qwen36_w4a16_p97_active_moe_interval_layer16_20260518_053622.json`
- `reports/qwen36_35b/r6000_qwen36_w4a16_p97_active_moe_interval_layer24_20260518_053622.json`
- `reports/qwen36_35b/r6000_qwen36_w4a16_p97_active_moe_interval_layer28_20260518_053622.json`
- `reports/qwen36_35b/r6000_qwen36_w4a16_p97_active_moe_interval_layer32_20260518_053622.json`
- `reports/qwen36_35b/r6000_qwen36_w4a16_p97_active_moe_interval_layer36_20260518_053622.json`
- `reports/qwen36_35b/r6000_qwen36_w4a16_p37_down_cuda_tile1_gqa_gate_20260518_055325.json`
- `reports/qwen36_35b/r6000_qwen36_w4a16_p49_down_tile1_true_decode_20260518_055633.json`
- `reports/qwen36_35b/r6000_qwen36_w4a16_p50_down_tile1_first_divergence_20260518_055633.json`
- `reports/qwen36_35b/r6000_qwen36_w4a16_p37_down_cuda_tile1_graphoff_gate_20260518_055939.json`
- `reports/qwen36_35b/r6000_qwen36_w4a16_p37_down_tile1_layer0_20260518_062756.json`
- `reports/qwen36_35b/r6000_qwen36_w4a16_p37_down_tile1_layer8_20260518_062756.json`
- `reports/qwen36_35b/r6000_qwen36_w4a16_p37_down_tile1_layer16_20260518_062756.json`
- `reports/qwen36_35b/r6000_qwen36_w4a16_p37_down_tile1_layer24_20260518_062756.json`
- `reports/qwen36_35b/r6000_qwen36_w4a16_p37_down_tile1_layer28_20260518_062756.json`
- `reports/qwen36_35b/r6000_qwen36_w4a16_p37_down_tile1_layer32_20260518_062756.json`
- `reports/qwen36_35b/r6000_qwen36_w4a16_p37_down_tile1_layer36_20260518_062756.json`
- `reports/qwen36_35b/r6000_qwen36_w4a16_p53_triton_scale_hoist_20260518_062117.json`
- `reports/qwen36_35b/r6000_qwen36_w4a16_p37_down_cuda_tile_gate_20260518_040259.json`
- `reports/qwen36_35b/r6000_qwen36_w4a16_p37_grouped_per16_nonatomic_gate_20260518_040522.json`
- `reports/qwen36_35b/p33_grouped_per16_nonatomic_linear_divergence_20260518_113506.json`
- `reports/qwen36_35b/p33_grouped_per16_nonatomic_layers_0_1_2_4_5_6_8_9_10_12_13_14_20260518_113707.json`
- `reports/qwen36_35b/p33_grouped_per16_nonatomic_layers_16_17_18_20_21_22_20260518_113728.json`
- `reports/qwen36_35b/p33_grouped_per16_nonatomic_layers_24_25_26_28_29_30_20260518_113750.json`
- `reports/qwen36_35b/p33_grouped_per16_nonatomic_layers_32_33_34_36_37_38_20260518_113812.json`
- `reports/qwen36_35b/p33_grouped_per16_nonatomic_layer_0_20260518_113849.json`
- `reports/qwen36_35b/p33_grouped_per16_nonatomic_layer_16_20260518_113911.json`
- `reports/qwen36_35b/p33_grouped_per16_nonatomic_layer_24_20260518_113932.json`
- `reports/qwen36_35b/p33_grouped_per16_nonatomic_layer_32_20260518_113953.json`
- `reports/qwen36_35b/p37_grouped_per16_nonatomic_layers16_32_20260518_114034.json`
- `reports/qwen36_35b/p37_moe_block_ff0_20260518_115209.json`
- `reports/qwen36_35b/p37_moe_block_ff0_gh64_20260518_115209.json`
- `reports/qwen36_35b/p37_moe_block_ff0_gi16_20260518_115209.json`
- `reports/qwen36_35b/p37_moe_block_ff0_dh16_20260518_115209.json`
- `reports/qwen36_35b/p37_moe_block_ff0_di256_20260518_115209.json`
- `reports/qwen36_35b/p37_moe_block_ff0_gi16_gh128_20260518_115209.json`
- `reports/qwen36_35b/r6000_qwen36_w4a16_p25_server_decode_tps_ff0_p25_20260518_115717.json`

### Full-Attention RoPE Table Cache

`LYNN_FULL_ATTN_ROPE_CACHE=1` is now part of the R6000 W4A16 fast service
profile. The cache precomputes full-attention RoPE cos/sin tables once per
device, dtype, rotary dimension, theta, and max sequence length, then gathers by
GPU position tensor during decode. This removes per-full-attention-layer
`arange -> inv_freq -> cos/sin` work without changing the QK/RoPE math.

The first probe was exact but exposed a serving startup detail: building the
table during the first decode request hurts short-request P25. The implementation
was then wired into prefill as well, so the existing `LYNN_PREFILL_WARMUP=1`
path builds the table before the measured decode requests.

| Probe | Result |
|---|---:|
| P37 RoPE cache, decode-only parity | GREEN, 3/3 exact |
| P37 median TPS before prewarm | `104.74 -> 106.76` |
| P37 median TPS after prewarm | `104.05 -> 108.13` |
| P25 128-token wall / decode TPS | 56.39 / 106.46 |
| P25 256-token wall / decode TPS | 81.61 / 107.54 |
| P25 512-token wall / decode TPS | 88.91 / 107.31 |
| Structured OpenAI gate | GREEN, 14/14 format-clean |
| Structured gate decode TPS | mean 107.43, min 103.52 |

This is the new safe default speed line. It is a meaningful incremental win over
the previous 104-105 decode TPS default, but it does not change the remaining
shape of the problem: 155 TPS still needs a strict MoE/native kernel island,
full-attention core work, or a real accepted speculation path.

Copied reports:

- `reports/qwen36_35b/p37_fullattn_rope_cache_20260518_120416.json`
- `reports/qwen36_35b/r6000_qwen36_w4a16_p25_server_decode_tps_rope_cache_p25_20260518_120512.json`
- `reports/qwen36_35b/p37_fullattn_rope_cache_prewarm_20260518_120655.json`
- `reports/qwen36_35b/r6000_qwen36_w4a16_p25_server_decode_tps_rope_cache_prewarm_p25_20260518_120753.json`
- `reports/qwen36_35b/structured_gate_rope_cache_structured_20260518_121102.json`

Follow-up profiling on the new safe default shows the cache moved the bottleneck
back toward the linear/MoE body:

| P26 phase after RoPE cache | Mean ms/token |
|---|---:|
| Linear-attention graph blocks | 6.17 |
| Full-attention layers | 2.82 |
| Norm + native FP4 lm_head | 0.33 |
| Host gap | 0.15 |
| Total wall | 9.52 |
| Decode TPS from wall | 105.07 |

P28 layer timing confirms the 10 linear blocks are now the uniform hot path:
layer 0 averages `0.635 ms`, layer 4 averages `0.628 ms`, and the remaining
linear blocks sit around `0.613-0.614 ms`. Full-attention layers are mostly
around `0.27-0.28 ms`; the layer 3 mean is inflated by a one-off outlier, while
its median is `0.281 ms`.

Two tempting follow-up switches were then closed:

| Candidate | Result |
|---|---|
| `LYNN_PACKED_DECODE_FULL_ATTN=1` | RED: P37 0/3 exact, median TPS `106.54 -> 92.41`; do not open packed full-attn decode by default |
| `LYNN_FULL_ATTN_GATE_INPLACE=1` | P37 exact and +0.6%, but service P25 512 decode TPS `106.79` is below the RoPE-cache default `107.31`; not promoted |
| Merged top-k gate/up scheduling | Numerically clean but slower: reference gate/up is about `0.033 ms/layer`, best merged top-k is `0.068-0.070 ms/layer`; not a kernel path |

Copied reports:

- `reports/qwen36_35b/p26_rope_cache_profile_20260518_121508.json`
- `reports/qwen36_35b/p28_rope_cache_profile_20260518_121508.json`
- `reports/qwen36_35b/p37_packed_fullattn_rope_cache_20260518_121802.json`
- `reports/qwen36_35b/p37_fullattn_gate_inplace_rope_cache_20260518_122013.json`
- `reports/qwen36_35b/p25_gate_inplace_rope_cache_20260518_122134.json`
- `reports/qwen36_35b/structured_gate_gate_inplace_rope_cache_20260518_122134.json`
- `reports/qwen36_35b/p26_gateup_merged_topk_qwen36_20260518_122603.json`

## Speed Profile

R6000 P26 phase profile on graph+in-place narrows the 155 TPS gap:

| Phase | Mean ms/token |
|---|---:|
| Linear-attention graph blocks | 8.68 |
| Full-attention layers | 4.14 |
| Norm + native FP4 lm_head | 0.33 |
| Host gap | 0.14 |
| Total profiled wall | 13.33 |

The immediate bottleneck is GPU work, not Python/server overhead. To reach 155
TPS, the target is roughly 6.45 ms/token, so the next runtime work should focus
on linear-block replay and full-attention layer fusion before C++ service-loop
rewrites.

Supporting micro-profiles:

- Full-attention layer 31: `layer.full_decode` 1.02 ms, with attention decode
  0.53 ms, qk norm/RoPE 0.36 ms, and MoE 0.22 ms.
- Linear-attention layer 0 core: recomposed core 0.42 ms; top segments are gated
  RMSNorm 0.078 ms, native FP4 fused in-proj 0.074 ms, recurrent update 0.038 ms,
  conv update 0.033 ms.
- P28 hybrid block timing shows the hot path is uniform, not a single bad layer:
  10 linear graph blocks sum to 8.69 ms/token, while the 10 eager full-attention
  layers sum to 4.21 ms/token. Each linear block is about 0.866 ms.
- P38/P39 packed-MoE profiling across sampled linear layers shows current full
  packed MoE averages 0.205 ms/layer: router 0.037 ms, active packed experts
  0.113 ms, shared BF16 expert 0.061 ms. Across 30 linear layers, MoE alone is
  about 6.15 ms/token, roughly 71% of the linear-block budget.
- Rechecking `LYNN_PACKED_SHARED_EXPERT=1` on 35B confirms it is still not a
  promote lever: shared BF16 averages 0.061 ms/layer, packed scalar bridge
  averages 0.141 ms/layer, and native fast 2D averages 0.234 ms/layer with worse
  local cosine. Keep BF16 shared expert until there is a fused native shared
  kernel, not three separate packed calls.
- P26 upper-bound ablations show MoE work matters but is not sufficient alone:
  skipping shared reaches 82.67 TPS, skipping active reaches 91.65 TPS, and
  skipping all MoE reaches 102.49 TPS. Even making MoE free would not reach 155,
  so the route needs MoE fusion plus full-attention/linear-attention fusion or a
  real accepted speculation path.
- Full-attention graph slots are exact only under the same captured prompt state:
  P9-V passed strict logit parity at 11.18 ms/token, but P9-W cross-prompt reuse
  failed with graph next id 0 versus eager next id 248068. Do not promote
  reusable full-attention graph slots across requests; treat them as a
  per-request capture or future state-parametrized graph research branch.

Copied reports:

- `reports/qwen36_35b/r6000_qwen36_w4a16_graph_inplace_p26_phase_20260518_014002.json`
- `reports/qwen36_35b/r6000_qwen36_w4a16_graph_inplace_p27_full_layer31_20260518_014148.json`
- `reports/qwen36_35b/r6000_qwen36_w4a16_p10c_linear_layer0_20260518_014332.json`
- `reports/qwen36_35b/r6000_qwen36_w4a16_p28_hybrid_block_timing_20260518_024830.json`
- `reports/qwen36_35b/r6000_qwen36_w4a16_p38_moe_multilayer_20260518_025304.json`
- `reports/qwen36_35b/r6000_qwen36_w4a16_p39_active_moe_inner_20260518_025121.json`
- `reports/qwen36_35b/r6000_qwen36_w4a16_p38_moe_multilayer_packed_shared_20260518_031708.json`
- `reports/qwen36_35b/r6000_qwen36_w4a16_p26_skip_shared_upper_bound_20260518_031849.json`
- `reports/qwen36_35b/r6000_qwen36_w4a16_p26_skip_active_upper_bound_20260518_032051.json`
- `reports/qwen36_35b/r6000_qwen36_w4a16_p26_skip_all_moe_upper_bound_20260518_032228.json`
- `reports/qwen36_35b/r6000_qwen36_w4a16_p9v_real_state_full_attn_slots_20260518_014801.json`
- `reports/qwen36_35b/r6000_qwen36_w4a16_p9w_cross_prompt_full_attn_slots_20260518_021815.json`

Spark also now has the R6000 W4A16 artifact finalized at:

- `/home/merkyor/models/Qwen3.6-35B-A3B-lynn-native-w4a16-nvfp4-from-r6000`

## W4A8 Matrix

R6000 W4A8 comparison:

| Mode | Exact | Min prefix | Mean prefix | Decision |
|---|---:|---:|---:|---|
| Quality-safe gateup | 4/6 | 1 | 9.50 | Experimental only |
| Quality-safe full | 3/6 | 1 | 8.67 | RED |
| Graph+in-place gateup | 5/6 | 4 | 17.17 | Experimental only |
| Graph+in-place full | 5/6 | 4 | 18.83 | Still not default; one structured Chinese prompt paraphrased |

W4A16 remains the quality route. W4A8 should not be promoted until structured,
code, and tool-call parity holds.

Copied report:

- `reports/qwen36_35b/r6000_qwen36_w4a16_w4a8_matrix_graph_inplace_20260518.json`

## MTP Status

Official Qwen3.6-35B MTP sidecar on Lynn-native W4A16:

| Probe | Result |
|---|---|
| Shape audit | GREEN |
| Forward smoke | GREEN, finite logits |
| Iterative accept | RED, 0/24 accepted |
| P120 alignment sweep on W4A16 | RED, 0/15 top1 and 0/15 top8 |
| P120 alignment sweep on BF16 | RED, 0/9 top1 and 0/9 top8 |

Atlas/Qwen3.6 external framing is mixed and should not be treated as a free
multiplier. Atlas README pins 131 tok/s to Qwen3.5-35B-A3B MTP, while its public
site currently lists Qwen3.6-35B-A3B around 71 tok/s. Independent reports suggest
MTP can help Qwen3.6 MoE in some llama.cpp/IQ quant settings, but local Lynn
accept is currently zero. Treat MTP as an empirical branch until it passes our
gate.

The Qwen3.6 hybrid-SSM detail is now part of the gate. Atlas has shipped
Qwen3.6 support, but its public docs also warn that speculative decoding on
hybrid SSM models can be slower than regular decode, and public issue traffic
has reported Qwen3.6 MTP not engaging in some NVFP4 settings. That matches the
local result: shape and forward compatibility are not enough; Lynn must measure
accept rate and end-to-end TPS on the exact W4A16 runtime before counting MTP
credit.

P120 rules out the easiest "off by one" explanations for the official sidecar.
It sweeps current-token embedding versus oracle next-token embedding, position
offsets `-1/0/+1/+2`, and immediate-next versus next-after-current targets. The
best W4A16 variant reaches only `2/15` inside top32 and no top8/top1; the BF16
official base similarly stays at no top8/top1. This means W4A16 quantization is
not the reason official MTP is unusable locally. Treat the official 35B sidecar
as a warm-start/diagnostic asset, not a plug-and-play proposer.

Copied reports:

- `reports/qwen36_35b/p120_official_mtp_alignment_sweep_20260518_084820.json`
- `reports/qwen36_35b/p120_official_mtp_alignment_sweep_bf16base_20260518_084906.json`

References:

- <https://github.com/Avarok-Cybersecurity/atlas>
- <https://atlasinference.io/>
- <https://arxiv.org/abs/2605.01106>
- <https://njannasch.dev/blog/mtp-speculative-decoding-qwen-3-6-5060ti/>

## Benchmark Harness Hygiene

After promoting `LYNN_LINEAR_ATTN_GQA_RECURRENT=1`, the P32/P33/P36/P37/P38
MoE gates and P50/P62 first-divergence probes must run on the same fast baseline
as the service path. The harnesses now include that flag by default, P105 W4A8
fake-quant comparisons inherit it, and P11/P12 shadow-release graph checks
require it explicitly. This prevents future native-kernel probes from comparing
against the stale pre-GQA recurrent path.

## A100 Retire Decision

The A100 host is no longer on the critical path for the official Qwen3.6-35B
W4A16 serving route. On the 2026-05-18 retire check both A100 GPUs were idle
(`0%` utilization, `14 MiB` memory each), with no active Python/torch benchmark
or training jobs. The useful A100 output is already represented by the checked-in
reports plus the retained sidecars on R6000.

R6000 currently holds the sidecars worth keeping:

- official Qwen3.6-35B MTP sidecar
- A100 v27 weak-miss-in-top-k sidecar
- R6000 v34 rank-flip sidecar

The remaining A100-only v45/v46 hard-miss diagnostics did not improve the
promotable route and do not justify another rental day. A100 should only be
restarted if a concrete 35B W4A16 blocker requires large-GPU BF16 calibration
or if we deliberately start a new self-trained Qwen3.6 hybrid-SSM MTP project.

## Immediate Work

1. Stop open-ended A100 quality repair for 27B/W4A8 unless a concrete 35B
   W4A16 blocker appears.
2. Treat graph+in-place plus gate/up fastdecode, triton-pair QK/RoPE, triton
   gated RMSNorm, and GQA recurrent as the current R6000 W4A16 serving baseline.
3. Push native-kernel work from the 104-105 safe decode TPS line toward the 155
   target: numerically strict active/shared MoE boundary fusion first, then
   full-attention or linear-attention fusion. Python serving overhead is not the
   current high-ROI lever.
4. Keep MTP as a trained/calibrated sidecar project, not default promote credit.
