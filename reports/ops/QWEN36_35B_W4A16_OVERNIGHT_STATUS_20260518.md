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
| Qwen3.6-35B-A3B Lynn-native W4A16 NVFP4 | 84.40% | 49.49% |
| Delta W4A16 vs BF16 | -2.00pp | +4.04pp |

This validates the pivot away from custom 27B recovery as the primary quality
route. The model quality problem is no longer broad repair; it is preserving the
official model through native quantization and runtime optimization.

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

References:

- <https://github.com/Avarok-Cybersecurity/atlas>
- <https://atlasinference.io/>
- <https://arxiv.org/abs/2605.01106>
- <https://njannasch.dev/blog/mtp-speculative-decoding-qwen-3-6-5060ti/>

## Immediate Work

1. Stop open-ended A100 quality repair for 27B/W4A8 unless a concrete 35B
   W4A16 blocker appears.
2. Make graph+in-place the next R6000 speed baseline candidate and run a broader
   structured/tool-call gate before default serving promotion.
3. Push native-kernel work from 82 decode TPS toward the 155 target: packed
   prefill/decode parity, MoE grouped kernel, and Python serving overhead.
4. Keep MTP as a trained/calibrated sidecar project, not default promote credit.
