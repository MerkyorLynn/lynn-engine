# Qwen3.5-9B NVFP4 Dense FFN Phase Profile Scaffold

**Date:** 2026-05-19  
**Model:** `/root/autodl-tmp/models/Qwen3.5-9B-lynn-native-w4a16-nvfp4-v0`  
**Probe:** `benchmarks/p155_qwen35_9b_dense_ffn_phase_profile.py`

## Purpose

P155 opens the official Qwen3.5-9B Lynn-native W4A16 NVFP4 resident decode path
and measures per-token costs for the current dense architecture:

| Phase | JSON key |
|---|---|
| Dense FFN gate projection | `dense_gate_ms` |
| Dense FFN up projection | `dense_up_ms` |
| Dense FFN activation/multiply | `dense_act_mul_ms` |
| Dense FFN down projection | `dense_down_ms` |
| Linear-attention / SSM decode block | `linear_ssm_ms` |
| Full attention decode block | `full_attention_ms` |
| Final norm | `norm_ms` |
| LM head | `lm_head_ms` |
| Argmax | `argmax_ms` |
| Host/runtime residual gap | `host_gap_ms` |

The wrapper does not export Lynn runtime knobs. It profiles the model defaults
unless the caller explicitly exports overrides before launch.

## R6000 Command

```bash
cd /root/autodl-tmp/lynn-engine
bash scripts/r6000_qwen35_9b_dense_ffn_phase_profile.sh
```

Optional run length:

```bash
MAX_NEW=256 SKIP_STEPS=16 bash scripts/r6000_qwen35_9b_dense_ffn_phase_profile.sh
```

## Summary Artifacts

The P155 JSON embeds these quality artifact paths so phase data can be read
without losing the release-quality context:

- `reports/qwen35_9b/nvfp4_openai_quality_20260519_022635_mmlu_n500.summary.json`
- `reports/qwen35_9b/nvfp4_openai_quality_20260519_022635_mmlu_n500.jsonl`
- `reports/qwen35_9b/nvfp4_openai_quality_20260519_022635_gpqa.summary.json`
- `reports/qwen35_9b/nvfp4_openai_quality_20260519_022635_gpqa.jsonl`

The P155 JSON also preserves the current release metrics from:

- `reports/qwen35_9b/qwen35_9b_release_matrix.json`
- `reports/qwen35_9b/qwen35_9b_release_matrix.md`
- `reports/qwen35_9b/p151_qwen35_9b_nvfp4_linear_graph_matrix_summary_20260519_0418.json`

## R6000 Short Validation

Artifact:
`reports/qwen35_9b/p155_qwen35_9b_dense_ffn_phase_profile_20260519_041506_short.json`

Run shape: 64 max-new, skip first 8 decode steps, no fast env exported by the
wrapper.

| Phase | Mean ms/token |
|---|---:|
| wall | 31.6655 |
| linear-attn / SSM | 12.2953 |
| dense FFN total | 7.3656 |
| full attention | 4.7572 |
| LM head | 1.3389 |
| host gap | 1.8008 |

Dense FFN breakdown:

| Dense FFN sub-phase | Mean ms/token |
|---|---:|
| gate projection | 2.6314 |
| up projection | 2.2480 |
| activation/multiply | 0.1544 |
| down projection | 2.3319 |

Interpretation: dense FFN is a real 9B target, but it is not the only blocker.
The largest default decode block is still linear-attn/SSM, so the 9B speed line
should keep the already-safe linear graph path while opening a separate dense
FFN repack/fusion fixture track.

## Preserved Release Metrics

Do not replace the existing release matrix with P155 probe numbers. The current
published NVFP4 release metrics remain:

| Metric | Value |
|---|---:|
| MMLU | 0.7520 (376/500) |
| GPQA | 0.4293 (85/198) |
| Single TPS | 128t=35.1 / 256t=41.0 / 512t=40.9 |
| Concurrent TPS | x2=40.4 / x4=40.2 / x8=40.2 |
| Long context TPS | 4k=39.7 / 16k=37.7 / 32k=34.5 |

P151 remains the safe opt-in linear-graph candidate at roughly 60 TPS class
single stream; P155 is only the dense FFN and phase-cost scaffold for the next
kernel work.
