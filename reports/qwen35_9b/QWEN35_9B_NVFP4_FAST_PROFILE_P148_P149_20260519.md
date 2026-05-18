# Qwen3.5-9B NVFP4 Fast Profile Sweep

**Date:** 2026-05-19  
**Model:** `/root/autodl-tmp/models/Qwen3.5-9B-lynn-native-w4a16-nvfp4-v0`  
**Purpose:** find a quality-safe speed profile for Lynn-native W4A16 NVFP4 9B before writing dense FFN kernels.

## Verdict

P149 found a safe opt-in speed candidate:

```text
linear_graph_only
```

It is exact against the conservative baseline on the 3-prompt direct-runner gate
at both 128 and 512 generated tokens.

| Gate | Baseline decode TPS | `linear_graph_only` TPS | Speedup | Exact |
|---|---:|---:|---:|---:|
| 128 tokens | 40.05 | 59.73 | 1.491x | 3/3 |
| 512 tokens | 40.71 | 60.16 | 1.478x | 3/3 |

This should become the next 9B serving-gate candidate. It is not a release
default until it passes the OpenAI/P25 matrix.

## Closed Or Cautioned Knobs

The full 35B fast profile is not safe for 9B yet:

| Candidate | 128-token result | 512-token result | Readout |
|---|---:|---:|---|
| `full_fast_profile` | 1/3 exact, 1.151x | 1/3 exact, 1.144x | closed for default |
| `triton_core_no_graph_no_packed` | 2/3 exact, 1.367x | 2/3 exact, 1.375x | drift despite speed |
| `native_lm_head_only` | 1/3 exact, 1.026x | 1/3 exact, 1.024x | drift |
| `packed_decode_only` | 2/3 exact, 0.770x | 2/3 exact, 0.775x | slower and drift |
| `native_inproj_only` | 2/3 exact, 0.876x | 2/3 exact, 0.878x | slower and drift |
| `fast_no_packed_decode` | 1/3 exact, 1.883x | 1/3 exact, 1.858x | fastest but drift |
| `fast_no_native_lm_head` | 3/3 exact, 1.112x | 2/3 exact, 1.104x | not stable at 512 |

The safe speed is specifically from reusable linear-block graphing, not from
packed decode, native in-proj, or native LM head.

## Next Step

Run a 9B OpenAI/P25 serving gate with only these env knobs:

```text
LYNN_LINEAR_STATE_UPDATE=inplace
LYNN_LINEAR_BLOCK_GRAPH=1
LYNN_LINEAR_BLOCK_GRAPH_REUSE=1
LYNN_LINEAR_BLOCK_GRAPH_PREWARM=1
```

If the serving gate is exact and holds the 60 TPS class, the published NVFP4
runtime should be updated from 40.9 TPS to the linear-graph profile. Further
speed work should then move to dense FFN packed/fused kernels.

## Artifacts

- `benchmarks/p148_qwen35_9b_nvfp4_fast_profile.py`
- `benchmarks/p149_qwen35_9b_nvfp4_fast_knob_sweep.py`
- `scripts/r6000_qwen35_9b_nvfp4_fast_profile.sh`
- `scripts/r6000_qwen35_9b_nvfp4_fast_knob_sweep.sh`
- `reports/qwen35_9b/p148_qwen35_9b_nvfp4_fast_profile_20260519_0338.json`
- `reports/qwen35_9b/p149_qwen35_9b_nvfp4_fast_knob_sweep_20260519_0343.json`
- `reports/qwen35_9b/p149_qwen35_9b_nvfp4_fast_knob_sweep_20260519_0352.json`
