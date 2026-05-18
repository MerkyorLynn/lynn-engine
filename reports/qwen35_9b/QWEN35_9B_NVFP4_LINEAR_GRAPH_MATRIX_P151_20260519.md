# Qwen3.5-9B NVFP4 Linear-Graph Serving Matrix

**Date:** 2026-05-19  
**Model:** `/root/autodl-tmp/models/Qwen3.5-9B-lynn-native-w4a16-nvfp4-v0`  
**Profile:** `linear_graph_only`

## Verdict

P151 confirms the safe single-stream improvement and exposes the next bottleneck:
Lynn Engine's current 9B OpenAI server path does not scale total throughput with
concurrent requests.

## Matrix

| Lane | Result |
|---|---:|
| Single 128 wall TPS | 52.35 |
| Single 256 wall TPS | 59.16 |
| Single 512 wall TPS | 60.09 |
| Concurrent x2 total TPS | 60.03 |
| Concurrent x4 total TPS | 60.08 |
| Concurrent x8 total TPS | 60.11 |
| Long context 4k wall TPS | 56.11 |
| Long context 16k wall TPS | 51.38 |
| Long context 32k wall TPS | 45.02 |

## Readout

The 9B NVFP4 NVIDIA path now has a stable 60 TPS class single-stream profile,
up from the old 40.9 TPS release matrix.  It still does not compete with the
Q4_K_M llama.cpp throughput profile at 168.23 TPS single and 420.63 TPS x8.

The next useful 9B work is therefore:

1. server concurrency/batching path, if product value needs multi-request TPS;
2. dense FFN packed/fused kernel work, if single-stream speed is the priority;
3. avoid enabling broader 35B fast-profile knobs because P149 already showed
   they drift on 9B.

## Artifacts

- `scripts/r6000_qwen35_9b_nvfp4_linear_graph_matrix.sh`
- `reports/qwen35_9b/p151_qwen35_9b_nvfp4_linear_graph_matrix_20260519_0418.json`
- `reports/qwen35_9b/p151_qwen35_9b_nvfp4_linear_graph_matrix_summary_20260519_0418.json`
