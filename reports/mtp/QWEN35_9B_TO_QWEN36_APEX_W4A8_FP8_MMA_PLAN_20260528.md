# Qwen 9B -> Qwen 35B APEX-MTP W4A8 / FP8-MMA Plan

Date: 2026-05-28

## Decision

Use **W4A8 / FP8-MMA** as the Spark-native acceleration path before attempting
more aggressive FP4-activation or true FP4-MMA variants.

Spark GB10 (`sm_121`) has usable FP8 MMA, while FP4 MMA is not the reliable
software/hardware path for this machine today. That makes W4A8 the practical
bridge between quality-preserving quantization and real TensorCore speed.

## Why Start With Qwen3.5-9B

Qwen3.5-9B already has a strong W4A8 quality signal:

| Variant | MMLU 500 | GPQA Diamond 198 | Delta vs W4A16 |
|---|---:|---:|---|
| W4A16 NVFP4 | 76.00% | 42.93% | baseline |
| W4A8 fake-quant FFN-only | 75.80% | 43.94% | -0.20pp / +1.01pp |

This makes 9B the right first closed loop:

1. Repack Lynn-native W4A16 NVFP4 weights to FP8 E4M3.
2. Run the FP8-MMA path, not only fake quant.
3. Compare TPS against W4A16.
4. Re-run MMLU/GPQA quick gates.

If 9B cannot show a clean speed win with flat quality, the 35B MoE/APEX-MTP
version is not ready for engineering time.

## Existing Building Blocks

Already landed:

| File | Role |
|---|---|
| `scripts/spark_pack_w4a8_fp8.py` | Full-dir NVFP4 -> FP8 E4M3 repack; includes 3D MoE expert V2 support |
| `scripts/spark_fp8_e2e_tps_smoke.py` | W4A16 vs W4A8 end-to-end TPS smoke |
| `scripts/spark_w4a8_vs_w4a16_quality_regression.py` | MMLU-100 + GPQA-50 quality gate |
| `triton_kernels/spark_fp8_gate_up_fused.py` | Dense FP8 fused gate/up + SwiGLU kernel |
| `triton_kernels/spark_fp8_moe_expert_fused.py` | MoE expert FP8 kernel |
| `reports/qwen35_9b/SPARK_QWEN35_9B_W4A8_VS_W4A16_QUALITY_REPORT_20260519.md` | 9B W4A8 quality evidence |

New runner:

```text
scripts/spark_qwen35_9b_w4a8_fp8_mma_closed_loop.sh
```

It does:

1. Refuse to run if the live APEX service is processing requests, unless
   `ALLOW_BUSY_APEX=1`.
2. Run `spark_pack_w4a8_fp8.py self-test`.
3. Repack 9B W4A16 NVFP4 -> W4A8 FP8 if the output dir is missing.
4. Run end-to-end TPS smoke.
5. Run MMLU/GPQA quick quality regression.

## 9B Execution Command

On Spark, after the current 35B Quality32K / think-off run is done:

```bash
cd /home/merkyor/lynn-engine

bash scripts/spark_qwen35_9b_w4a8_fp8_mma_closed_loop.sh
```

Useful overrides:

```bash
FORCE_REPACK=1 \
MAX_NEW_VALUES="64 128 256" \
OUT_ROOT=/home/merkyor/reports/qwen35_9b/w4a8_fp8_mma_closed_loop_manual \
bash scripts/spark_qwen35_9b_w4a8_fp8_mma_closed_loop.sh
```

Default paths:

```text
W4A16 input:
/home/merkyor/models/Qwen3.5-9B-lynn-native-w4a16-nvfp4-v0

W4A8 output:
/home/merkyor/models/Qwen3.5-9B-lynn-native-w4a8-fp8
```

## 9B Gates

Promotion from "candidate" to "useful Spark path" requires:

| Gate | Threshold |
|---|---:|
| Packer self-test | PASS |
| Full-dir repack deferred tensors | 0 |
| FP8 cosine failures | 0 below 0.999 |
| TPS vs W4A16 | > 1.25x |
| MMLU quick regression | <= 1pp |
| GPQA quick regression | <= 2pp |

If TPS lift is below 1.25x, the likely issue is runtime path integration
rather than quantization quality. In that case inspect whether the run is
actually using FP8 kernels or falling back to BF16.

## 35B APEX-MTP Follow-Up

Do not build 35B W4A8 by dequantizing the current APEX-MTP GGUF and
requantizing it. That path loses the source layout, calibration metadata, and
MTP provenance.

Correct source path:

```text
source safetensors / HF weights
  -> Lynn W4A8 FP8-MMA pack
  -> preserve or reattach MTP tensors
  -> calibrate base + MTP head
  -> benchmark accept rate + TPS + quality
```

APEX / I-Balanced should be treated as a **quantization policy reference**:

| Keep higher precision | Candidate low precision |
|---|---|
| router / norm / lm_head | dense FFN gate/up/down |
| sensitive first/last layers | low-sensitivity MoE experts |
| MTP head if accept drops | robust attention projections |

For 35B-A3B the calibration must be routing-aware:

1. Collect expert activation / routing frequency on representative prompts.
2. Keep top-risk experts or layers in higher precision.
3. Quantize low-risk expert matrices to FP8.
4. Measure MTP accept rate after quantization, not only MMLU/GPQA.

## 35B Gates

| Gate | Threshold |
|---|---:|
| MTP accept rate | >= current ~60% band |
| Single-stream TPS | > llama.cpp APEX-MTP 77 TPS |
| 2-way concurrency | no worse than current llama.cpp MTP total TPS |
| MMLU full / 500 | <= 1pp regression |
| GPQA Diamond | <= 2pp regression |
| Tool-call V8 stage1 | ship gate PASS |
| V9 | no obvious domain collapse |

The goal is not just lower memory. The 35B W4A8/APEX line only matters if it
beats the current `llama.cpp` APEX-MTP service on useful Spark serving:

```text
single user  -> faster than 77 TPS
2+ users     -> dynamic MTP admission or AR fallback
quality      -> within current W4A16/Q4KM/APEX confidence band
```

## Current Recommendation

Run the 9B closed loop first. If it passes, start 35B APEX-MTP W4A8 from
source weights plus MTP tensors, with imatrix-style calibration and accept-rate
measurement as first-class gates.
