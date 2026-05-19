# Qwen3.5-9B Q4_K_M-imatrix Gate

Status: queued on R6000 behind the current official 9B long-running round.

Important naming note: `Q4_K_M` is the llama.cpp quantization type. It does not
prove that an artifact was calibrated with an importance matrix. The current
official 9B GGUF on R6000 is therefore tracked as standard/unknown-imatrix until
this gate produces a separate `Q4_K_M-imatrix` artifact.

## Current Reference

| Artifact | Path | imatrix status |
| --- | --- | --- |
| 9B official Q4_K_M | `/root/autodl-tmp/models/Qwen3.5-9B-Q4_K_M.gguf` | unknown / not encoded in path |
| 9B generated Q4_K_M-imatrix | `/root/autodl-tmp/models/Qwen3.5-9B-GGUF-imatrix/Qwen3.5-9B-Q4_K_M-imatrix.gguf` | queued |

## Queued Pipeline

Script:

`/root/autodl-tmp/lynn-engine/scripts/r6000_qwen35_9b_q4km_imatrix_gate.sh`

Stages:

1. Wait for the current Q4_K_M CUDA 32K GPQA run and official 9B round watcher.
2. Build missing llama.cpp `llama-imatrix`, `llama-quantize`, and bench tools.
3. Convert official BF16 HF package to F16 GGUF.
4. Build calibration text from Lynn calibration JSONL.
5. Run llama.cpp imatrix calibration.
6. Quantize `Q4_K_M` with the produced imatrix.
7. Run MMLU 500 5-shot, GPQA Diamond, GPQA thinking-on 32K, and llama.cpp TPS.

Primary queued log on R6000:

`/root/autodl-tmp/reports/qwen35_9b/q4km_imatrix_gate_queue_20260519_123147.log`

## Promotion Use

This is a comparison artifact, not an automatic replacement. It must beat or
match the current official/unknown-imatrix Q4_K_M quality without a meaningful
speed regression before it becomes the recommended GGUF download.
