# Stage 6 Phase 1 — single dense projection packed-NVFP4 PoC

**Date:** 2026-06-04
**Host:** Spark GB10 (`dgx-via-n5`)
**Model:** `/home/merkyor/models/Qwen3.6-35B-A3B-lynn-native-w4a16-nvfp4-from-bf16-20260526`
**Projection:** `model.language_model.layers.0.linear_attn.in_proj_qkv.weight`
**Runner:** `scripts/spark_stage6_p1_dense_projection_poc.py`
**Remote run dir:** `/home/merkyor/lynn-engine/reports/stage6/p1_dense_projection_poc_20260604_000859`
**APEX safety:** APEX stayed online; metrics showed `requests_processing=0`, `requests_deferred=0` before the run and `/health` returned `{"status":"ok"}` after the run.

Command:

```bash
docker run --rm --gpus all --ipc=host \
  -e PYTHONNOUSERSITE=1 -e PYTHONUNBUFFERED=1 \
  -v /home/merkyor:/home/merkyor \
  -w /home/merkyor/lynn-engine \
  lynn-eval-base:cu13 \
  python3 -u scripts/spark_stage6_p1_dense_projection_poc.py \
    --json-out reports/stage6/p1_dense_projection_poc_20260604_000859/result.json
```

## Verdict

**Phase 1 single dense projection PoC PASSED.**

This is the first true fused 4-bit / zero-shadow dense projection gate: one real
35B projection ran directly from packed NVFP4 bytes, after deleting all BF16/FP32
reference tensors, with numeric parity and a small speed win over BF16
`F.linear`.

Important correction: planning assumed E4M3 scale storage, but the actual
Lynn-native 35B artifact stores this projection's scale as **FP16**. The measured
byte ratio is therefore **32.00 MiB BF16 shadow / 10.00 MiB packed+scale =
3.20x**, not the larger ratio that E4M3 scales would imply.

## Shape And Bytes

| item | value |
|---|---:|
| original shape | `8192 x 2048` |
| packed dtype / shape | `torch.uint8`, `8192 x 1024` |
| scale dtype / shape | `torch.float16`, `8192 x 128` |
| global scale dtype / shape | `torch.float32`, scalar |
| BF16 shadow | 32.00 MiB |
| packed bytes | 8.00 MiB |
| scale bytes | 2.00 MiB |
| packed+scale+global | 10.00 MiB |
| timed packed arg bytes | 10.04 MiB |
| BF16 / packed+scale ratio | 3.20x |

## Numeric Gate

| comparison | cosine | rel L2 | max abs | argmax |
|---|---:|---:|---:|---|
| packed Triton vs FP32 dequant oracle | 1.000000000 | 1.387e-07 | 7.153e-07 | match |
| packed Triton vs BF16-shadow `F.linear` | 0.999998123 | 1.938e-03 | 8.632e-03 | match |

Primary correctness oracle is FP32 dequant from the same packed tensors. The
BF16-shadow comparison is reported separately because it includes BF16 weight
rounding.

## Microbench

| path | median | min | max |
|---|---:|---:|---:|
| packed Triton `nvfp4_matvec_packed` | 160.29 us | 158.61 us | 162.13 us |
| BF16-shadow `F.linear` | 190.03 us | 182.91 us | 192.29 us |
| speedup | **1.186x** |  |  |

This is M=1 matvec only. It proves the single-projection packed contract and
gives a real positive microbench signal; it does not yet prove batched prefill or
full dense-path integration.

## No-Shadow Evidence

The harness deletes `w_fp32`, `w_bf16`, and both reference outputs before the
timed packed benchmark, then resets peak CUDA memory.

| metric | value |
|---|---:|
| memory before packed bench | 0.0177 GiB |
| memory after packed bench | 0.0177 GiB |
| peak during packed bench | 0.0177 GiB |
| single projection BF16 shadow | 0.03125 GiB |

The timed packed path cannot be reading a resident BF16 shadow; the wide
reference tensors have been deleted and the peak remains below the projection's
BF16 shadow size.

## Gate Result

| evidence | result |
|---|---|
| byte-count | PASS |
| numeric parity | PASS |
| microbench | PASS, packed 1.186x vs BF16 |
| no hidden BF16 shadow | PASS |
| all | PASS |

## Next Step

Promote from single projection to P1-A:

1. Add a batched/M>1 packed projection kernel for the same
   `linear_attn.in_proj_qkv` shape.
2. Gate it against this M=1 oracle and BF16 prefill for prompt-length slices.
3. Then expand to linear-attn `z/b/a/out` and full-attn `q/k/v/o`.
4. Keep MoE grouped prefill as P2; do not mix it into this dense projection gate.
