# Qwen3.6-35B-A3B Q4_K_M imatrix + APEX-MTP Build Plan

Date: 2026-05-28

## Short Answer

Yes, our own Q4_K_M imatrix package can carry APEX-MTP, but only if the MTP
weights are present before quantization. A plain base-only Q4_K_M GGUF cannot be
made MTP-capable by metadata edits alone.

The correct path is:

```text
MTP-enabled HF/BF16 checkpoint
  -> convert_hf_to_gguf.py --outtype f16
  -> llama-imatrix calibration
  -> llama-quantize --imatrix ... Q4_K_M
  -> verify qwen35moe.nextn_predict_layers=1 and blk.40.nextn.* tensors
  -> short A/B TPS + MMLU/GPQA/tool regression
```

## Spark Inventory

Confirmed on `dgx-spark` via `dgx-via-n5`:

```text
/home/merkyor/models/Qwen3.6-35B-A3B-BF16-official-n5
  size: 67G
  model shards: model-00001-of-00026.safetensors ... model-00026-of-00026.safetensors
  index contains MTP keys:
    mtp.fc.weight
    mtp.layers.0.input_layernorm.weight
    mtp.layers.0.mlp.experts.down_proj
    mtp.layers.0.mlp.experts.gate_up_proj
    mtp.layers.0.mlp.gate.weight
    mtp.layers.0.self_attn.{q,k,v,o}_proj.weight
    mtp.norm.weight
    mtp.pre_fc_norm_embedding.weight
    mtp.pre_fc_norm_hidden.weight

/home/merkyor/models/mtp_sidecars/qwen36-35b-a3b-mtp-official-lynn-fused/mtp.safetensors
  size: 1.6G
  role: Lynn-engine fused sidecar path, not needed by embedded GGUF serving

/home/merkyor/models/Qwen3.6-35B-A3B-APEX-MTP-GGUF/Qwen3.6-35B-A3B-APEX-MTP-I-Balanced.gguf
  size: 25G
  general.file_type = 17
  quantize.imatrix.entries_count = 510
  quantize.imatrix.chunks_count = 2820
  qwen35moe.nextn_predict_layers = 1
  qwen35moe.block_count = 41
  qwen35moe.expert_count = 256
```

The old Spark staging symlink for
`/home/merkyor/.publish_stage/qwen36_q4km/hf/Qwen3.6-35B-A3B-Q4_K_M-imatrix.gguf`
currently points at `/home/merkyor/models/Qwen3.6-35B-A3B-GGUF-imatrix/...`, but
that target is not present. Treat the 35B Q4_K_M package as needing a rebuild.

## Why Metadata Edits Are Not Enough

llama.cpp `draft-mtp` needs real tensors for the extra prediction block. The
server checks architecture metadata and then loads tensors such as trailing
`nextn` / `mtp` projection, norm, attention, and MoE weights. Adding
`nextn_predict_layers=1` to a base-only GGUF would pass neither tensor loading
nor quality verification.

For release wording, a package is APEX-MTP-capable only if both are true:

- metadata has `qwen35moe.nextn_predict_layers = 1`;
- the GGUF contains the trailing MTP tensors, for the current embedded build
  observed as `blk.40.nextn.*` in GGUF / `mtp.*` in the HF checkpoint.

## Build Command Skeleton

Reuse the existing Q4_K_M backstop shape, but point it at the MTP-enabled BF16
source and emit a distinct filename so it cannot be confused with base-only
Q4_K_M:

```bash
ssh dgx-via-n5 '
set -euo pipefail
BF16=/home/merkyor/models/Qwen3.6-35B-A3B-BF16-official-n5
OUT=/home/merkyor/models/Qwen3.6-35B-A3B-APEX-MTP-Q4KM-imatrix
CALIB=/home/merkyor/calibration/calib_combined.txt
LLAMA=/home/merkyor/build/llama.cpp/build-cuda-sm121/bin
CONVERT=/home/merkyor/build/llama.cpp/convert_hf_to_gguf.py
mkdir -p "$OUT"

python3 "$CONVERT" "$BF16" \
  --outfile "$OUT/Qwen3.6-35B-A3B-APEX-MTP-F16.gguf" \
  --outtype f16

"$LLAMA/llama-imatrix" \
  -m "$OUT/Qwen3.6-35B-A3B-APEX-MTP-F16.gguf" \
  -f "$CALIB" \
  -o "$OUT/Qwen3.6-35B-A3B-APEX-MTP.imatrix" \
  --chunks 200 -ngl 99 -c 512

"$LLAMA/llama-quantize" \
  --imatrix "$OUT/Qwen3.6-35B-A3B-APEX-MTP.imatrix" \
  "$OUT/Qwen3.6-35B-A3B-APEX-MTP-F16.gguf" \
  "$OUT/Qwen3.6-35B-A3B-APEX-MTP-Q4_K_M-imatrix.gguf" \
  Q4_K_M
'
```

This should run only in a controlled Spark slot. `llama-imatrix` and
`llama-quantize` use GPU memory; do not compete with production voice services
or the 35B fallback service unless the service window is explicitly cleared.

## Verification Gate

Before publishing:

```bash
PYTHONPATH=/home/merkyor/build/llama.cpp/gguf-py \
python3 -m gguf.scripts.gguf_dump --json --no-tensors \
  /home/merkyor/models/Qwen3.6-35B-A3B-APEX-MTP-Q4KM-imatrix/Qwen3.6-35B-A3B-APEX-MTP-Q4_K_M-imatrix.gguf
```

Required:

- `qwen35moe.nextn_predict_layers = 1`
- `general.file_type` is a Q4_K_M-compatible quantized type
- `quantize.imatrix.entries_count > 0`
- `strings <gguf> | grep -E 'nextn|mtp'` finds the trailing MTP block tensors
- `llama-server --spec-type draft-mtp --spec-draft-n-max 4` logs
  `common_speculative_impl_draft_mtp`

Minimum runtime gate:

```text
single stream:
  AR control with speculative.n_max=0
  MTP n_max=4
  expect MTP > AR if accept stays healthy

4-way:
  AR may remain faster than MTP, same as current I-Balanced service

quality:
  run MMLU 500 and GPQA Diamond thinking-on/off smoke before release
```

## Expected ROI

The current I-Balanced APEX-MTP package already reaches 77.01 wall TPS
single-stream in the short Spark HTTP benchmark. Q4_K_M imatrix should reduce
weight bandwidth and may improve single-stream decode further, but it can also
change MTP accept rate and quality. This is a fresh artifact, not a guaranteed
drop-in replacement.

Recommended publish rule:

```text
publish Q4_K_M-MTP only if:
  single-stream TPS beats I-Balanced, and
  MMLU/GPQA/tool-call deltas stay inside the existing Q4_K_M quality band.
```

## 2026-05-28 Build Result

The self-quantized Q4_K_M imatrix rebuild is complete on Spark:

```text
/home/merkyor/models/Qwen3.6-35B-A3B-APEX-MTP-Q4KM-imatrix/
  Qwen3.6-35B-A3B-APEX-MTP-F16.gguf                  67G
  Qwen3.6-35B-A3B-APEX-MTP.imatrix                   184M
  Qwen3.6-35B-A3B-APEX-MTP-Q4_K_M-imatrix.gguf        21G
```

Checksums:

```text
6354476fe8f8820e59613394e13eb3bf2fb9276d2f05c6da17a4a798a50bc0f0  Qwen3.6-35B-A3B-APEX-MTP-Q4_K_M-imatrix.gguf
996e9875df061c426e3787babd30101bf09d1c301700bd11422e3d07ad8afdab  Qwen3.6-35B-A3B-APEX-MTP.imatrix
```

Calibration and quantization:

- `llama-imatrix --chunks 200 -ngl 99 -c 512`
- imatrix final PPL: `4.0908 +/- 0.03924`
- `llama-quantize --imatrix ... Q4_K_M`
- quantized size: `20696.09 MiB`
- quantized BPW: `4.89`

Metadata gate passed:

```text
general.file_type=15
qwen35moe.nextn_predict_layers=1
qwen35moe.block_count=41
qwen35moe.expert_count=256
quantize.imatrix.entries_count=510
quantize.imatrix.chunks_count=200
nextn_tensor_count=4
nextn_tensor=blk.40.nextn.eh_proj.weight
nextn_tensor=blk.40.nextn.enorm.weight
nextn_tensor=blk.40.nextn.hnorm.weight
nextn_tensor=blk.40.nextn.shared_head_norm.weight
```

Runtime smoke passed on temporary port `18099`:

```text
common_speculative_impl_draft_mtp: adding speculative implementation 'draft-mtp'
common_speculative_impl_draft_mtp: - n_max=4
```

Short think-off smoke:

```text
finish=stop
completion_tokens=38
predicted_per_second=55.11
draft_n=68
draft_n_accepted=23
content=Q4_K_M 是一种用于量化压缩模型权重的技术，而 MTP（多令牌预测）是一种通过并行预测多个未来词元来加速推理过程的生成策略。
```

## 2026-05-28 TPS Result

Single-stream 512-token `/no_think` benchmark, Spark local HTTP, `temperature=0`,
5 measured runs after warmup:

| Serving Mode | GGUF | Server TPS Mean | Wall TPS Mean | Draft Accept |
|---|---|---:|---:|---:|
| Q4_K_M imatrix, AR only | self-quantized Q4_K_M-MTP artifact served without `--spec-type` | **77.73** | **76.51** | n/a |
| Q4_K_M imatrix, draft-MTP n=4 | self-quantized Q4_K_M-MTP artifact served with `--spec-type draft-mtp` | 64.08 | 63.12 | 36.34% |
| I-Balanced, draft-MTP n=4 | current production fallback on `18098` | 53.29 | 52.52 | 36.85% |

Important conclusion: the rebuilt Q4_K_M artifact is valid and fast, but current
llama.cpp `draft-mtp` is not the fastest serving mode for this artifact on this
prompt family. Q4_K_M AR reaches the historical `~77 TPS` single-stream class by
itself, while MTP loses speed because the observed accept rate is only about
36% and the draft context overhead is larger than the accepted-token savings.

Treat the artifact and runtime policy separately:

- **Artifact:** publishable only after quality gates pass; it correctly contains
  embedded MTP tensors.
- **Runtime default:** prefer AR for single-stream speed unless a later prompt
  suite shows materially higher MTP accept rate. Keep `draft-mtp` as an opt-in
  experiment, not the default, for this Q4_K_M package.

## 2026-05-28 Think-Off Quality Result

After the TPS result, the Q4_K_M-MTP quality run was stopped during the
thinking-on section and the production `18098` APEX-MTP I-Balanced fallback was
restored. The completed think-off results are enough to mark this route as not
ready to replace the existing public quality anchors:

| Model / Runtime | MMLU 500 5-shot | GPQA Diamond 198 | Notes |
|---|---:|---:|---|
| BF16 official, historical anchor | 86.40% | 45.45% | prior Spark quality table |
| Q4_K_M-imatrix base, historical anchor | 83.00% | 50.00% | prior Spark quality table |
| Lynn-native W4A16 NVFP4, historical anchor | 84.40% | 49.49% | prior Spark quality table |
| Q4_K_M-MTP self build on `18099`, think-off | 81.40% | 41.41% | `407/500`, `82/198`; run stopped after MMLU thinking-on partial |
| APEX-MTP I-Balanced production on `18098`, think-off | 82.20% | 42.42% | `411/500`, `84/198`; production fallback stayed active |

Interpretation:

- The GPQA drop is too large to treat as normal quantization noise.
- The drop is not isolated to the new Q4_K_M-MTP self build: the existing
  APEX-MTP I-Balanced package also lands in the same low-40% GPQA band under
  explicit think-off.
- APEX-MTP I-Balanced still has strong separate 32K thinking-on results, so the
  publish/use decision should split by mode: use APEX-MTP for thinking-on
  workflows, but do not position it as the best think-off quality package.
- For think-off quality, the older Q4_K_M base and W4A16 NVFP4 anchors remain
  the stronger references until an AR-only run of this exact Q4_K_M-MTP artifact
  proves otherwise.

Result artifacts copied into the repo:

```text
reports/qwen36_35b/q4km_mtp_quality_20260528_181812/
reports/qwen36_35b/apex_thinkoff_20260528_191853/
reports/mtp/qwen36_q4km_mtp_tps_20260528/
```
