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

