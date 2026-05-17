#!/usr/bin/env bash
set -euo pipefail

# Build and smoke-test a Lynn-native NVFP4 artifact for the unpruned
# Lynn V4-Pro 35B-A3B checkpoint.
#
# This is a pivot probe: if 35B NVFP4 preserves most of the V4-Pro quality and
# the memory delta versus 27B is small enough, 27B pruning becomes a speed/edge
# specialization instead of the default quality route.

REPO=${REPO:-/root/autodl-tmp/lynn-engine}
PY=${PY:-/root/autodl-tmp/conda-envs/r6000-eval/bin/python}
BF16_MODEL=${BF16_MODEL:-/root/autodl-tmp/models/Lynn-V4-Pro-Distill-Qwen-35B-A3B-BF16-merged}
EXISTING_V8_MODEL=${EXISTING_V8_MODEL:-/root/autodl-tmp/models/Lynn-V4-Pro-Distill-Qwen-35B-A3B-NVFP4-v8-RTN}
OUT_MODEL=${OUT_MODEL:-/root/autodl-tmp/models/Lynn-V4-Pro-Distill-Qwen-35B-A3B-lynn-native-nvfp4-v0}
COMPARE_27B_MODEL=${COMPARE_27B_MODEL:-/root/autodl-tmp/models/lynn-27b-a3b-w4a8-nvfp4-v2}
REPORT_ROOT=${REPORT_ROOT:-/root/autodl-tmp/reports}
DOWNLOAD_BF16_IF_MISSING=${DOWNLOAD_BF16_IF_MISSING:-1}
DOWNLOAD_SOURCE=${DOWNLOAD_SOURCE:-modelscope}
MS_REPO=${MS_REPO:-Merkyor/Lynn-V4-Pro-Distill-Qwen-35B-A3B}
HF_REPO=${HF_REPO:-Merkyor/Lynn-V4-Pro-Distill-Qwen-35B-A3B}
TS=${TS:-$(date +%Y%m%d_%H%M%S)}
REPORT_DIR=$REPORT_ROOT/v4pro_nvfp4
LOG=$REPORT_DIR/r6000_v4pro_nvfp4_pivot_${TS}.log
SUMMARY=$REPORT_DIR/r6000_v4pro_nvfp4_pivot_summary_${TS}.json
MAX_NEW=${MAX_NEW:-4}
TOP_K=${TOP_K:-10}
SKIP_PACK=${SKIP_PACK:-0}
ALLOW_OVERWRITE=${ALLOW_OVERWRITE:-0}
RUN_RESIDENT_SMOKE=${RUN_RESIDENT_SMOKE:-1}
CLEAN_DISK_IF_NEEDED=${CLEAN_DISK_IF_NEEDED:-1}
MIN_FREE_GIB_FOR_DOWNLOAD=${MIN_FREE_GIB_FOR_DOWNLOAD:-90}
MIN_FREE_GIB_FOR_PACK=${MIN_FREE_GIB_FOR_PACK:-35}

mkdir -p "$REPORT_DIR"

cd "$REPO"
export PYTHONPATH="$REPO"
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export LYNN_MOE_IMPL=${LYNN_MOE_IMPL:-packed_nvfp4}
export LYNN_NATIVE_FP4_LM_HEAD=${LYNN_NATIVE_FP4_LM_HEAD:-1}
export LYNN_MOE_FAST_FIXED=${LYNN_MOE_FAST_FIXED:-1}
export LYNN_FULL_TOKEN_GRAPH_SLOT=0
export LYNN_MTP_SHADOW_VERIFY=0
export LYNN_MTP_VERIFY=0

free_gib_for_path() {
  local path="$1"
  mkdir -p "$path"
  df -BG "$path" | awk 'NR==2 {gsub("G", "", $4); print $4}'
}

cleanup_pivot_scratch() {
  if [[ "$CLEAN_DISK_IF_NEEDED" != "1" ]]; then
    return
  fi
  echo "[v4pro-nvfp4][disk] cleaning regenerable pivot scratch"
  find "$REPORT_DIR" -maxdepth 1 -type f \
    \( -name 'r6000_v4pro_pack_stdout_*.json' -o -name 'r6000_v4pro_nvfp4_pivot_*.log' \) \
    -mtime +1 -print -delete 2>/dev/null || true
  if [[ -d "$OUT_MODEL" && ! -f "$OUT_MODEL/model.safetensors.index.json" ]]; then
    echo "[v4pro-nvfp4][disk] removing incomplete OUT_MODEL=$OUT_MODEL"
    rm -rf "$OUT_MODEL"
  fi
}

require_free_gib() {
  local path="$1"
  local need="$2"
  local label="$3"
  local free
  free="$(free_gib_for_path "$path")"
  echo "[v4pro-nvfp4][disk] ${label}: free=${free}GiB need>=${need}GiB path=$path"
  if (( free >= need )); then
    return
  fi
  cleanup_pivot_scratch
  free="$(free_gib_for_path "$path")"
  echo "[v4pro-nvfp4][disk] ${label} after cleanup: free=${free}GiB need>=${need}GiB"
  if (( free < need )); then
    echo "[v4pro-nvfp4][disk] still below target; largest model dirs:" >&2
    du -xhd1 /root/autodl-tmp/models 2>/dev/null | sort -h | tail -30 >&2 || true
    echo "[v4pro-nvfp4][disk] largest report dirs:" >&2
    du -xhd2 "$REPORT_ROOT" 2>/dev/null | sort -h | tail -30 >&2 || true
    echo "[v4pro-nvfp4][disk] continuing anyway; user authorized disk management, but script will not delete non-pivot model dirs automatically." >&2
  fi
}

{
  echo "[v4pro-nvfp4] start $(date)"
  echo "[v4pro-nvfp4] bf16=$BF16_MODEL"
  echo "[v4pro-nvfp4] existing_v8=$EXISTING_V8_MODEL"
  echo "[v4pro-nvfp4] out_model=$OUT_MODEL"
  echo "[v4pro-nvfp4] compare_27b=$COMPARE_27B_MODEL"
  echo "[v4pro-nvfp4] download_if_missing=$DOWNLOAD_BF16_IF_MISSING source=$DOWNLOAD_SOURCE"
  df -h "$(dirname "$BF16_MODEL")" "$REPORT_ROOT" 2>/dev/null || true

  if [[ ! -d "$BF16_MODEL" ]]; then
    if [[ "$DOWNLOAD_BF16_IF_MISSING" != "1" ]]; then
      echo "[v4pro-nvfp4][fail] missing BF16_MODEL=$BF16_MODEL" >&2
      echo "[v4pro-nvfp4][hint] set DOWNLOAD_BF16_IF_MISSING=1 or pre-stage the BF16 model" >&2
      exit 2
    fi
    echo "[v4pro-nvfp4] BF16 model missing; downloading from $DOWNLOAD_SOURCE"
    require_free_gib "$(dirname "$BF16_MODEL")" "$MIN_FREE_GIB_FOR_DOWNLOAD" "before BF16 download"
    mkdir -p "$(dirname "$BF16_MODEL")"
    "$PY" - "$BF16_MODEL" "$DOWNLOAD_SOURCE" "$MS_REPO" "$HF_REPO" <<'PY'
import pathlib
import sys

target = pathlib.Path(sys.argv[1])
source = sys.argv[2].strip().lower()
ms_repo = sys.argv[3]
hf_repo = sys.argv[4]

target.parent.mkdir(parents=True, exist_ok=True)

if source in {"modelscope", "ms"}:
    try:
        from modelscope import snapshot_download
    except Exception as exc:
        raise SystemExit(
            f"[fail] modelscope package is unavailable for {ms_repo}: {exc}. "
            "Install modelscope or set DOWNLOAD_SOURCE=hf."
        )
    got = snapshot_download(ms_repo, local_dir=str(target))
elif source in {"hf", "huggingface"}:
    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:
        raise SystemExit(
            f"[fail] huggingface_hub package is unavailable for {hf_repo}: {exc}. "
            "Install huggingface_hub or set DOWNLOAD_SOURCE=modelscope."
        )
    got = snapshot_download(
        repo_id=hf_repo,
        local_dir=str(target),
        local_dir_use_symlinks=False,
    )
else:
    raise SystemExit(f"[fail] unsupported DOWNLOAD_SOURCE={source!r}")

print({"repo": ms_repo if source in {"modelscope", "ms"} else hf_repo, "local_dir": got})
PY
  fi
  if [[ ! -f "$BF16_MODEL/model.safetensors.index.json" ]]; then
    echo "[v4pro-nvfp4][fail] BF16 model directory is incomplete: $BF16_MODEL" >&2
    echo "[v4pro-nvfp4][hint] expected model.safetensors.index.json after download/staging" >&2
    exit 2
  fi

  pack_args=(
    scripts/a100_pack_lynn_native_nvfp4.py
    --src-model "$BF16_MODEL"
    --out-model "$OUT_MODEL"
  )
  if [[ "$ALLOW_OVERWRITE" == "1" ]]; then
    pack_args+=(--overwrite)
  fi

  if [[ "$SKIP_PACK" == "1" && -d "$OUT_MODEL" ]]; then
    echo "[v4pro-nvfp4] skip pack because SKIP_PACK=1 and OUT_MODEL exists"
  else
    require_free_gib "$(dirname "$OUT_MODEL")" "$MIN_FREE_GIB_FOR_PACK" "before NVFP4 pack"
    echo "[v4pro-nvfp4] pack Lynn-native per16 NVFP4"
    "$PY" "${pack_args[@]}" \
      > "$REPORT_DIR/r6000_v4pro_pack_stdout_${TS}.json"
  fi

  echo "[v4pro-nvfp4] scan manifests"
  "$PY" benchmarks/quant_manifest_probe.py "$BF16_MODEL" \
    --out "$REPORT_DIR/r6000_v4pro_bf16_manifest_${TS}.json"
  "$PY" benchmarks/quant_manifest_probe.py "$OUT_MODEL" \
    --out "$REPORT_DIR/r6000_v4pro_lynn_native_nvfp4_manifest_${TS}.json"
  if [[ -d "$EXISTING_V8_MODEL" ]]; then
    "$PY" benchmarks/quant_manifest_probe.py "$EXISTING_V8_MODEL" \
      --out "$REPORT_DIR/r6000_v4pro_existing_v8_manifest_${TS}.json"
  else
    echo "[v4pro-nvfp4] existing v8 model missing; skip v8 manifest"
  fi

  echo "[v4pro-nvfp4] resident BF16-vs-native-NVFP4 logit/top-k smoke"
  "$PY" benchmarks/p2_resident_logit_analysis.py \
    --bf16 "$BF16_MODEL" \
    --v8 "$OUT_MODEL" \
    --out "$REPORT_DIR/r6000_v4pro_lynn_native_nvfp4_resident_logit_${TS}.json" \
    --max-new "$MAX_NEW" \
    --top-k "$TOP_K"

  if [[ "$RUN_RESIDENT_SMOKE" == "1" ]]; then
    echo "[v4pro-nvfp4] native NVFP4 resident generation smoke"
    "$PY" benchmarks/resident_cli.py \
      --model "$OUT_MODEL" \
      --prompt "Return one JSON object with keys city and unit for Tokyo in metric units. No markdown." \
      --out "$REPORT_DIR/r6000_v4pro_lynn_native_nvfp4_generate_smoke_${TS}.json" \
      --max-new 32 \
      --top-k 5 \
      --chat-template \
      --quiet
  fi

  echo "[v4pro-nvfp4] size summary"
  "$PY" - "$SUMMARY" "$BF16_MODEL" "$OUT_MODEL" "$EXISTING_V8_MODEL" "$COMPARE_27B_MODEL" <<'PY'
import json
import os
import pathlib
import subprocess
import sys

out = pathlib.Path(sys.argv[1])
paths = {
    "v4pro_bf16": pathlib.Path(sys.argv[2]),
    "v4pro_lynn_native_nvfp4": pathlib.Path(sys.argv[3]),
    "v4pro_existing_v8": pathlib.Path(sys.argv[4]),
    "compare_27b_nvfp4": pathlib.Path(sys.argv[5]),
}

def du_bytes(path: pathlib.Path):
    if not path.exists():
        return None
    got = subprocess.check_output(["du", "-sk", str(path)], text=True).split()[0]
    return int(got) * 1024

sizes = {name: du_bytes(path) for name, path in paths.items()}
delta_vs_27b = None
if sizes["v4pro_lynn_native_nvfp4"] is not None and sizes["compare_27b_nvfp4"] is not None:
    delta_vs_27b = sizes["v4pro_lynn_native_nvfp4"] - sizes["compare_27b_nvfp4"]

summary = {
    "schema_version": "lynn-v4pro-nvfp4-pivot-summary-v1",
    "paths": {name: str(path) for name, path in paths.items()},
    "sizes_bytes": sizes,
    "sizes_gib": {
        name: (value / (1024**3) if value is not None else None)
        for name, value in sizes.items()
    },
    "delta_v4pro_native_vs_27b_bytes": delta_vs_27b,
    "delta_v4pro_native_vs_27b_gib": (
        delta_vs_27b / (1024**3) if delta_vs_27b is not None else None
    ),
    "decision": (
        "MEASURE_QUALITY_NEXT: native V4-Pro NVFP4 artifact exists; run MMLU/GPQA before changing default route."
        if sizes["v4pro_lynn_native_nvfp4"] is not None
        else "RED: native V4-Pro NVFP4 artifact missing."
    ),
}
out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False, indent=2))
PY

  echo "[v4pro-nvfp4] summary=$SUMMARY"
  echo "[v4pro-nvfp4] done $(date)"
} >> "$LOG" 2>&1

echo "$LOG"
