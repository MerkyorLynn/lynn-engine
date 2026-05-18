#!/usr/bin/env bash
set -euo pipefail

# Download/quantize Lynn V4-Pro and V4-Flash 35B-A3B BF16 checkpoints into
# compressed-tensors NVFP4 v8-RTN artifacts for Spark MMLU/GPQA evaluation.

REPO=${REPO:-$(pwd)}
PY=${PY:-python3}
MODEL_ROOT=${MODEL_ROOT:-/mnt/data2/lynn-spark/models}
REPORT_ROOT=${REPORT_ROOT:-/mnt/data2/lynn-spark/reports/v4_35b_quant}
DOWNLOAD_SOURCE=${DOWNLOAD_SOURCE:-modelscope}
DOWNLOAD_BF16_IF_MISSING=${DOWNLOAD_BF16_IF_MISSING:-1}
VARIANTS=${VARIANTS:-"Pro Flash"}
ALLOW_OVERWRITE=${ALLOW_OVERWRITE:-0}
SKIP_IF_OUTPUT_EXISTS=${SKIP_IF_OUTPUT_EXISTS:-1}
MIN_FREE_GIB_FOR_BF16=${MIN_FREE_GIB_FOR_BF16:-95}
MIN_FREE_GIB_FOR_QUANT=${MIN_FREE_GIB_FOR_QUANT:-35}
TS=${TS:-$(date +%Y%m%d_%H%M%S)}
LOG=$REPORT_ROOT/spark_v4_35b_v8_rtn_quant_batch_${TS}.log
SUMMARY=$REPORT_ROOT/spark_v4_35b_v8_rtn_quant_summary_${TS}.json

mkdir -p "$MODEL_ROOT" "$REPORT_ROOT"
cd "$REPO"
export PYTHONPATH="$REPO"

model_name() {
  local variant="$1"
  echo "Lynn-V4-${variant}-Distill-Qwen-35B-A3B"
}

ms_repo() {
  local variant="$1"
  echo "Merkyor/$(model_name "$variant")-BF16-merged"
}

hf_repo() {
  local variant="$1"
  echo "nerkyor/$(model_name "$variant")-BF16-merged"
}

bf16_dir() {
  local variant="$1"
  echo "$MODEL_ROOT/$(model_name "$variant")-BF16-merged"
}

v8_dir() {
  local variant="$1"
  echo "$MODEL_ROOT/$(model_name "$variant")-NVFP4-v8-RTN"
}

free_gib_for_path() {
  local path="$1"
  mkdir -p "$path"
  df -BG "$path" | awk 'NR==2 {gsub("G", "", $4); print $4}'
}

require_free_gib() {
  local path="$1"
  local need="$2"
  local label="$3"
  local free
  free="$(free_gib_for_path "$path")"
  echo "[v4-quant][disk] ${label}: free=${free}GiB need>=${need}GiB path=$path"
  if (( free < need )); then
    echo "[v4-quant][disk] below target; largest model dirs:" >&2
    du -xhd1 "$MODEL_ROOT" 2>/dev/null | sort -h | tail -40 >&2 || true
    echo "[v4-quant][disk] largest report dirs:" >&2
    du -xhd2 "$REPORT_ROOT" 2>/dev/null | sort -h | tail -40 >&2 || true
    echo "[v4-quant][disk] continuing; user authorized disk management, but this batch does not delete non-owned model dirs automatically." >&2
  fi
}

download_bf16() {
  local variant="$1"
  local target="$2"
  local source="$DOWNLOAD_SOURCE"
  local ms
  local hf
  ms="$(ms_repo "$variant")"
  hf="$(hf_repo "$variant")"
  if [[ -f "$target/model.safetensors.index.json" ]]; then
    echo "[v4-quant][$variant] BF16 already staged: $target"
    return
  fi
  if [[ "$DOWNLOAD_BF16_IF_MISSING" != "1" ]]; then
    echo "[v4-quant][$variant][fail] missing BF16: $target" >&2
    exit 2
  fi
  require_free_gib "$MODEL_ROOT" "$MIN_FREE_GIB_FOR_BF16" "before $variant BF16 download"
  mkdir -p "$target"
  "$PY" - "$target" "$source" "$ms" "$hf" <<'PY'
import pathlib
import sys

target = pathlib.Path(sys.argv[1])
source = sys.argv[2].strip().lower()
ms_repo = sys.argv[3]
hf_repo = sys.argv[4]

if source in {"modelscope", "ms"}:
    from modelscope import snapshot_download
    got = snapshot_download(ms_repo, local_dir=str(target))
elif source in {"hf", "huggingface"}:
    from huggingface_hub import snapshot_download
    got = snapshot_download(
        repo_id=hf_repo,
        local_dir=str(target),
        local_dir_use_symlinks=False,
    )
else:
    raise SystemExit(f"unsupported DOWNLOAD_SOURCE={source!r}")

print({"repo": ms_repo if source in {"modelscope", "ms"} else hf_repo, "local_dir": got})
PY
  if [[ ! -f "$target/model.safetensors.index.json" ]]; then
    echo "[v4-quant][$variant][fail] BF16 download/stage incomplete: $target" >&2
    exit 2
  fi
}

run_variant() {
  local variant="$1"
  local src
  local out
  local report
  src="$(bf16_dir "$variant")"
  out="$(v8_dir "$variant")"
  report="$REPORT_ROOT/spark_v4_${variant,,}_v8_rtn_quant_${TS}.json"

  echo "[v4-quant][$variant] src=$src"
  echo "[v4-quant][$variant] out=$out"
  download_bf16 "$variant" "$src"

  if [[ "$SKIP_IF_OUTPUT_EXISTS" == "1" && -d "$out" && -e "$out/config.json" ]]; then
    echo "[v4-quant][$variant] output exists; writing status report and skipping quant"
    "$PY" - "$report" "$src" "$out" <<'PY'
import json
import pathlib
import sys

report = pathlib.Path(sys.argv[1])
src = pathlib.Path(sys.argv[2])
out = pathlib.Path(sys.argv[3])
report.write_text(json.dumps({
    "schema_version": "lynn-v4-35b-v8-rtn-quantize-v1",
    "src_model": str(src),
    "out_model": str(out),
    "decision": "SKIPPED: output already exists.",
}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(report)
PY
    return
  fi

  require_free_gib "$MODEL_ROOT" "$MIN_FREE_GIB_FOR_QUANT" "before $variant v8-RTN quant"
  args=(
    scripts/v4_35b_v8_rtn_quantize.py
    --src-model "$src"
    --out-model "$out"
    --report "$report"
  )
  if [[ "$ALLOW_OVERWRITE" == "1" ]]; then
    args+=(--overwrite)
  fi
  "$PY" "${args[@]}"
}

{
  echo "[v4-quant] start $(date)"
  echo "[v4-quant] repo=$REPO"
  echo "[v4-quant] model_root=$MODEL_ROOT"
  echo "[v4-quant] report_root=$REPORT_ROOT"
  echo "[v4-quant] variants=$VARIANTS source=$DOWNLOAD_SOURCE"
  df -h "$MODEL_ROOT" "$REPORT_ROOT" 2>/dev/null || true

  for variant in $VARIANTS; do
    run_variant "$variant"
  done

  "$PY" - "$SUMMARY" "$REPORT_ROOT" "$TS" <<'PY'
import json
import pathlib
import sys

summary_path = pathlib.Path(sys.argv[1])
report_root = pathlib.Path(sys.argv[2])
ts = sys.argv[3]
reports = sorted(report_root.glob(f"spark_v4_*_v8_rtn_quant_{ts}.json"))
rows = []
for path in reports:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows.append({
        "report": str(path),
        "src_model": data.get("src_model"),
        "out_model": data.get("out_model"),
        "decision": data.get("decision"),
        "elapsed_seconds": data.get("elapsed_seconds"),
        "output_bytes": data.get("output_bytes"),
    })
summary = {
    "schema_version": "lynn-v4-35b-v8-rtn-quant-batch-v1",
    "timestamp": ts,
    "reports": rows,
    "decision": (
        "GREEN: all requested V4 35B v8-RTN quantization jobs produced or found outputs."
        if rows and all(str(row.get("decision", "")).startswith(("GREEN", "SKIPPED")) for row in rows)
        else "CHECK: one or more V4 35B v8-RTN jobs need inspection."
    ),
}
summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False, indent=2))
PY
  echo "[v4-quant] summary=$SUMMARY"
  echo "[v4-quant] done $(date)"
} >> "$LOG" 2>&1

echo "$LOG"
