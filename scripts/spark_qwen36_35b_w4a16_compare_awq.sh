#!/usr/bin/env bash
set -euo pipefail

# Spark/DGX helper for the official Qwen3.6-35B-A3B route:
#   official BF16 -> Lynn-native W4A16/NVFP4 -> compare with existing AWQ W4A16.
#
# The script intentionally refuses Lynn internal layer-split BF16 artifacts
# (`layers-*.safetensors`) so we do not accidentally quantize a stale reformat.

WORKDIR="${WORKDIR:-/home/merkyor/lynn-engine}"
MODEL_ROOT="${MODEL_ROOT:-/home/merkyor/models}"
REPORT_ROOT="${REPORT_ROOT:-/home/merkyor/reports}"
REPORT_DIR="${REPORT_DIR:-$REPORT_ROOT/qwen36_w4a16_compare_awq}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
LOG="${LOG:-$REPORT_DIR/spark_qwen36_35b_w4a16_compare_awq_${STAMP}.log}"

BF16_MODEL="${BF16_MODEL:-$MODEL_ROOT/Qwen3.6-35B-A3B-BF16-official}"
OUT_MODEL="${OUT_MODEL:-$MODEL_ROOT/Qwen3.6-35B-A3B-lynn-native-w4a16-nvfp4-v0}"
AWQ_MODEL="${AWQ_MODEL:-$MODEL_ROOT/Qwen3.6-35B-A3B-AWQ}"
MS_REPO="${MS_REPO:-Qwen/Qwen3.6-35B-A3B}"
HF_REPO="${HF_REPO:-Qwen/Qwen3.6-35B-A3B}"
DOWNLOAD_SOURCE="${DOWNLOAD_SOURCE:-modelscope}"
DOWNLOAD_IF_MISSING="${DOWNLOAD_IF_MISSING:-1}"
ALLOW_OVERWRITE="${ALLOW_OVERWRITE:-1}"
DOCKER_IMAGE="${DOCKER_IMAGE:-lynn-eval-base:cu13}"

mkdir -p "$REPORT_DIR"
cd "$WORKDIR"

exec > >(tee -a "$LOG") 2>&1

echo "[spark-qwen36] start $(date)"
echo "[spark-qwen36] bf16=$BF16_MODEL"
echo "[spark-qwen36] out=$OUT_MODEL"
echo "[spark-qwen36] awq=$AWQ_MODEL"
df -h "$MODEL_ROOT" "$REPORT_ROOT" 2>/dev/null || true
free -h || true

validate_official_layout() {
  local model="$1"
  python3 - "$model" <<'PY'
import json
import re
import sys
from pathlib import Path

model = Path(sys.argv[1])
index_path = model / "model.safetensors.index.json"
bad_layers = sorted(model.glob("layers-*.safetensors"))
if bad_layers:
    print(json.dumps({
        "ok": False,
        "reason": "internal_layer_split_artifact",
        "sample": [p.name for p in bad_layers[:5]],
    }, ensure_ascii=False))
    raise SystemExit(1)
if not index_path.exists():
    print(json.dumps({"ok": False, "reason": "missing_model_safetensors_index"}, ensure_ascii=False))
    raise SystemExit(1)
index = json.loads(index_path.read_text())
files = sorted(set(index.get("weight_map", {}).values()))
pat = re.compile(r"^model-\d{5}-of-\d{5}\.safetensors$")
bad_names = [f for f in files if not pat.match(f)]
missing = [f for f in files if not (model / f).exists()]
report = {
    "ok": not bad_names and not missing and bool(files),
    "model": str(model),
    "file_count": len(files),
    "weight_count": len(index.get("weight_map", {})),
    "bad_names": bad_names[:10],
    "missing": missing[:10],
}
print(json.dumps(report, ensure_ascii=False))
if not report["ok"]:
    raise SystemExit(1)
PY
}

if ! validate_official_layout "$BF16_MODEL"; then
  if [[ "$DOWNLOAD_IF_MISSING" != "1" ]]; then
    echo "[spark-qwen36][fail] BF16 is not official layout and download is disabled"
    exit 2
  fi
  echo "[spark-qwen36] downloading official BF16 into clean path"
  mkdir -p "$BF16_MODEL"
  python3 - "$BF16_MODEL" "$DOWNLOAD_SOURCE" "$MS_REPO" "$HF_REPO" <<'PY'
import pathlib
import sys

target = pathlib.Path(sys.argv[1])
source = sys.argv[2].strip().lower()
ms_repo = sys.argv[3]
hf_repo = sys.argv[4]
target.mkdir(parents=True, exist_ok=True)

if source in {"modelscope", "ms"}:
    from modelscope import snapshot_download
    got = snapshot_download(ms_repo, local_dir=str(target))
elif source in {"hf", "huggingface"}:
    from huggingface_hub import snapshot_download
    got = snapshot_download(repo_id=hf_repo, local_dir=str(target), max_workers=8)
else:
    raise SystemExit(f"unsupported DOWNLOAD_SOURCE={source!r}")
print({"repo": ms_repo if source in {"modelscope", "ms"} else hf_repo, "local_dir": got})
PY
fi

echo "[spark-qwen36] validating official BF16 layout"
validate_official_layout "$BF16_MODEL" | tee "$REPORT_DIR/spark_qwen36_35b_bf16_layout_${STAMP}.json"

pack_args=(scripts/a100_pack_lynn_native_nvfp4.py --src-model "/models/$(basename "$BF16_MODEL")" --out-model "/models/$(basename "$OUT_MODEL")")
if [[ "$ALLOW_OVERWRITE" == "1" ]]; then
  pack_args+=(--overwrite)
fi

echo "[spark-qwen36] pack official BF16 -> Lynn-native W4A16"
sudo docker run --rm \
  -v "$WORKDIR:/work" \
  -v "$MODEL_ROOT:/models" \
  -v "$REPORT_ROOT:/reports" \
  "$DOCKER_IMAGE" \
  bash -lc "cd /work && python3 ${pack_args[*]}" \
  | tee "$REPORT_DIR/spark_qwen36_35b_w4a16_pack_${STAMP}.json"

echo "[spark-qwen36] compare Lynn-native W4A16 against existing AWQ"
sudo docker run --rm \
  -v "$WORKDIR:/work" \
  -v "$MODEL_ROOT:/models" \
  -v "$REPORT_ROOT:/reports" \
  "$DOCKER_IMAGE" \
  bash -lc "python3 - /models/$(basename "$OUT_MODEL") /models/$(basename "$AWQ_MODEL") /reports/$(basename "$REPORT_DIR")/spark_qwen36_35b_w4a16_vs_awq_${STAMP}.json" <<'PY'
import json
import sys
from pathlib import Path

native = Path(sys.argv[1])
awq = Path(sys.argv[2])
out = Path(sys.argv[3])

def index_summary(path: Path) -> dict:
    idx = json.loads((path / "model.safetensors.index.json").read_text())
    files = sorted(set(idx.get("weight_map", {}).values()))
    return {
        "path": str(path),
        "exists": path.exists(),
        "total_size_index": int(idx.get("metadata", {}).get("total_size", 0)),
        "disk_bytes": sum((path / f).stat().st_size for f in files if (path / f).exists()),
        "shard_count": len(files),
        "weight_count": len(idx.get("weight_map", {})),
    }

report = {
    "schema_version": "spark-qwen36-w4a16-vs-awq-v1",
    "native_w4a16": index_summary(native),
    "awq_w4a16": index_summary(awq),
}

cfg_path = awq / "config.json"
if cfg_path.exists():
    cfg = json.loads(cfg_path.read_text())
    report["awq_w4a16"]["name_or_path"] = cfg.get("name_or_path")
    report["awq_w4a16"]["quantization_config"] = cfg.get("quantization_config")

manifest_path = native / "lynn_quant_manifest.json"
if manifest_path.exists():
    manifest = json.loads(manifest_path.read_text())
    report["native_w4a16"]["quantized_count"] = manifest.get("quantized_count")
    report["native_w4a16"]["kept_count"] = manifest.get("kept_count")
    report["native_w4a16"]["quantization"] = manifest.get("quantization")
    report["native_w4a16"]["runtime_contract"] = manifest.get("runtime_contract")

out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(report, ensure_ascii=False, indent=2))
PY

echo "[spark-qwen36] done $(date)"
