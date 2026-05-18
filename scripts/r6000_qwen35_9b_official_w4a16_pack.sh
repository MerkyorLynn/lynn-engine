#!/usr/bin/env bash
set -euo pipefail

# R6000 CPU/network pipeline for the official Qwen3.5-9B route:
#   official BF16 -> Lynn-native W4A16 NVFP4 artifact.
#
# This intentionally does not run generation/TPS gates. It is safe to run while
# GPU-side 35B kernel work continues, and leaves the 9B artifact ready for later
# quality and serving probes.

ROOT="${ROOT:-/root/autodl-tmp/lynn-engine}"
PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/conda-envs/r6000-eval/bin/python}"
MODEL_ROOT="${MODEL_ROOT:-/root/autodl-tmp/models}"
BF16_MODEL="${BF16_MODEL:-$MODEL_ROOT/Qwen3.5-9B-BF16}"
OUT_MODEL="${OUT_MODEL:-$MODEL_ROOT/Qwen3.5-9B-lynn-native-w4a16-nvfp4-v0}"
REPORT_ROOT="${REPORT_ROOT:-/root/autodl-tmp/reports}"
REPORT_DIR="${REPORT_DIR:-$REPORT_ROOT/qwen35_9b}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
LOG="${LOG:-$REPORT_DIR/r6000_qwen35_9b_official_w4a16_pack_${STAMP}.log}"

DOWNLOAD_SOURCE="${DOWNLOAD_SOURCE:-auto}"
MS_REPO="${MS_REPO:-Qwen/Qwen3.5-9B}"
HF_REPO="${HF_REPO:-Qwen/Qwen3.5-9B}"
DOWNLOAD_IF_MISSING="${DOWNLOAD_IF_MISSING:-1}"
ALLOW_OVERWRITE="${ALLOW_OVERWRITE:-1}"
MAX_SHARD_BYTES="${MAX_SHARD_BYTES:-2500000000}"

# Keep this pipeline CPU-only by default. Override CUDA_VISIBLE_DEVICES
# explicitly if a later smoke test is appended by the caller.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

mkdir -p "$REPORT_DIR" "$MODEL_ROOT"
cd "$ROOT"

download_model() {
  "$PYTHON_BIN" - "$BF16_MODEL" "$DOWNLOAD_SOURCE" "$MS_REPO" "$HF_REPO" <<'PY'
from __future__ import annotations

import pathlib
import sys

target = pathlib.Path(sys.argv[1])
source = sys.argv[2].strip().lower()
ms_repo = sys.argv[3]
hf_repo = sys.argv[4]
target.mkdir(parents=True, exist_ok=True)

errors: list[str] = []

def try_modelscope() -> bool:
    try:
        from modelscope import snapshot_download
        got = snapshot_download(ms_repo, local_dir=str(target))
        print({"source": "modelscope", "repo": ms_repo, "local_dir": got}, flush=True)
        return True
    except Exception as exc:  # noqa: BLE001 - surfaced in shell log.
        errors.append(f"modelscope:{type(exc).__name__}:{exc}")
        return False

def try_hf() -> bool:
    try:
        from huggingface_hub import snapshot_download
        got = snapshot_download(repo_id=hf_repo, local_dir=str(target), max_workers=8)
        print({"source": "huggingface", "repo": hf_repo, "local_dir": got}, flush=True)
        return True
    except Exception as exc:  # noqa: BLE001 - surfaced in shell log.
        errors.append(f"huggingface:{type(exc).__name__}:{exc}")
        return False

if source in {"modelscope", "ms"}:
    ok = try_modelscope()
elif source in {"hf", "huggingface"}:
    ok = try_hf()
elif source == "auto":
    ok = try_modelscope() or try_hf()
else:
    raise SystemExit(f"unsupported DOWNLOAD_SOURCE={source!r}")

if not ok:
    raise SystemExit("download failed: " + " | ".join(errors))
PY
}

validate_model() {
  "$PYTHON_BIN" - "$BF16_MODEL" "$REPORT_DIR/r6000_qwen35_9b_bf16_validate_${STAMP}.json" <<'PY'
from __future__ import annotations

import json
from pathlib import Path
import sys

from safetensors import safe_open

model = Path(sys.argv[1])
out = Path(sys.argv[2])
index_path = model / "model.safetensors.index.json"

internal_layers = sorted(p.name for p in model.glob("layers-*.safetensors"))
if not index_path.exists():
    report = {
        "schema_version": "qwen35-9b-bf16-safetensors-validate-v1",
        "model": str(model),
        "ok": False,
        "reason": "missing model.safetensors.index.json",
        "internal_layer_split_files": internal_layers[:8],
    }
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    raise SystemExit(2)

index = json.loads(index_path.read_text(encoding="utf-8"))
files = sorted(set(index.get("weight_map", {}).values()))
rows = []
for rel in files:
    path = model / rel
    rec = {
        "file": rel,
        "exists": path.exists(),
        "bytes": path.stat().st_size if path.exists() else 0,
    }
    if path.exists():
        with safe_open(path, framework="pt", device="cpu") as f:
            rec["tensor_count"] = len(list(f.keys()))
    rows.append(rec)

ok = bool(files) and all(r["exists"] and r.get("tensor_count", 0) > 0 for r in rows)
report = {
    "schema_version": "qwen35-9b-bf16-safetensors-validate-v1",
    "model": str(model),
    "ok": ok,
    "file_count": len(files),
    "total_bytes": sum(r["bytes"] for r in rows),
    "internal_layer_split_files": internal_layers[:8],
    "files": rows,
}
out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({
    "ok": ok,
    "file_count": report["file_count"],
    "total_gib": round(report["total_bytes"] / (1024 ** 3), 3),
}, ensure_ascii=False), flush=True)
if not ok:
    raise SystemExit(2)
PY
}

write_summary() {
  "$PYTHON_BIN" - "$BF16_MODEL" "$OUT_MODEL" "$REPORT_DIR" "$STAMP" <<'PY'
from __future__ import annotations

import json
from pathlib import Path
import sys

bf16 = Path(sys.argv[1])
out_model = Path(sys.argv[2])
report_dir = Path(sys.argv[3])
stamp = sys.argv[4]

def dir_bytes(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())

manifest_path = out_model / "lynn_quant_manifest.json"
manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
summary = {
    "schema_version": "qwen35-9b-official-w4a16-pack-summary-v1",
    "stamp": stamp,
    "bf16_model": str(bf16),
    "w4a16_model": str(out_model),
    "bf16_bytes": dir_bytes(bf16) if bf16.exists() else 0,
    "w4a16_bytes": dir_bytes(out_model) if out_model.exists() else 0,
    "quantized_count": manifest.get("quantized_count"),
    "kept_count": manifest.get("kept_count"),
    "output_shards": manifest.get("output_shards"),
    "runtime_contract": manifest.get("runtime_contract"),
    "pack_elapsed_seconds": manifest.get("elapsed_seconds"),
}
path = report_dir / f"r6000_qwen35_9b_official_w4a16_pack_summary_{stamp}.json"
path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps({
    "summary": str(path),
    "bf16_gib": round(summary["bf16_bytes"] / (1024 ** 3), 3),
    "w4a16_gib": round(summary["w4a16_bytes"] / (1024 ** 3), 3),
    "quantized_count": summary["quantized_count"],
}, ensure_ascii=False), flush=True)
PY
}

{
  echo "[qwen35-9b-pack] start $(date)"
  echo "[qwen35-9b-pack] root=$ROOT"
  echo "[qwen35-9b-pack] bf16=$BF16_MODEL"
  echo "[qwen35-9b-pack] out_model=$OUT_MODEL"
  echo "[qwen35-9b-pack] download_source=$DOWNLOAD_SOURCE ms=$MS_REPO hf=$HF_REPO"
  df -h "$MODEL_ROOT" "$REPORT_ROOT" 2>/dev/null || true

  if [[ ! -f "$BF16_MODEL/model.safetensors.index.json" ]]; then
    if [[ "$DOWNLOAD_IF_MISSING" != "1" ]]; then
      echo "[qwen35-9b-pack][fail] missing BF16 model index: $BF16_MODEL" >&2
      exit 2
    fi
    echo "[qwen35-9b-pack] downloading official BF16"
    download_model
  else
    echo "[qwen35-9b-pack] BF16 index already present; skip download"
  fi

  echo "[qwen35-9b-pack] validating official safetensors layout"
  validate_model

  pack_args=(scripts/a100_pack_lynn_native_nvfp4.py
    --src-model "$BF16_MODEL"
    --out-model "$OUT_MODEL"
    --max-shard-bytes "$MAX_SHARD_BYTES")
  if [[ "$ALLOW_OVERWRITE" == "1" ]]; then
    pack_args+=(--overwrite)
  fi

  echo "[qwen35-9b-pack] packing BF16 -> Lynn-native W4A16 NVFP4"
  "$PYTHON_BIN" "${pack_args[@]}" \
    > "$REPORT_DIR/r6000_qwen35_9b_w4a16_pack_${STAMP}.json"

  echo "[qwen35-9b-pack] manifest probes"
  "$PYTHON_BIN" benchmarks/quant_manifest_probe.py "$BF16_MODEL" \
    --out "$REPORT_DIR/r6000_qwen35_9b_bf16_manifest_${STAMP}.json"
  "$PYTHON_BIN" benchmarks/quant_manifest_probe.py "$OUT_MODEL" \
    --out "$REPORT_DIR/r6000_qwen35_9b_w4a16_manifest_${STAMP}.json"

  echo "[qwen35-9b-pack] summary"
  write_summary

  echo "[qwen35-9b-pack] done $(date)"
} 2>&1 | tee "$LOG"

echo "$LOG"
