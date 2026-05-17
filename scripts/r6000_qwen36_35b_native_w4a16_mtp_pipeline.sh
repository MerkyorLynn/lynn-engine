#!/usr/bin/env bash
set -euo pipefail

# End-to-end R6000 probe for the official Qwen3.6-35B-A3B route:
#   official BF16 -> Lynn-native W4A16 NVFP4 -> W4A16/W4A8 matrix -> official MTP.
#
# This supersedes the temporary V4-Pro distill pivot. The win condition is now:
# W4A16/NVFP4 quality tracks Q4_K_M/FP8 while keeping Lynn native runtime and
# direct official MTP attachment.

ROOT="${ROOT:-/root/autodl-tmp/lynn-engine}"
PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/conda-envs/r6000-eval/bin/python}"
BF16_MODEL="${BF16_MODEL:-/root/autodl-tmp/models/Qwen3.6-35B-A3B-BF16}"
OUT_MODEL="${OUT_MODEL:-/root/autodl-tmp/models/Qwen3.6-35B-A3B-lynn-native-w4a16-nvfp4-v0}"
SIDECAR_DIR="${SIDECAR_DIR:-/root/autodl-tmp/models/mtp_sidecars/qwen36-35b-a3b-mtp-official}"
SIDECAR_FILE="${SIDECAR_FILE:-$SIDECAR_DIR/mtp.safetensors}"
REPORT_ROOT="${REPORT_ROOT:-/root/autodl-tmp/reports}"
REPORT_DIR="${REPORT_DIR:-$REPORT_ROOT/qwen36_35b_native_w4a16_mtp}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
LOG="${LOG:-$REPORT_DIR/r6000_qwen36_35b_native_w4a16_mtp_${STAMP}.log}"
DOWNLOAD_SOURCE="${DOWNLOAD_SOURCE:-modelscope}"
MS_REPO="${MS_REPO:-Qwen/Qwen3.6-35B-A3B}"
HF_REPO="${HF_REPO:-Qwen/Qwen3.6-35B-A3B}"
DOWNLOAD_IF_MISSING="${DOWNLOAD_IF_MISSING:-1}"
ALLOW_OVERWRITE="${ALLOW_OVERWRITE:-1}"

export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export LYNN_MOE_IMPL="${LYNN_MOE_IMPL:-packed_nvfp4}"
export LYNN_NATIVE_FP4_LM_HEAD="${LYNN_NATIVE_FP4_LM_HEAD:-1}"
export LYNN_NATIVE_ACTIVE_MOE="${LYNN_NATIVE_ACTIVE_MOE:-1}"
export LYNN_NATIVE_ACTIVE_MOE_BACKEND="${LYNN_NATIVE_ACTIVE_MOE_BACKEND:-packed_nvfp4}"
export LYNN_ROUTER_TOPK_SORTED="${LYNN_ROUTER_TOPK_SORTED:-1}"
export LYNN_MTP_LAYER_MOE="${LYNN_MTP_LAYER_MOE:-decode_slot_sorted}"
export LYNN_FULL_TOKEN_GRAPH_SLOT=0

mkdir -p "$REPORT_DIR"
cd "$ROOT"

{
  echo "[qwen36-35b] start $(date)"
  echo "[qwen36-35b] bf16=$BF16_MODEL"
  echo "[qwen36-35b] out_model=$OUT_MODEL"
  echo "[qwen36-35b] sidecar=$SIDECAR_FILE"
  echo "[qwen36-35b] download_source=$DOWNLOAD_SOURCE ms=$MS_REPO hf=$HF_REPO"
  df -h "$(dirname "$BF16_MODEL")" "$REPORT_ROOT" 2>/dev/null || true

  if [[ ! -f "$BF16_MODEL/model.safetensors.index.json" ]]; then
    if [[ "$DOWNLOAD_IF_MISSING" != "1" ]]; then
      echo "[qwen36-35b][fail] missing BF16 model index: $BF16_MODEL" >&2
      exit 2
    fi
    echo "[qwen36-35b] downloading official BF16"
    mkdir -p "$BF16_MODEL"
    "$PYTHON_BIN" - "$BF16_MODEL" "$DOWNLOAD_SOURCE" "$MS_REPO" "$HF_REPO" <<'PY'
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

  echo "[qwen36-35b] validate safetensors index"
  "$PYTHON_BIN" - "$BF16_MODEL" "$REPORT_DIR/r6000_qwen36_35b_bf16_validate_${STAMP}.json" <<'PY'
import json
from pathlib import Path
from safetensors import safe_open

model = Path(__import__("sys").argv[1])
out = Path(__import__("sys").argv[2])
index = json.loads((model / "model.safetensors.index.json").read_text())
files = sorted(set(index["weight_map"].values()))
rows = []
for rel in files:
    path = model / rel
    rec = {"file": rel, "exists": path.exists(), "bytes": path.stat().st_size if path.exists() else 0}
    if path.exists():
        with safe_open(path, framework="pt", device="cpu") as f:
            rec["tensor_count"] = len(list(f.keys()))
    rows.append(rec)
report = {
    "schema_version": "qwen36-35b-bf16-safetensors-validate-v1",
    "model": str(model),
    "file_count": len(files),
    "ok": all(r["exists"] and r.get("tensor_count", 0) > 0 for r in rows),
    "files": rows,
}
out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"ok": report["ok"], "file_count": report["file_count"]}, ensure_ascii=False))
if not report["ok"]:
    raise SystemExit(2)
PY

  pack_args=(scripts/a100_pack_lynn_native_nvfp4.py --src-model "$BF16_MODEL" --out-model "$OUT_MODEL")
  if [[ "$ALLOW_OVERWRITE" == "1" ]]; then
    pack_args+=(--overwrite)
  fi
  echo "[qwen36-35b] pack official BF16 -> Lynn-native W4A16"
  "$PYTHON_BIN" "${pack_args[@]}" > "$REPORT_DIR/r6000_qwen36_35b_w4a16_pack_${STAMP}.json"

  echo "[qwen36-35b] manifest probes"
  "$PYTHON_BIN" benchmarks/quant_manifest_probe.py "$BF16_MODEL" \
    --out "$REPORT_DIR/r6000_qwen36_35b_bf16_manifest_${STAMP}.json"
  "$PYTHON_BIN" benchmarks/quant_manifest_probe.py "$OUT_MODEL" \
    --out "$REPORT_DIR/r6000_qwen36_35b_w4a16_manifest_${STAMP}.json"

  echo "[qwen36-35b] BF16-vs-W4A16 logit smoke"
  "$PYTHON_BIN" benchmarks/p2_resident_logit_analysis.py \
    --bf16 "$BF16_MODEL" \
    --v8 "$OUT_MODEL" \
    --out "$REPORT_DIR/r6000_qwen36_35b_w4a16_resident_logit_${STAMP}.json" \
    --max-new "${LOGIT_MAX_NEW:-4}" \
    --top-k "${TOP_K:-10}"

  echo "[qwen36-35b] W4A16/W4A8 generation matrix"
  "$PYTHON_BIN" benchmarks/v4_w4a16_w4a8_generation_matrix.py \
    --model "$OUT_MODEL" \
    --out "$REPORT_DIR/r6000_qwen36_35b_w4a16_w4a8_generation_matrix_${STAMP}.json" \
    --device "${DEVICE:-cuda}" \
    --dtype "${DTYPE:-bf16}" \
    --max-new "${MATRIX_MAX_NEW:-48}" \
    --top-k "${TOP_K:-5}" \
    --use-chat-template

  if [[ -f "$SIDECAR_FILE" ]]; then
    echo "[qwen36-35b] official MTP shape audit"
    "$PYTHON_BIN" scripts/a100_mtp_sidecar_shape_audit.py \
      --sidecar-dir "$SIDECAR_DIR" \
      --base-model "$BF16_MODEL" \
      --out "$REPORT_DIR/r6000_qwen36_35b_mtp_shape_audit_${STAMP}.json"

    echo "[qwen36-35b] official MTP BF16 forward smoke"
    "$PYTHON_BIN" scripts/a100_mtp_forward_smoke.py \
      --base-model "$BF16_MODEL" \
      --sidecar-file "$SIDECAR_FILE" \
      --out "$REPORT_DIR/r6000_qwen36_35b_mtp_bf16_forward_smoke_${STAMP}.json" \
      --prompt "Return one JSON object with keys city and unit for Tokyo in metric units. No markdown." \
      --use-chat-template \
      --top-k "${TOP_K:-8}" \
      --dtype "${MTP_DTYPE:-bfloat16}"

    echo "[qwen36-35b] official MTP BF16 iterative accept"
    "$PYTHON_BIN" scripts/a100_mtp_iterative_accept_probe.py \
      --base-model "$BF16_MODEL" \
      --sidecar-file "$SIDECAR_FILE" \
      --out "$REPORT_DIR/r6000_qwen36_35b_mtp_bf16_iter_accept_${STAMP}.json" \
      --use-chat-template \
      --max-new "${MTP_MAX_NEW:-8}" \
      --top-k "${TOP_K:-8}" \
      --dtype "${MTP_DTYPE:-bfloat16}"

    echo "[qwen36-35b] official MTP W4A16 P107 shadow"
    "$PYTHON_BIN" benchmarks/p107_mtp_shadow_serving_credit_probe.py \
      --model "$OUT_MODEL" \
      --sidecar-file "$SIDECAR_FILE" \
      --out "$REPORT_DIR/r6000_qwen36_35b_mtp_w4a16_p107_shadow_${STAMP}.json" \
      --use-chat-template \
      --max-new "${P107_MAX_NEW:-16}" \
      --top-k "${P107_TOP_K:-8}" \
      --dtype "${MTP_DTYPE:-bfloat16}"
  else
    echo "[qwen36-35b][warn] missing sidecar; skipped MTP probes: $SIDECAR_FILE"
  fi

  echo "[qwen36-35b] done $(date)"
} 2>&1 | tee "$LOG"

echo "$LOG"
