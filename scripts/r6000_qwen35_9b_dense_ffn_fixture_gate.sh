#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/root/autodl-tmp/lynn-engine}"
PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/conda-envs/r6000-eval/bin/python}"
MODEL="${MODEL:-/root/autodl-tmp/models/Qwen3.5-9B-lynn-native-w4a16-nvfp4-v0}"
REPORT_DIR="${REPORT_DIR:-/root/autodl-tmp/reports/qwen35_9b}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
LAYERS="${LAYERS:-0,8,16,-1}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-4096}"
FIXTURE_DIR="${FIXTURE_DIR:-${REPORT_DIR}/p159_dense_ffn_fixtures_${STAMP}}"
P160_JSON="${P160_JSON:-${REPORT_DIR}/p160_dense_ffn_fixture_contract_${STAMP}.json}"
SUMMARY_JSON="${SUMMARY_JSON:-${REPORT_DIR}/p159_p160_dense_ffn_fixture_summary_${STAMP}.json}"
EXPORT_WEIGHTS="${EXPORT_WEIGHTS:-0}"

cd "$ROOT"
mkdir -p "$REPORT_DIR"

echo "[p159/p160] root=$ROOT"
echo "[p159/p160] python=$PYTHON_BIN"
echo "[p159/p160] model=$MODEL"
echo "[p159/p160] layers=$LAYERS"
echo "[p159/p160] fixture_dir=$FIXTURE_DIR"

export_args=()
if [[ "$EXPORT_WEIGHTS" == "1" ]]; then
  export_args+=(--export-weights)
fi

"$PYTHON_BIN" benchmarks/p159_qwen35_9b_dense_ffn_fixture_export.py \
  --model "$MODEL" \
  --layers "$LAYERS" \
  --max-seq-len "$MAX_SEQ_LEN" \
  --out "$FIXTURE_DIR" \
  --export-intermediates \
  "${export_args[@]}"

"$PYTHON_BIN" benchmarks/p160_qwen35_9b_dense_ffn_fixture_contract.py \
  --fixtures "$FIXTURE_DIR" \
  --model "$MODEL" \
  --out "$P160_JSON" \
  --warmup 8 \
  --repeat 32

"$PYTHON_BIN" - "$FIXTURE_DIR" "$P160_JSON" "$SUMMARY_JSON" <<'PY'
import json, sys
from pathlib import Path

fixture_dir = Path(sys.argv[1])
p160 = Path(sys.argv[2])
out = Path(sys.argv[3])
manifest = json.loads((fixture_dir / "manifest.json").read_text())
contract = json.loads(p160.read_text())
summary = {
    "schema": "lynn-qwen35-9b-dense-ffn-fixture-gate-summary-v1",
    "fixture_dir": str(fixture_dir),
    "p160_json": str(p160),
    "total_fixtures": manifest.get("total_fixtures"),
    "selected_layers": manifest.get("selected_layers"),
    "p160_decision": contract.get("decision"),
    "p160_passed": contract.get("passed"),
    "p160_total": contract.get("total"),
    "p160_exact": contract.get("exact"),
    "ref_ms_mean": contract.get("ref_ms_mean"),
    "max_abs_max": contract.get("max_abs_max"),
    "cosine_min": contract.get("cosine_min"),
    "next_gate": "candidate_output_dir_or_native_dense_ffn_kernel" if contract.get("decision") == "DENSE_FFN_FIXTURE_GREEN" else "fix_fixture_contract",
}
out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(summary, ensure_ascii=False, indent=2))
PY

echo "[p159/p160] done summary=$SUMMARY_JSON"
