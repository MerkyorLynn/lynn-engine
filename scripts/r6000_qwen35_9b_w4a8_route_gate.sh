#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/root/autodl-tmp/lynn-engine}"
PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/conda-envs/r6000-eval/bin/python}"
MODEL="${MODEL:-/root/autodl-tmp/models/Qwen3.5-9B-lynn-native-w4a16-nvfp4-v0}"
REPORT_DIR="${REPORT_DIR:-/root/autodl-tmp/reports/qwen35_9b}"
PROMPTS_JSON="${PROMPTS_JSON:-${ROOT}/scripts/qwen36_structured_hard_prompts_70.json}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
LIMIT="${LIMIT:-70}"
MAX_NEW="${MAX_NEW:-64}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-4096}"
WARMUP="${WARMUP:-8}"
REPEAT="${REPEAT:-32}"

FIXTURE_DIR="${FIXTURE_DIR:-}"
P185_JSON="${P185_JSON:-${REPORT_DIR}/p185_qwen35_9b_dense_w4a8_fixture_gate_${STAMP}.json}"
P186_JSON="${P186_JSON:-${REPORT_DIR}/p186_qwen35_9b_dense_w4a8_resident_gate_${STAMP}.json}"
SUMMARY_JSON="${SUMMARY_JSON:-${REPORT_DIR}/qwen35_9b_w4a8_route_gate_summary_${STAMP}.json}"

cd "$ROOT"
mkdir -p "$REPORT_DIR"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

if [[ -z "$FIXTURE_DIR" ]]; then
  FIXTURE_DIR="$(find "$REPORT_DIR" -maxdepth 1 -type d -name 'p159_dense_ffn_fixtures_*' | sort | tail -n 1 || true)"
fi
if [[ -z "$FIXTURE_DIR" || ! -f "$FIXTURE_DIR/manifest.json" ]]; then
  FIXTURE_DIR="${REPORT_DIR}/p159_dense_ffn_fixtures_w4a8_${STAMP}"
  echo "[w4a8-route] no existing P159 fixture found; exporting $FIXTURE_DIR"
  "$PYTHON_BIN" benchmarks/p159_qwen35_9b_dense_ffn_fixture_export.py \
    --model "$MODEL" \
    --layers "0,8,16,-1" \
    --max-seq-len "$MAX_SEQ_LEN" \
    --out "$FIXTURE_DIR" \
    --export-intermediates
fi

echo "[w4a8-route] root=$ROOT"
echo "[w4a8-route] model=$MODEL"
echo "[w4a8-route] fixture_dir=$FIXTURE_DIR"
echo "[w4a8-route] p185=$P185_JSON"
echo "[w4a8-route] p186=$P186_JSON"

"$PYTHON_BIN" benchmarks/p185_qwen35_9b_dense_w4a8_fixture_gate.py \
  --fixtures "$FIXTURE_DIR" \
  --model "$MODEL" \
  --out "$P185_JSON" \
  --warmup "$WARMUP" \
  --repeat "$REPEAT"

"$PYTHON_BIN" benchmarks/p186_qwen35_9b_dense_w4a8_resident_gate.py \
  --model "$MODEL" \
  --prompts-json "$PROMPTS_JSON" \
  --limit "$LIMIT" \
  --max-new "$MAX_NEW" \
  --max-seq-len "$MAX_SEQ_LEN" \
  --out "$P186_JSON"

"$PYTHON_BIN" - "$P185_JSON" "$P186_JSON" "$SUMMARY_JSON" <<'PY'
import json, sys
from pathlib import Path

p185 = Path(sys.argv[1])
p186 = Path(sys.argv[2])
out = Path(sys.argv[3])
fixture = json.loads(p185.read_text())
resident = json.loads(p186.read_text())

def by_label(rows):
    return {r.get("label"): r for r in rows}

summaries = by_label(resident.get("summaries", []))
summary = {
    "schema": "lynn-qwen35-9b-w4a8-route-gate-summary-v1",
    "p185_json": str(p185),
    "p186_json": str(p186),
    "fixture_decision": fixture.get("decision"),
    "fixture_summaries": fixture.get("summaries"),
    "resident_decision": resident.get("decision"),
    "resident_summaries": resident.get("summaries"),
    "resident_gateup_exact": resident.get("comparison_gateup", {}).get("exact_count"),
    "resident_gateup_total": resident.get("comparison_gateup", {}).get("total"),
    "resident_gateup_min_prefix": resident.get("comparison_gateup", {}).get("min_prefix"),
    "resident_gateup_mean_prefix": resident.get("comparison_gateup", {}).get("mean_prefix"),
    "resident_full_exact": resident.get("comparison_full", {}).get("exact_count"),
    "resident_full_total": resident.get("comparison_full", {}).get("total"),
    "resident_full_min_prefix": resident.get("comparison_full", {}).get("min_prefix"),
    "resident_full_mean_prefix": resident.get("comparison_full", {}).get("mean_prefix"),
    "decode_tps": {
        "w4a16_reference": summaries.get("convstrict_w4a16_reference", {}).get("decode_tps_mean"),
        "w4a8_gateup_fake_quant": summaries.get("convstrict_w4a8_gateup", {}).get("decode_tps_mean"),
        "w4a8_full_fake_quant": summaries.get("convstrict_w4a8_full", {}).get("decode_tps_mean"),
    },
    "note": "Fake-quant TPS includes FP8 round-trip emulation overhead; it is not the target native FP8-active speed.",
}
if summary["fixture_decision"] in {"DENSE_W4A8_FIXTURE_GREEN", "DENSE_W4A8_GATEUP_GREEN_FULL_AMBER"}:
    if summary["resident_decision"] in {"DENSE_W4A8_FULL_GENERATION_EXACT", "DENSE_W4A8_GATEUP_EXACT_FULL_DRIFT", "DENSE_W4A8_GENERATION_AMBER_PREFIX"}:
        summary["next_gate"] = "build_native_fp8_active_dense_kernel_or_resident_w4a8_service_probe"
    else:
        summary["next_gate"] = "analyze_resident_generation_drift_before_kernel_work"
else:
    summary["next_gate"] = "do_not_promote_w4a8_dense_route"

out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(summary, ensure_ascii=False, indent=2))
PY

echo "[w4a8-route] done summary=$SUMMARY_JSON"
