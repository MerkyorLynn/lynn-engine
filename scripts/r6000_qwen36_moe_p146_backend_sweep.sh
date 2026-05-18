#!/usr/bin/env bash
# R6000 Qwen3.6 Native MoE resident P37 backend sweep.
#
# This is an admission sweep, not a promotion gate.  It runs P146 with
# linear-block graph disabled and records whether each backend is P37 exact
# before any candidate is allowed to spend time on P25 or structured gates.
set -euo pipefail

REPO_DIR="${REPO_DIR:-/root/autodl-tmp/lynn-engine}"
PY="${PY:-/root/autodl-tmp/conda-envs/r6000-eval/bin/python}"
MODEL_DIR="${MODEL_DIR:-/root/autodl-tmp/models/Qwen3.6-35B-A3B-lynn-native-w4a16-nvfp4-v0}"
REPORT_DIR="${REPORT_DIR:-/root/autodl-tmp/reports/qwen36_35b}"
MAX_NEW="${MAX_NEW:-8}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-${REPORT_DIR}/p146_backend_sweep_${STAMP}}"
SUMMARY_OUT="${SUMMARY_OUT:-${REPORT_DIR}/p146_backend_sweep_${STAMP}_summary.json}"
ATTEMPTS_JSONL="${ATTEMPTS_JSONL:-${OUT_DIR}/attempts.jsonl}"

# Format:
#   backend
#   backend:KEY=VALUE,KEY2=VALUE2
#
# Keep already-closed graphsafe/pretransposed routes out of the default sweep.
# They are documented in P143/P146 resident reports; this wrapper focuses on
# remaining native active-MoE backends that can be admitted or closed uniformly.
DEFAULT_CANDIDATES=$'grouped_per16_nonatomic\ngrouped_per16_fused\ngrouped_per16\ncuda_scalar_contract\ncuda_scalar'
CANDIDATES="${CANDIDATES:-${DEFAULT_CANDIDATES}}"

if [ ! -d "${REPO_DIR}" ]; then
    echo "ERROR: repo not found: ${REPO_DIR}" >&2
    exit 1
fi
if [ ! -d "${MODEL_DIR}" ]; then
    echo "ERROR: model not found: ${MODEL_DIR}" >&2
    exit 1
fi

cd "${REPO_DIR}"
mkdir -p "${OUT_DIR}" "$(dirname "${SUMMARY_OUT}")"
rm -f "${ATTEMPTS_JSONL}"
"${PY}" -m py_compile benchmarks/p146_resident_moe_backend_p37_probe.py

echo "═══════════════════════════════════════════════════════════════════════"
echo " R6000 Qwen3.6 Native MoE P146 Backend Sweep"
echo "═══════════════════════════════════════════════════════════════════════"
echo " Repo:    ${REPO_DIR}"
echo " Model:   ${MODEL_DIR}"
echo " Out dir: ${OUT_DIR}"
echo " Summary: ${SUMMARY_OUT}"
echo " Max new: ${MAX_NEW}"
echo ""

while IFS= read -r spec; do
    [ -z "${spec}" ] && continue
    backend="${spec%%:*}"
    env_blob=""
    if [ "${spec}" != "${backend}" ]; then
        env_blob="${spec#*:}"
    fi
    safe_backend="$(printf '%s' "${backend}" | tr -c 'A-Za-z0-9_-' '_')"
    out="${OUT_DIR}/p146_${safe_backend}.json"
    log="${OUT_DIR}/p146_${safe_backend}.log"

    args=(
        benchmarks/p146_resident_moe_backend_p37_probe.py
        --model "${MODEL_DIR}"
        --candidate-backend "${backend}"
        --max-new "${MAX_NEW}"
        --out "${out}"
    )
    if [ -n "${env_blob}" ]; then
        IFS=',' read -r -a env_items <<< "${env_blob}"
        for item in "${env_items[@]}"; do
            [ -n "${item}" ] && args+=(--env "${item}")
        done
    fi

    echo "── P146 backend=${backend} env=${env_blob:-<none>}"
    set +e
    "${PY}" "${args[@]}" > "${log}" 2>&1
    rc=$?
    set -e
    echo "   rc=${rc} out=${out} log=${log}"
    "${PY}" - "${ATTEMPTS_JSONL}" "${backend}" "${env_blob}" "${rc}" "${out}" "${log}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
row = {
    "candidate_backend": sys.argv[2],
    "env_blob": sys.argv[3],
    "return_code": int(sys.argv[4]),
    "out": sys.argv[5],
    "log": sys.argv[6],
}
with path.open("a", encoding="utf-8") as f:
    f.write(json.dumps(row, ensure_ascii=False) + "\n")
PY
done <<< "${CANDIDATES}"

"${PY}" - "${OUT_DIR}" "${SUMMARY_OUT}" "${ATTEMPTS_JSONL}" <<'PY'
import json
import sys
from pathlib import Path

out_dir = Path(sys.argv[1])
summary_out = Path(sys.argv[2])
attempts_path = Path(sys.argv[3])
attempts = []
if attempts_path.exists():
    for line in attempts_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            attempts.append(json.loads(line))

attempts_by_out = {Path(a["out"]).resolve(): a for a in attempts}
rows = []
for path in sorted(out_dir.glob("p146_*.json")):
    attempt = attempts_by_out.get(path.resolve(), {})
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - defensive summary path
        rows.append({
            "path": str(path),
            "candidate_backend": attempt.get("candidate_backend"),
            "return_code": attempt.get("return_code"),
            "error": str(exc),
        })
        continue
    results = data.get("results") or []
    rows.append(
        {
            "path": str(path),
            "candidate_backend": data.get("candidate_backend"),
            "extra_env": data.get("extra_env", {}),
            "return_code": attempt.get("return_code"),
            "verdict": data.get("verdict"),
            "exact_count": data.get("exact_count"),
            "total_prompts": data.get("total_prompts"),
            "collapse_detected": data.get("collapse_detected"),
            "first_drift": next(
                (
                    {
                        "prompt_id": r.get("prompt_id"),
                        "drift_token_index": r.get("drift_token_index"),
                        "baseline_ids": r.get("baseline_ids"),
                        "candidate_ids": r.get("candidate_ids"),
                    }
                    for r in results
                    if not r.get("exact")
                ),
                None,
            ),
            "candidate_tps": [r.get("candidate_tps") for r in results],
            "baseline_tps": [r.get("baseline_tps") for r in results],
        }
    )

reported_outs = {Path(r["path"]).resolve() for r in rows if r.get("path")}
for attempt in attempts:
    out_path = Path(attempt["out"]).resolve()
    if out_path in reported_outs:
        continue
    log_path = Path(attempt["log"])
    tail = ""
    if log_path.exists():
        tail = "\n".join(log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-20:])
    rows.append(
        {
            "path": attempt["out"],
            "candidate_backend": attempt["candidate_backend"],
            "extra_env": attempt.get("env_blob") or {},
            "return_code": attempt.get("return_code"),
            "verdict": "ERROR_NO_REPORT",
            "exact_count": None,
            "total_prompts": None,
            "collapse_detected": None,
            "first_drift": None,
            "candidate_tps": [],
            "baseline_tps": [],
            "log": attempt["log"],
            "log_tail": tail,
        }
    )

summary = {
    "schema_version": "lynn-qwen36-p146-backend-sweep-v1",
    "out_dir": str(out_dir),
    "attempts_jsonl": str(attempts_path),
    "total_attempts": len(attempts),
    "total_reports": len(rows),
    "p37_exact_backends": [r.get("candidate_backend") for r in rows if r.get("verdict") == "P37_EXACT"],
    "closed_backends": [r.get("candidate_backend") for r in rows if str(r.get("verdict", "")).startswith("CLOSED")],
    "error_backends": [r.get("candidate_backend") for r in rows if str(r.get("verdict", "")).startswith("ERROR")],
    "rows": rows,
}
summary_out.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2, ensure_ascii=False))
PY

echo ""
echo "Summary: ${SUMMARY_OUT}"
