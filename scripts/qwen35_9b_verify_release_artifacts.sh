#!/usr/bin/env bash
set -euo pipefail

# Qwen3.5-9B release artifact verification script.
#
# Reads the release artifact manifest JSON, resolves paths against a model root,
# and verifies file existence, sizes, and optional SHA256 checksums.
#
# Usage:
#   bash scripts/qwen35_9b_verify_release_artifacts.sh --model-root /path/to/models
#   bash scripts/qwen35_9b_verify_release_artifacts.sh --model-root /path --out /path/to/report.json
#   bash scripts/qwen35_9b_verify_release_artifacts.sh --model-root /path --verify-checksums
#
# Exit codes:
#   0 = all checks passed
#   1 = one or more checks failed
#   2 = script error (missing deps, bad args)

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$ROOT"

MANIFEST_DEFAULT="$ROOT/reports/qwen35_9b/qwen35_9b_release_artifact_manifest.json"
MANIFEST=""
MODEL_ROOT=""
OUT=""
VERIFY_CHECKSUMS=0
TOLERANCE_GIB=0.5

while [[ $# -gt 0 ]]; do
  case "$1" in
    --manifest)      MANIFEST="$2"; shift 2 ;;
    --model-root)    MODEL_ROOT="$2"; shift 2 ;;
    --out)           OUT="$2"; shift 2 ;;
    --verify-checksums) VERIFY_CHECKSUMS=1; shift ;;
    --tolerance-gib) TOLERANCE_GIB="$2"; shift 2 ;;
    *) echo "[ERROR] unknown arg: $1" >&2; exit 2 ;;
  esac
done

MANIFEST="${MANIFEST:-$MANIFEST_DEFAULT}"
if [[ ! -f "$MANIFEST" ]]; then
  echo "[ERROR] manifest not found: $MANIFEST" >&2; exit 2
fi

if [[ -z "$MODEL_ROOT" ]]; then
  echo "[ERROR] --model-root is required" >&2; exit 2
fi
MODEL_ROOT="$(cd "$MODEL_ROOT" 2>/dev/null && pwd)" || {
  echo "[ERROR] model root not found: $MODEL_ROOT" >&2; exit 2
}

OUT="${OUT:-$ROOT/reports/qwen35_9b/qwen35_9b_release_artifact_verification_$(date +%Y%m%d_%H%M%S).json}"

python3 - "$MANIFEST" "$MODEL_ROOT" "$OUT" "$VERIFY_CHECKSUMS" "$TOLERANCE_GIB" <<'PYEOF'
import json
import os
import sys
import time
import hashlib
import subprocess
from pathlib import Path, PurePosixPath

manifest_path = sys.argv[1]
model_root = Path(sys.argv[2])
out_path = sys.argv[3]
verify_checksums = int(sys.argv[4]) == 1
tolerance_gib = float(sys.argv[5])

with open(manifest_path, encoding="utf-8") as f:
    manifest = json.load(f)

schema = manifest.get("schema", "")
model_id = manifest.get("model_id", "unknown")
artifacts = manifest.get("artifacts", {})
now = time.strftime("%Y-%m-%dT%H:%M:%S%z")

# --- helpers ---
def file_size_gib(path):
    return path.stat().st_size / (1024 ** 3)

def dir_size_gib(path):
    total = 0
    for f in path.rglob("*"):
        if f.is_file():
            total += f.stat().st_size
    return total / (1024 ** 3)

def sha256_file(path, chunk_size=1024*1024):
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()

def check_lynn_quant_manifest(dirpath):
    """Parse lynn_quant_manifest.json inside an NVFP4 directory."""
    lqm = dirpath / "lynn_quant_manifest.json"
    if not lqm.exists():
        return None
    with open(lqm, encoding="utf-8") as f:
        data = json.load(f)
    return {
        "schema_version": data.get("schema_version"),
        "quantized_count": data.get("quantized_count"),
        "kept_count": data.get("kept_count"),
        "output_shards": data.get("output_shards"),
        "pack_elapsed_seconds": data.get("pack_elapsed_seconds"),
    }

# --- verify each artifact ---
results = []
overall_ok = True

for art_id, art in sorted(artifacts.items()):
    art_type = art.get("type", "file")
    label = art.get("label", art_id)
    expected_gib = art.get("expected_size_gib")
    decision = art.get("decision", "UNKNOWN")
    result = {
        "artifact_id": art_id,
        "label": label,
        "type": art_type,
        "decision": decision,
        "classification": art.get("classification"),
        "checks": [],
        "ok": True,
    }

    if art_type == "file":
        fname = art.get("filename", "")
        resolved = model_root / fname
        result["resolved_path"] = str(resolved)
        result["path_hint"] = art.get("path_hint", "")

        # existence
        if not resolved.exists():
            result["checks"].append({"check": "existence", "ok": False, "detail": "file not found"})
            result["ok"] = False
        elif not resolved.is_file():
            result["checks"].append({"check": "existence", "ok": False, "detail": "path exists but is not a file"})
            result["ok"] = False
        else:
            result["checks"].append({"check": "existence", "ok": True, "detail": "file exists"})
            # size
            actual_gib = file_size_gib(resolved)
            result["actual_size_gib"] = round(actual_gib, 3)
            if expected_gib is not None:
                diff = abs(actual_gib - expected_gib)
                size_ok = diff <= tolerance_gib
                result["checks"].append({
                    "check": "size", "ok": size_ok,
                    "detail": f"{actual_gib:.3f} GiB (expected {expected_gib:.3f} ± {tolerance_gib})",
                })
                if not size_ok:
                    result["ok"] = False
            # sha256 (optional)
            if verify_checksums:
                sha = sha256_file(resolved)
                result["sha256"] = sha
                result["checks"].append({"check": "sha256", "ok": True, "detail": sha})

    elif art_type == "directory":
        dirname = art.get("dirname", "")
        resolved = model_root / dirname
        result["resolved_path"] = str(resolved)
        result["path_hint"] = art.get("path_hint", "")

        # existence
        if not resolved.exists():
            result["checks"].append({"check": "existence", "ok": False, "detail": "directory not found"})
            result["ok"] = False
        elif not resolved.is_dir():
            result["checks"].append({"check": "existence", "ok": False, "detail": "path exists but is not a directory"})
            result["ok"] = False
        else:
            result["checks"].append({"check": "existence", "ok": True, "detail": "directory exists"})
            # size
            actual_gib = dir_size_gib(resolved)
            result["actual_size_gib"] = round(actual_gib, 3)
            if expected_gib is not None:
                diff = abs(actual_gib - expected_gib)
                size_ok = diff <= tolerance_gib
                result["checks"].append({
                    "check": "size", "ok": size_ok,
                    "detail": f"{actual_gib:.3f} GiB (expected {expected_gib:.3f} ± {tolerance_gib})",
                })
                if not size_ok:
                    result["ok"] = False
            # critical files
            critical = art.get("critical_files", [])
            missing_critical = []
            for cf in critical:
                if not (resolved / cf).exists():
                    missing_critical.append(cf)
            if missing_critical:
                result["checks"].append({
                    "check": "critical_files", "ok": False,
                    "detail": f"missing: {', '.join(missing_critical)}",
                })
                result["ok"] = False
            else:
                result["checks"].append({
                    "check": "critical_files", "ok": True,
                    "detail": f"all {len(critical)} critical files present",
                })
            # lynn_quant_manifest (NVFP4 specific)
            lqm = check_lynn_quant_manifest(resolved)
            if lqm is not None:
                result["lynn_quant_manifest"] = lqm
            # sha256 (optional, on critical files only)
            if verify_checksums:
                sha_results = []
                for cf in critical:
                    cf_path = resolved / cf
                    if cf_path.exists() and cf_path.is_file():
                        sha_results.append({"file": cf, "sha256": sha256_file(cf_path)})
                result["sha256_critical_files"] = sha_results
                result["checks"].append({
                    "check": "sha256", "ok": True,
                    "detail": f"hashed {len(sha_results)} critical files",
                })

    if not result["ok"]:
        overall_ok = False
    results.append(result)

# --- build output report ---
report = {
    "schema": "lynn-qwen35-9b-release-artifact-verification-v1",
    "created": now,
    "model_id": model_id,
    "model_root": str(model_root),
    "manifest": os.path.basename(manifest_path),
    "verify_checksums": verify_checksums,
    "tolerance_gib": tolerance_gib,
    "overall_ok": overall_ok,
    "artifact_count": len(results),
    "ok_count": sum(1 for r in results if r["ok"]),
    "fail_count": sum(1 for r in results if not r["ok"]),
    "artifacts": results,
}

os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)
    f.write("\n")

# --- console summary ---
print(f"[verify-artifacts] model_root: {model_root}")
print(f"[verify-artifacts] manifest:   {os.path.basename(manifest_path)}")
for r in results:
    status = "✅" if r["ok"] else "❌"
    details = "; ".join(c["detail"] for c in r["checks"])
    print(f"  {status} {r['label']}: {details}")
print(f"[verify-artifacts] overall: {'ALL OK' if overall_ok else 'FAILURES DETECTED'}")
print(f"[verify-artifacts] report: {out_path}")

sys.exit(0 if overall_ok else 1)
PYEOF
