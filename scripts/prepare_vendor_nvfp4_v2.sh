#!/usr/bin/env bash
set -euo pipefail

# Prepare/fail-loud wrapper for the vendor-friendly NVFP4 v2 artifact.
#
# This script intentionally does not overwrite the Lynn-native artifact. It
# checks the environment and model shape before delegating to a real ModelOpt /
# llmcompressor quantization script.

SRC_BF16="${SRC_BF16:-/root/autodl-tmp/models/lynn-27b-variable-recovery-step5000-bf16-final}"
OUT_VENDOR="${OUT_VENDOR:-/root/autodl-tmp/models/lynn-27b-variable-recovery-step5000-nvfp4-vendor-v2}"
PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/conda-envs/r6000-eval/bin/python}"
ALLOW_OVERWRITE="${ALLOW_OVERWRITE:-0}"

echo "ARTIFACT_KIND=vendor_friendly"
echo "SOURCE_BF16=${SRC_BF16}"
echo "OUTPUT=${OUT_VENDOR}"
echo "ALLOW_OVERWRITE=${ALLOW_OVERWRITE}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "[fail] PYTHON_BIN not executable: ${PYTHON_BIN}" >&2
  exit 2
fi
if [[ ! -d "${SRC_BF16}" ]]; then
  echo "[fail] source BF16 directory missing: ${SRC_BF16}" >&2
  exit 2
fi
if [[ -e "${OUT_VENDOR}" && "${ALLOW_OVERWRITE}" != "1" ]]; then
  echo "[fail] output exists; set ALLOW_OVERWRITE=1 only after confirming it is not the Lynn-native artifact" >&2
  exit 2
fi
if [[ "${OUT_VENDOR}" == *"nvfp4-final" && "${OUT_VENDOR}" != *"vendor"* ]]; then
  echo "[fail] output name looks like canonical Lynn-native artifact; refusing" >&2
  exit 2
fi

"${PYTHON_BIN}" - <<'PY'
import importlib.util
import json
import os
from pathlib import Path

src = Path(os.environ.get("SRC_BF16", "/root/autodl-tmp/models/lynn-27b-variable-recovery-step5000-bf16-final"))
spec_path = src / "lynn_engine_variable_expert_spec.json"
spec = json.loads(spec_path.read_text()) if spec_path.exists() else {}
mods = {m: importlib.util.find_spec(m) is not None for m in ["modelopt", "llmcompressor", "compressed_tensors"]}
print("[env]", mods)
print("[variable_expert]", spec.get("hf_vanilla_compatible") is False)
print("[remaining_counts]", sorted(set(int(v) for v in spec.get("remaining_experts_by_layer", {}).values())) or None)
if not (mods["modelopt"] or mods["llmcompressor"]):
    raise SystemExit(
        "[fail] current Python env lacks ModelOpt/llmcompressor BF16->NVFP4 quantizer. "
        "Use a separate quantization env; compressed_tensors alone only converts already-quantized ModelOpt artifacts."
    )
if spec.get("hf_vanilla_compatible") is False:
    print("[warn] BF16 source is physical variable-expert. Vendor v2 requires padding/mask or vendor variable-expert support.")
PY

cat <<'EOF'
[next]
  1. If using ModelOpt: run BF16 -> ModelOpt NVFP4 in this separate vendor output dir.
  2. If needed, run compressed-tensors modelopt_nvfp4 converter.
  3. Run P57/P54-style numeric gates before serving/publishing.

This wrapper is only the guard rail; it does not perform quantization yet.
EOF
