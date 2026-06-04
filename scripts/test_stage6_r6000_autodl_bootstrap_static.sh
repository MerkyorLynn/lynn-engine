#!/usr/bin/env bash
# Static check for the AutoDL R6000 Stage6 bootstrap wrapper.
set -euo pipefail

cd "$(dirname "$0")/.."
bash -n scripts/r6000_stage6_autodl_bootstrap.sh
if command -v shellcheck >/dev/null 2>&1; then
  shellcheck scripts/r6000_stage6_autodl_bootstrap.sh
fi
echo "Stage 6 R6000 AutoDL bootstrap static check PASS"
