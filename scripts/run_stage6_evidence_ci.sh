#!/usr/bin/env bash
# GPU-free evidence checks for Stage 6 contracts/tooling.
set -euo pipefail

cd "$(dirname "$0")/.."
python3 scripts/test_stage6_p2o_evidence_tools.py
python3 scripts/test_stage6_p3_contract_static.py
