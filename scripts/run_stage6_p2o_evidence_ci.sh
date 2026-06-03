#!/usr/bin/env bash
# GPU-free CI entry point for Stage 6 P2-O evidence tooling.
set -euo pipefail

cd "$(dirname "$0")/.."
python3 scripts/test_stage6_p2o_evidence_tools.py
