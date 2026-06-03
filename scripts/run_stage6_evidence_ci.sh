#!/usr/bin/env bash
# GPU-free evidence checks for Stage 6 contracts/tooling.
set -euo pipefail

cd "$(dirname "$0")/.."
python3 scripts/test_stage6_p2o_evidence_tools.py
python3 scripts/test_stage6_p3_contract_static.py
python3 scripts/test_stage6_p3a_evidence_tools.py
python3 scripts/test_stage6_gpu_gate_suite.py
python3 scripts/test_stage6_p3b_contract_static.py
python3 scripts/test_stage6_p3b_evidence_tools.py
python3 scripts/test_stage6_p3c_evidence_tools.py
python3 scripts/test_stage6_p3d_evidence_tools.py
python3 scripts/test_stage6_p3e_evidence_tools.py
python3 scripts/test_stage6_p4_native_abi_static.py
python3 scripts/test_stage6_p4_evidence_tools.py
python3 scripts/test_stage6_p4_zero_shadow_firewall.py
