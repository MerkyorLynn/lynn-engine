#!/usr/bin/env python3
"""Local self-test for the Stage 6 evidence ledger."""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(
        cmd,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        raise AssertionError(
            f"command failed with {proc.returncode}: {cmd}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
    return proc


def by_gate(data: dict) -> dict[str, dict]:
    return {gate["gate"]: gate for gate in data["gates"]}


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp_s:
        tmp = Path(tmp_s)
        md = tmp / "ledger.md"
        js = tmp / "ledger.json"
        run([
            sys.executable,
            "scripts/write_stage6_evidence_ledger.py",
            "--markdown-out",
            str(md),
            "--json-out",
            str(js),
            "--spark-status",
            "TEST_BLOCKED_BY_SSH",
            "--spark-note",
            "fixture",
        ])
        data = json.loads(js.read_text(encoding="utf-8"))
        gates = by_gate(data)
        text = md.read_text(encoding="utf-8")

        assert data["schema"] == "lynn-stage6-evidence-ledger-v1"
        assert data["promotion_boundaries"]["p4_banked_fused_kernel"] is False
        assert data["promotion_boundaries"]["p4_default_promotion"] is False
        assert gates["stage6_60g_decode_release"]["status"] == "BANKED"
        assert gates["p1a_batched_dense_rejected"]["status"] == "CLOSED_NEGATIVE"
        assert gates["p2ka_gated_delta_loop_rejected"]["status"] == "CLOSED_NEGATIVE"
        assert gates["p2n_wider_layer_block_linear"]["status"] == "BANKED"
        assert gates["p3e_rc_quality_battery"]["status"] == "READY_WAITING_SPARK"
        assert gates["p4_runtime_bridge_preflight"]["status"] == "READY_WAITING_SPARK"
        assert gates["p4b_single_kernel_preflight"]["status"] == "READY_WAITING_SPARK"
        assert gates["p4b_single_kernel_contract"]["status"] == "CONTRACT_READY_UNIMPLEMENTED"
        assert "P4 fused kernel banked | `false`" in text
        assert "TEST_BLOCKED_BY_SSH" in text

    print("Stage 6 evidence ledger self-test PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
