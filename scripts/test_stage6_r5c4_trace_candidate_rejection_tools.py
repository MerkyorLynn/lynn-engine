#!/usr/bin/env python3
"""GPU-free checks for R5-C4 trace-derived candidate rejection tooling."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT / "scripts" / "stage6_r5c4_trace_candidate_rejection.py"
SUMMARY = ROOT / "scripts" / "summarize_stage6_r5c4_trace_candidate_rejection.py"


def main() -> int:
    failures: list[str] = []
    for path, needles in {
        PROBE: [
            "PASS_R5C4_TRACE_DERIVED_CANDIDATE_REJECTED",
            "FAIL_R5C4_FULL_ACTIVE_MOE_PREFILL_SPEED_AB",
            "same_scope_ab",
            "real_model_weights",
            "candidate_full_active_moe_boundary_timed",
            "decode_tps_not_banked",
        ],
        SUMMARY: [
            "PASS_R5C4_TRACE_DERIVED_CANDIDATE_REJECTED",
            "Validator rejected trace candidate",
            "strict-exit",
        ],
    }.items():
        if not path.exists():
            failures.append(f"missing {path.relative_to(ROOT)}")
            continue
        text = path.read_text(encoding="utf-8")
        for needle in needles:
            if needle not in text:
                failures.append(f"{path.relative_to(ROOT)} missing {needle!r}")

    with tempfile.TemporaryDirectory() as tmp_s:
        tmp = Path(tmp_s)
        result = tmp / "result.json"
        summary = tmp / "summary.md"
        proc = subprocess.run(
            [sys.executable, str(PROBE), "--root", str(ROOT), "--out", str(result)],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if proc.returncode != 0:
            failures.append(f"probe failed: {proc.stderr or proc.stdout}")
        data = json.loads(result.read_text(encoding="utf-8"))
        if data.get("decision") != "PASS_R5C4_TRACE_DERIVED_CANDIDATE_REJECTED":
            failures.append(f"unexpected probe decision {data.get('decision')}")
        ok = subprocess.run(
            [sys.executable, str(SUMMARY), str(result), "--markdown-out", str(summary), "--strict-exit"],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if ok.returncode != 0:
            failures.append(f"summary strict failed: {ok.stderr or ok.stdout}")
        if "does not bank full active-MoE prefill speed" not in summary.read_text(encoding="utf-8"):
            failures.append("summary missing non-claim boundary")

    if failures:
        print("Stage 6 R5-C4 trace-derived candidate rejection tooling self-test FAILED")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print("Stage 6 R5-C4 trace-derived candidate rejection tooling self-test PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
