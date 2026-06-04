#!/usr/bin/env python3
"""GPU-free checks for the R5-C4 raw-metrics to candidate adapter."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
ADAPTER = ROOT / "scripts" / "stage6_r5c4_candidate_from_metrics.py"
TEMPLATE = ROOT / "scripts" / "stage6_r5c4_candidate_metrics_template.json"
VALIDATOR = ROOT / "scripts" / "r6000_stage6_r5c4_full_active_moe_speed_ab.py"
WRAPPER = ROOT / "scripts" / "r6000_stage6_r5c4_full_active_moe_speed_ab.sh"
INPUT_R5C3C = ROOT / "reports" / "stage6" / "r5c3c_down_weighted_parity_smoke_20260604_130243" / "result.json"


def _lane(candidate_ms: float = 10.0, baseline_ms: float = 12.0) -> dict:
    return {
        "shape": {"tokens": 16, "hidden": 2048, "intermediate": 512, "top_k": 8, "experts": 128},
        "candidate_ms": candidate_ms,
        "baseline_w4a16_ms": baseline_ms,
        "baseline_packed_p3_ms": baseline_ms,
        "numeric_max_abs": 0.0,
        "numeric_rel_l2": 0.0,
        "numeric_cosine": 1.0,
        "route_order_preserved": True,
        "repack_cost_reported": True,
        "fault_injections_detected": True,
        "shape_regression": False,
    }


def _raw_metrics() -> dict:
    return {
        "candidate_name": "r5c4-fixture",
        "implementation": {"kernel": "fixture-only"},
        "passes": {
            "same_scope_ab": True,
            "real_model_weights": True,
            "real_router_outputs": True,
            "candidate_no_active_bf16_shadow": True,
            "candidate_no_reload": True,
            "candidate_no_bf16_weight_materialization": True,
            "candidate_full_active_moe_boundary_timed": True,
            "timing_includes_gateup_swiglu_down_weighted_scatter": True,
        },
        "lanes": {
            "smoke": _lane(),
            "production_shape": _lane(),
        },
    }


def main() -> int:
    failures: list[str] = []
    for path in [ADAPTER, TEMPLATE, VALIDATOR, WRAPPER, INPUT_R5C3C]:
        if not path.exists():
            failures.append(f"missing {path.relative_to(ROOT)}")
    if ADAPTER.exists():
        text = ADAPTER.read_text(encoding="utf-8")
        for needle in [
            "lynn-stage6-r5c4-candidate-v1",
            "candidate_no_bf16_weight_materialization",
            "timing_includes_gateup_swiglu_down_weighted_scatter",
            "promotion_boundary_request",
            "adapter_failures",
        ]:
            if needle not in text:
                failures.append(f"{ADAPTER.relative_to(ROOT)} missing {needle!r}")
    if TEMPLATE.exists():
        template = json.loads(TEMPLATE.read_text(encoding="utf-8"))
        for lane in ["smoke", "production_shape"]:
            if lane not in template.get("lanes", {}):
                failures.append(f"template missing lane {lane}")
        if template.get("passes", {}).get("candidate_no_bf16_weight_materialization") is not True:
            failures.append("template missing required no-materialization pass flag")

    with tempfile.TemporaryDirectory() as tmp_s:
        tmp = Path(tmp_s)
        raw = tmp / "raw.json"
        candidate = tmp / "candidate.json"
        result = tmp / "result.json"
        raw.write_text(json.dumps(_raw_metrics()), encoding="utf-8")
        proc = subprocess.run(
            [sys.executable, str(ADAPTER), "--raw-metrics", str(raw), "--out", str(candidate), "--strict"],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if proc.returncode != 0:
            failures.append(f"adapter rejected valid fixture: {proc.stderr or proc.stdout}")
        data = json.loads(candidate.read_text(encoding="utf-8"))
        if data.get("schema") != "lynn-stage6-r5c4-candidate-v1":
            failures.append("adapter emitted wrong schema")
        if data.get("adapter_failures"):
            failures.append(f"adapter reported failures for valid fixture: {data['adapter_failures']}")
        vproc = subprocess.run(
            [
                sys.executable,
                str(VALIDATOR),
                "--input-r5c3c",
                str(INPUT_R5C3C),
                "--candidate-json",
                str(candidate),
                "--out",
                str(result),
            ],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        vdata = json.loads(result.read_text(encoding="utf-8"))
        if vdata.get("decision") != "PASS_R5C4_FULL_ACTIVE_MOE_PREFILL_SPEED_AB":
            failures.append(f"validator rejected adapter output: {vproc.stderr or vproc.stdout}")

        bad_raw = _raw_metrics()
        bad_raw["passes"]["real_model_weights"] = False
        bad_raw["banked_decode_tps"] = True
        bad_path = tmp / "bad.raw.json"
        bad_candidate = tmp / "bad.candidate.json"
        bad_path.write_text(json.dumps(bad_raw), encoding="utf-8")
        bad_proc = subprocess.run(
            [sys.executable, str(ADAPTER), "--raw-metrics", str(bad_path), "--out", str(bad_candidate), "--strict"],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        bad_data = json.loads(bad_candidate.read_text(encoding="utf-8"))
        if bad_proc.returncode == 0:
            failures.append("strict adapter accepted missing/forbidden evidence")
        failure_text = "\n".join(bad_data.get("adapter_failures", []))
        if "real_model_weights" not in failure_text or "banked_decode_tps" not in failure_text:
            failures.append("bad fixture did not report expected adapter failures")
        wrapper_out = tmp / "wrapper_artifact"
        wproc = subprocess.run(
            [
                "bash",
                str(WRAPPER),
            ],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env={
                "ROOT": str(ROOT),
                "PYTHON_BIN": sys.executable,
                "CANDIDATE_METRICS_JSON": str(raw),
                "OUT_DIR": str(wrapper_out),
                "INPUT_R5C3C": str(INPUT_R5C3C),
                "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
            },
        )
        if wproc.returncode != 0:
            failures.append(f"wrapper rejected CANDIDATE_METRICS_JSON: {wproc.stderr or wproc.stdout}")
        if not (wrapper_out / "candidate.json").exists() or not (wrapper_out / "result.json").exists():
            failures.append("wrapper did not emit candidate/result artifacts for metrics input")

    if failures:
        print("Stage 6 R5-C4 candidate adapter self-test FAILED")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print("Stage 6 R5-C4 candidate adapter self-test PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
