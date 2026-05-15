#!/usr/bin/env python3
"""P3-K regression gate for packed NVFP4 bridge reports.

This gate intentionally consumes existing JSON reports instead of rerunning all
benchmarks. It is the lightweight checkpoint to run after changing packed
kernel internals: all P3 bridge reports must remain within their correctness
thresholds before P4/P5 performance work can be trusted.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def _check_cmp(report: dict[str, Any], path: str, *, cosine: float, rel_l2: float) -> dict[str, Any]:
    cur: Any = report
    for part in path.split("."):
        cur = cur[part]
    ok = cur["cosine"] >= cosine and cur["rel_l2"] <= rel_l2
    return {
        "path": path,
        "cosine": cur["cosine"],
        "rel_l2": cur["rel_l2"],
        "thresholds": {"cosine": cosine, "rel_l2": rel_l2},
        "verdict": "PASS" if ok else "FAIL",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports-dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    reports_dir = Path(args.reports_dir)
    checks: list[dict[str, Any]] = []

    p3g = _load(reports_dir / "p3_nvfp4_layer_decode_packed_probe.json")
    checks.append({
        "name": "P3-G full layer bridge",
        "report": "p3_nvfp4_layer_decode_packed_probe.json",
        "verdict": p3g["verdict"],
        "topk_exact_match": p3g["router"]["topk_exact_match"],
        "comparisons": [
            _check_cmp(p3g, "comparisons.final_layer_output", cosine=0.999, rel_l2=0.02),
            _check_cmp(p3g, "comparisons.moe_out_same_h_moe_norm", cosine=0.999, rel_l2=0.02),
        ],
    })

    p3h = _load(reports_dir / "p3_nvfp4_layer_decode_multiseed_probe.json")
    checks.append({
        "name": "P3-H multiseed bridge",
        "report": "p3_nvfp4_layer_decode_multiseed_probe.json",
        "verdict": p3h["verdict"],
        "summary": p3h["summary"],
        "requirements": {
            "pass_count_equals_seed_count": p3h["summary"]["pass_count"] == p3h["summary"]["seed_count"],
            "topk_set_count_equals_seed_count": p3h["summary"]["topk_set_count"] == p3h["summary"]["seed_count"],
            "min_final_cosine_ge_0_999": p3h["summary"]["min_final_cosine"] >= 0.999,
            "max_final_rel_l2_le_0_02": p3h["summary"]["max_final_rel_l2"] <= 0.02,
        },
    })

    p3j = _load(reports_dir / "p3_nvfp4_layer_decode_packed_shared_probe.json")
    checks.append({
        "name": "P3-J packed shared expert bridge",
        "report": "p3_nvfp4_layer_decode_packed_shared_probe.json",
        "verdict": p3j["verdict"],
        "shared_expert": p3j["packed_components"]["shared_expert"],
        "comparisons": [
            _check_cmp(p3j, "comparisons.final_layer_output", cosine=0.999, rel_l2=0.02),
            _check_cmp(p3j, "comparisons.moe_out_same_h_moe_norm", cosine=0.999, rel_l2=0.02),
        ],
    })

    p3i = _load(reports_dir / "p3_nvfp4_cross_layer_bridge_summary.json")
    checks.append({
        "name": "P3-I cross-layer bridge",
        "report": "p3_nvfp4_cross_layer_bridge_summary.json",
        "verdict": p3i["verdict"],
        "summary": {
            "layers": p3i["layers"],
            "min_cosine": p3i["min_cosine"],
            "max_rel_l2": p3i["max_rel_l2"],
        },
        "requirements": {
            "all_layers_pass": p3i["verdict"] == "PASS",
            "min_cosine_ge_0_999": p3i["min_cosine"] >= 0.999,
            "max_rel_l2_le_0_02": p3i["max_rel_l2"] <= 0.02,
            "topk_all_exact": all(item["topk_exact_match"] for item in p3i["items"]),
        },
    })

    def check_ok(check: dict[str, Any]) -> bool:
        if check.get("verdict") != "PASS":
            return False
        if "topk_exact_match" in check and not check["topk_exact_match"]:
            return False
        if "shared_expert" in check and check["shared_expert"] != "packed-nvfp4-bridge":
            return False
        if any(c["verdict"] != "PASS" for c in check.get("comparisons", [])):
            return False
        if any(v is False for v in check.get("requirements", {}).values()):
            return False
        return True

    result = {
        "schema_version": "lynn-engine-p3-regression-gate-v1",
        "reports_dir": str(reports_dir),
        "checks": checks,
        "verdict": "PASS" if all(check_ok(check) for check in checks) else "FAIL",
        "notes": [
            "P3-K is a report-level regression gate for packed NVFP4 bridge correctness.",
            "Run this after changing packed kernel internals before trusting P4/P5 performance numbers.",
        ],
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["verdict"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
