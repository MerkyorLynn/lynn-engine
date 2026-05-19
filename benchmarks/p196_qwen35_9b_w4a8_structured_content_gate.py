#!/usr/bin/env python3
"""P196 - Qwen3.5-9B W4A8 structured-content gate.

Evaluates structured content generation quality across quantization variants.
Protects W4A16 as the mandatory reference; scores W4A8 full and W4A8 gateup
independently.  Quality drift → fall back to A16, no debate.

Consumes pre-computed test results (JSON) from the runner script.
Each test result contains: prompt_id, variant, output_text, structural_pass,
decode_tps, latency_ms.

Verdicts:
  W4A8_CONTENT_GREEN   - W4A8 pass rate ≥ W4A16 × 0.95 AND decode TPS not worse
  W4A8_CONTENT_AMBER   - W4A8 pass rate ≥ W4A16 × 0.80 but below green
  RED_FALLBACK_A16     - W4A8 pass rate < W4A16 × 0.80 or W4A16 itself fails

Usage:
  # From pre-computed results
  python benchmarks/p196_qwen35_9b_w4a8_structured_content_gate.py \\
    --results /path/to/p196_test_results.json \\
    --out /root/autodl-tmp/reports/qwen35_9b/p196_w4a8_content_gate.json

  # With custom thresholds
  python benchmarks/p196_qwen35_9b_w4a8_structured_content_gate.py \\
    --results /path/to/results.json \\
    --green-rate 0.95 --amber-rate 0.80 \\
    --out /path/to/output.json
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


# ─────────────────────────────────────────────────────────────
# Schema / version
# ─────────────────────────────────────────────────────────────
SCHEMA = "lynn-qwen35-9b-p196-w4a8-structured-content-v1"
REPORT_SCHEMA = "lynn-qwen35-9b-p196-w4a8-structured-content-report-v1"

# ─────────────────────────────────────────────────────────────
# Default thresholds
# ─────────────────────────────────────────────────────────────
DEFAULT_GREEN_RATE = 0.95      # W4A8 pass rate ≥ W4A16 × this
DEFAULT_AMBER_RATE = 0.80      # W4A8 pass rate ≥ W4A16 × this
DEFAULT_TPS_GREEN_RATIO = 0.95 # W4A8 TPS ≥ W4A16 TPS × this
DEFAULT_TPS_AMBER_RATIO = 0.80 # W4A8 TPS ≥ W4A16 TPS × this

VARIANTS = ("w4a16", "w4a8_full", "w4a8_gateup")


# ─────────────────────────────────────────────────────────────
# Built-in test case definitions (for runner reference)
# ─────────────────────────────────────────────────────────────
BUILTIN_TEST_CASES = [
    {
        "id": "json_object",
        "prompt": (
            "Generate a JSON object with keys: name (string), age (integer), "
            "city (string). Use realistic values. Output ONLY the JSON, no explanation."
        ),
        "validator": "json_parse",
        "description": "Valid JSON with required keys",
    },
    {
        "id": "json_array",
        "prompt": (
            "Generate a JSON array of exactly 3 objects, each with keys: "
            "id (integer), label (string), score (float 0-1). Output ONLY the JSON."
        ),
        "validator": "json_array_parse",
        "description": "Valid JSON array with correct element count",
    },
    {
        "id": "python_function",
        "prompt": (
            "Write a Python function `merge_sorted(a: list[int], b: list[int]) -> list[int]` "
            "that merges two sorted lists into one sorted list. Output ONLY the function, "
            "no tests or explanation."
        ),
        "validator": "python_syntax",
        "description": "Syntactically valid Python function",
    },
    {
        "id": "markdown_table",
        "prompt": (
            "Create a markdown table with columns: Fruit, Color, Price ($). "
            "Include exactly 4 rows of data. Output ONLY the table."
        ),
        "validator": "markdown_table",
        "description": "Valid markdown table with correct structure",
    },
    {
        "id": "yaml_config",
        "prompt": (
            "Generate a YAML configuration for a web server with keys: "
            "server (host, port), logging (level, file), database (driver, host, name). "
            "Output ONLY the YAML."
        ),
        "validator": "yaml_parse",
        "description": "Valid YAML with required structure",
    },
    {
        "id": "csv_data",
        "prompt": (
            "Generate CSV data with header: id,name,score,grade. "
            "Include exactly 5 data rows. Output ONLY the CSV."
        ),
        "validator": "csv_parse",
        "description": "Valid CSV with correct column count",
    },
    {
        "id": "key_value_pairs",
        "prompt": (
            "Output exactly 5 key-value pairs, one per line, format: key=value. "
            "Keys: hostname, port, debug, max_connections, timeout. "
            "Output ONLY the pairs."
        ),
        "validator": "key_value_lines",
        "description": "Correctly formatted key=value lines",
    },
    {
        "id": "numbered_list",
        "prompt": (
            "Create a numbered list of exactly 5 programming best practices. "
            "Format: 1. Practice description. Output ONLY the list."
        ),
        "validator": "numbered_list",
        "description": "Correctly formatted numbered list",
    },
    {
        "id": "regex_pattern",
        "prompt": (
            "Output 3 valid Python regular expressions (re.compile compatible), "
            "one per line, for: email, phone (US), IPv4. "
            "Each line should be just the pattern string."
        ),
        "validator": "regex_lines",
        "description": "Valid regex patterns, one per line",
    },
    {
        "id": "json_nested",
        "prompt": (
            "Generate a JSON object representing a company with: "
            "name (string), founded (integer), departments (array of objects, "
            "each with name:string, head_count:integer, budget:float). "
            "Include exactly 3 departments. Output ONLY the JSON."
        ),
        "validator": "json_nested",
        "description": "Nested JSON with correct types and array length",
    },
]


# ─────────────────────────────────────────────────────────────
# Validators (offline, no model needed)
# ─────────────────────────────────────────────────────────────
def validate_json_parse(text: str) -> bool:
    try:
        obj = json.loads(text.strip())
        return isinstance(obj, dict) and len(obj) >= 2
    except (json.JSONDecodeError, ValueError):
        return False


def validate_json_array_parse(text: str) -> bool:
    try:
        obj = json.loads(text.strip())
        return isinstance(obj, list) and len(obj) == 3
    except (json.JSONDecodeError, ValueError):
        return False


def validate_python_syntax(text: str) -> bool:
    text = text.strip()
    if "```" in text:
        parts = text.split("```")
        text = parts[1] if len(parts) >= 2 else text
        if text.startswith("python"):
            text = text[len("python"):].strip()
    try:
        compile(text, "<p196>", "exec")
        return "def " in text
    except SyntaxError:
        return False


def validate_markdown_table(text: str) -> bool:
    lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
    if len(lines) < 3:
        return False
    header = lines[0]
    separator = lines[1]
    if "|" not in header or "|" not in separator:
        return False
    if "---" not in separator and "===" not in separator:
        return False
    data_rows = lines[2:]
    return len(data_rows) >= 3


def validate_yaml_parse(text: str) -> bool:
    try:
        import yaml
        obj = yaml.safe_load(text.strip())
        return isinstance(obj, dict) and len(obj) >= 2
    except Exception:
        lines = [l.strip() for l in text.strip().split("\n") if l.strip() and not l.strip().startswith("#")]
        has_server = any("server" in l.lower() for l in lines)
        has_colon = any(":" in l for l in lines)
        return has_server and has_colon and len(lines) >= 4


def validate_csv_parse(text: str) -> bool:
    lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
    if len(lines) < 3:
        return False
    header_cols = len(lines[0].split(","))
    if header_cols < 3:
        return False
    data_rows = [l for l in lines[1:] if not l.startswith("---")]
    return len(data_rows) >= 3


def validate_key_value_lines(text: str) -> bool:
    lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
    kv_count = sum(1 for l in lines if "=" in l)
    return kv_count >= 4


def validate_numbered_list(text: str) -> bool:
    lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
    numbered = sum(1 for l in lines if l and l[0].isdigit() and "." in l[:4])
    return numbered >= 4


def validate_regex_lines(text: str) -> bool:
    import re
    lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
    if len(lines) < 2:
        return False
    valid = 0
    for line in lines[:5]:
        try:
            re.compile(line)
            valid += 1
        except re.error:
            pass
    return valid >= 2


def validate_json_nested(text: str) -> bool:
    try:
        obj = json.loads(text.strip())
        if not isinstance(obj, dict):
            return False
        depts = obj.get("departments", [])
        return isinstance(depts, list) and len(depts) == 3
    except (json.JSONDecodeError, ValueError):
        return False


VALIDATORS = {
    "json_parse": validate_json_parse,
    "json_array_parse": validate_json_array_parse,
    "python_syntax": validate_python_syntax,
    "markdown_table": validate_markdown_table,
    "yaml_parse": validate_yaml_parse,
    "csv_parse": validate_csv_parse,
    "key_value_lines": validate_key_value_lines,
    "numbered_list": validate_numbered_list,
    "regex_lines": validate_regex_lines,
    "json_nested": validate_json_nested,
}


# ─────────────────────────────────────────────────────────────
# Aggregate per-variant metrics
# ─────────────────────────────────────────────────────────────
def _variant_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)
    if total == 0:
        return {
            "total": 0,
            "pass_count": 0,
            "pass_rate": 0.0,
            "failure_ids": [],
            "decode_tps_values": [],
            "decode_tps_mean": None,
            "decode_tps_median": None,
            "latency_ms_mean": None,
        }

    passed = [r for r in results if r.get("structural_pass")]
    failed = [r for r in results if not r.get("structural_pass")]
    tps_values = [r["decode_tps"] for r in results if r.get("decode_tps") is not None]
    lat_values = [r["latency_ms"] for r in results if r.get("latency_ms") is not None]

    return {
        "total": total,
        "pass_count": len(passed),
        "pass_rate": len(passed) / total,
        "failure_ids": [r["prompt_id"] for r in failed],
        "decode_tps_values": tps_values,
        "decode_tps_mean": statistics.mean(tps_values) if tps_values else None,
        "decode_tps_median": statistics.median(tps_values) if tps_values else None,
        "latency_ms_mean": statistics.mean(lat_values) if lat_values else None,
    }


# ─────────────────────────────────────────────────────────────
# Gate logic
# ─────────────────────────────────────────────────────────────
def run_gate(results_path: str, thresholds: dict[str, float]) -> dict[str, Any]:
    results = json.loads(Path(results_path).read_text(encoding="utf-8"))

    if not results:
        return {
            "schema": REPORT_SCHEMA,
            "decision": "RED_FALLBACK_A16",
            "reasons": ["No test results provided"],
        }

    # Group by variant
    by_variant: dict[str, list[dict[str, Any]]] = {}
    for r in results:
        v = r.get("variant", "unknown")
        by_variant.setdefault(v, []).append(r)

    # Validate W4A16 reference
    w4a16 = by_variant.get("w4a16", [])
    w4a16m = _variant_metrics(w4a16)
    if w4a16m["total"] == 0:
        return {
            "schema": REPORT_SCHEMA,
            "decision": "RED_FALLBACK_A16",
            "reasons": ["No W4A16 reference results; cannot establish baseline"],
        }
    if w4a16m["pass_rate"] < 1.0:
        return {
            "schema": REPORT_SCHEMA,
            "decision": "RED_FALLBACK_A16",
            "reasons": [
                f"W4A16 reference pass rate {w4a16m['pass_rate']:.1%} < 100%; "
                f"failures: {w4a16m['failure_ids']}"
            ],
            "per_variant": {"w4a16": w4a16m},
        }

    # Evaluate W4A8 variants
    reasons: list[str] = []
    worst_verdict = "W4A8_CONTENT_GREEN"
    per_variant: dict[str, Any] = {"w4a16": w4a16m}

    for variant in ("w4a8_full", "w4a8_gateup"):
        vm = _variant_metrics(by_variant.get(variant, []))
        per_variant[variant] = vm

        if vm["total"] == 0:
            reasons.append(f"{variant}: no results (skipped)")
            continue

        rate_ratio = vm["pass_rate"] / w4a16m["pass_rate"]
        tps_ratio = (
            (vm["decode_tps_mean"] / w4a16m["decode_tps_mean"])
            if vm["decode_tps_mean"] and w4a16m["decode_tps_mean"]
            else None
        )

        per_variant[variant]["rate_vs_w4a16"] = rate_ratio
        per_variant[variant]["tps_vs_w4a16"] = tps_ratio

        if rate_ratio < thresholds["amber_rate"]:
            worst_verdict = "RED_FALLBACK_A16"
            reasons.append(
                f"{variant}: pass rate {vm['pass_rate']:.1%} = "
                f"{rate_ratio:.1%} of W4A16 < amber {thresholds['amber_rate']:.0%}"
            )
        elif rate_ratio < thresholds["green_rate"]:
            if worst_verdict != "RED_FALLBACK_A16":
                worst_verdict = "W4A8_CONTENT_AMBER"
            reasons.append(
                f"{variant}: pass rate {vm['pass_rate']:.1%} = "
                f"{rate_ratio:.1%} of W4A16 < green {thresholds['green_rate']:.0%}"
            )
        elif tps_ratio is not None and tps_ratio < thresholds["tps_green_ratio"]:
            if worst_verdict != "RED_FALLBACK_A16":
                worst_verdict = "W4A8_CONTENT_AMBER"
            reasons.append(
                f"{variant}: decode TPS ratio {tps_ratio:.2f} < "
                f"green {thresholds['tps_green_ratio']:.2f}"
            )
        else:
            reasons.append(f"{variant}: PASS")

    if not reasons:
        reasons.append("all variants pass")

    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "created": datetime.now(timezone.utc).isoformat(),
        "thresholds": thresholds,
        "decision": worst_verdict,
        "reasons": reasons,
        "per_variant": per_variant,
        "per_prompt": results,
    }
    return report


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(
        description="P196 Qwen3.5-9B W4A8 structured-content gate"
    )
    ap.add_argument(
        "--results", required=False,
        help="Path to p196 test results JSON (array of {prompt_id, variant, output_text, structural_pass, decode_tps, latency_ms})"
    )
    ap.add_argument(
        "--out", required=False,
        help="Output report path"
    )
    ap.add_argument(
        "--green-rate", type=float, default=DEFAULT_GREEN_RATE,
        help=f"W4A8 pass rate must be ≥ W4A16 × this for GREEN (default: {DEFAULT_GREEN_RATE})"
    )
    ap.add_argument(
        "--amber-rate", type=float, default=DEFAULT_AMBER_RATE,
        help=f"W4A8 pass rate must be ≥ W4A16 × this to avoid RED (default: {DEFAULT_AMBER_RATE})"
    )
    ap.add_argument(
        "--tps-green", type=float, default=DEFAULT_TPS_GREEN_RATIO,
        help=f"W4A8 TPS must be ≥ W4A16 × this for GREEN (default: {DEFAULT_TPS_GREEN_RATIO})"
    )
    ap.add_argument(
        "--tps-amber", type=float, default=DEFAULT_TPS_AMBER_RATIO,
        help=f"W4A8 TPS must be ≥ W4A16 × this for AMBER (default: {DEFAULT_TPS_AMBER_RATIO})"
    )
    ap.add_argument(
        "--list-tests", action="store_true",
        help="Print built-in test case definitions and exit"
    )
    args = ap.parse_args()

    if args.list_tests:
        print(json.dumps(BUILTIN_TEST_CASES, indent=2))
        return

    if not args.results or not args.out:
        ap.error("--results and --out are required (unless --list-tests)")

    thresholds = {
        "green_rate": args.green_rate,
        "amber_rate": args.amber_rate,
        "tps_green_ratio": args.tps_green,
        "tps_amber_ratio": args.tps_amber,
    }

    report = run_gate(args.results, thresholds)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8")

    # Print verdict to stdout
    print(f"P196 decision: {report['decision']}")
    for r in report.get("reasons", []):
        print(f"  {r}")

    per_var = report.get("per_variant", {})
    for vname in VARIANTS:
        vm = per_var.get(vname)
        if vm:
            rate_str = f"{vm['pass_rate']:.1%}" if vm.get("pass_count") is not None else "N/A"
            tps_str = f"{vm['decode_tps_mean']:.2f}" if vm.get("decode_tps_mean") else "N/A"
            ratio_str = ""
            if vm.get("rate_vs_w4a16") is not None:
                ratio_str = f" (rate_ratio={vm['rate_vs_w4a16']:.2f})"
            print(f"  {vname}: pass {vm.get('pass_count',0)}/{vm.get('total',0)} "
                  f"= {rate_str}, TPS_mean={tps_str}{ratio_str}")

    print(f"\nReport: {args.out}")


if __name__ == "__main__":
    main()
